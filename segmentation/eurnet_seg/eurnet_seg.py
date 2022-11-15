import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import eurnet_seg.td_custom as td
from eurnet_seg.td_custom import data as td_data

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
import os
from mmcv_custom import load_checkpoint
from eurnet_seg.layers.message_passing import GatedRelationalMessagePassing
from eurnet_seg.layers.graph_construction import *


class FFN(nn.Module):
    """
    Module for Feed-Forward Network.

    Args:
        input_dim (int): Dimension of input representations.
        hidden_dim (int): Dimension of hidden representations.
        output_dim (int): Dimension of output representations.
        act_layer (nn.Module): Activation layer.
        drop_rate (float): Dropout rate.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if drop_rate:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout(x)

        return x

class EurNetBlock(nn.Module):
    """
    Building Block of EurNet.

    Args:
        input_dim (int): Dimension of input representations.
        num_relation (int): Number of all considered relations.
        context_relation (bool): Whether to add an extra global context relation.
        context_sizes (list of int): Sizes of context convolutional kernels.
        use_all_context (bool): Whether to use all context outputs as relations.
        ffn_ratio (float): Ratio of FFN hidden dimension to stage hidden dimension.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(self, input_dim, num_relation, 
                 context_relation=False, context_sizes=[3, 5], use_all_context=False, ffn_ratio=4.,
                 drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.ffn_ratio = ffn_ratio

        self.norm1 = norm_layer(input_dim)
        self.graph_conv = GatedRelationalMessagePassing(input_dim, input_dim, num_relation,
                                                           context_relation=context_relation, context_sizes=context_sizes,
                                                           use_all_context=use_all_context, act_layer=act_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(input_dim)
        self.proj = nn.Linear(input_dim, input_dim)
        ffn_hidden_dim = int(input_dim * ffn_ratio)
        self.ffn = FFN(input_dim, hidden_dim=ffn_hidden_dim, act_layer=act_layer, drop_rate=drop_rate)

    def forward(self, graph, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.graph_conv(graph, [x, H, W])
        x = self.proj(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    Args:
        input_dim (int): Dimension of input representations.
        output_dim (int): Dimension of output representations.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.downsample = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = norm_layer(4 * input_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        if L != H * W:
            raise ValueError("Patch number %d doesn't match with patch resolution (%d, %d)." % (L, H, W))
        
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)  # [B, H/2 * W/2, 4C]

        x = self.norm(x)
        x = self.downsample(x)  # [B, H/2 * W/2, 2C]

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding adapted from: https://github.com/microsoft/Focal-Transformer/blob/main/classification/focal_transformer.py

    Args:
        patch_size (int): Patch token size.
        input_dim (int): Input dimension of pixels.
        hidden_dim (int): Dimension of linear projection outputs.
        norm_layer (nn.Module): Normalization layer.
        use_conv_embed (bool): Whether to use overlapped convolutional embedding layer.
        is_stem (bool): Whether the patch embedding layer is used as stem.
    """

    def __init__(self, patch_size=4, input_dim=3, hidden_dim=96, norm_layer=None,
                 use_conv_embed=False, is_stem=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            if is_stem:
                kernel_size, padding, stride = 7, 2, 4
            else:
                kernel_size, padding, stride = 3, 1, 2
            self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(hidden_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # [B, Hp * Wp, D]
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.hidden_dim, Wh, Ww)

        return x


class EurNetStage(nn.Module):
    """
    The basic module for one stage of eurnet.

    Args:
        graph_construction_model (nn.Module): Graph construction model of the stage.
        input_dim (int): Dimension of input representations.
        output_dim (int): Dimension of output representations.
        num_relation (int): Number of all considered relations.
        depth (int): Number of blocks.
        ffn_ratio (float): Ratio of FFN hidden dimension to stage hidden dimension.
        context_relation (bool): Whether to add an extra global context relation.
        context_sizes (list of int): Sizes of context convolutional kernels.
        use_all_context (bool): Whether to use all context outputs as relations.
        drop_rate (float): Dropout rate.
        drop_path_rate (float or tuple[float]): Stochastic depth rate.
        norm_layer (nn.Module or None): Normalization layer.
        downsample (nn.Module or None): Downsampling layer at the end of stage.
    """

    def __init__(self, graph_construction_model, input_dim, output_dim, num_relation,
                 depth, ffn_ratio, context_relation=False, context_sizes=[3, 5], use_all_context=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.graph_construction_model = graph_construction_model
        self.input_dim = input_dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            EurNetBlock(input_dim=input_dim, num_relation=num_relation, context_relation=context_relation, 
                      context_sizes=context_sizes, use_all_context=use_all_context, ffn_ratio=ffn_ratio, 
                      drop_rate=drop_rate, drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                      norm_layer=norm_layer)
            for i in range(depth)])

        # downsampling layer
        if downsample is not None:
            if downsample == PatchMerging:
                self.downsample = downsample(input_dim=input_dim, output_dim=output_dim,
                                             norm_layer=norm_layer)
            else:
                raise ValueError("Downsampling scheme must be either `PatchMerging` or `PatchEmbed`.")
        else:
            self.downsample = None

    def forward(self, x, H, W):
        B, L, C = x.shape
        num_nodes = [L] * B
        node_feature = x.flatten(0, 1)
        edge_list = torch.zeros((0, 3), dtype=torch.long, device=x.device)
        num_edges = [0] * B
        graph = td_data.PackedGraph(num_nodes=num_nodes, node_feature=node_feature,
                                    edge_list=edge_list, num_edges=num_edges)
        graph = self.graph_construction_model(graph, H, W)
        x = graph.node_feature.view(graph.batch_size, -1, graph.node_feature.shape[-1])

        # message passing and feature transformation
        for block in self.blocks:
            x = block(graph, x, H, W)
        if x.shape[1] == L + 1:  # with a prepended virtual node
            virtual_x, x = x[:, 0, :], x[:, 1:, :]
        elif x.shape[1] == L:  # without virtual node
            virtual_x = None
        else:
            raise ValueError("Patch number should be either %d or %d." % (L, L + 1))

        # downsampling
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H+1) // 2, (W+1) // 2
            return virtual_x, x, H, W, x_down, Wh, Ww
        else:
            return virtual_x, x, H, W, x, H, W


@BACKBONES.register_module()
class EurNet(nn.Module):
    """
    Args:
        img_size (int or (int, int)): Input image size.
        patch_size (int or (int, int)): Patch size.
        input_dim (int): Input dimension of pixels.
        hidden_dim (int or list of int): First-stage hidden dimension or hidden dimensions of all stages.
        depths (list of int): Number of eurnet blocks in each stage.
        medium_stages (list of int): Stages to use medium range edges.
        edge_types (list of str): Types of edges to construct between patches.
        num_neighbors (int or list of int): Number of semantic neighbors in different stages.
        virtual_node (bool): If True, add a virtual node for whole-image representation.
        context_relation (bool): Whether to add an extra relation for surrounding global context.
        context_sizes (list of int): Sizes of context convolutional kernels.
        use_all_context (bool):  Whether to use all context outputs as relations.
        ffn_ratio (float): Ratio of FFN hidden dimension to stage hidden dimension.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        ape (bool): If True, add absolute position embedding to the patch embedding.
        patch_norm (bool): If True, add normalization after patch embedding.

        ### args for segmentation
        out_indices (tuple of int): the indices of the stages whose outputs are fed into FPN.
        frozen_stages (int): freeze the parameters of the stages below this value, by default no parameter freezing. 
        pretrained (str): the checkpoint file.
        window_length (int): size of window when calculating pair-wise distance.
        dilation (int): dilation when calculating window-pair-wise distance.
        ###
    """

    edge_type2num_relation = {"short": 4, "medium": 1, "long": 1}

    def __init__(self, img_size=224, patch_size=4, input_dim=3,
                 hidden_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], medium_stages=[1, 2, 3],
                 edge_types=["short", "medium", "long"], num_neighbors=[12, 12, 12, 12],
                 virtual_node=True, context_relation=True, context_sizes=[9, 11, 13], use_all_context=True,
                 ffn_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 out_indices=(0,1,2,3), frozen_stages=-1, pretrained=None, window_length=144, dilation=2, **kwargs):
        super().__init__()

        self.depths = depths
        self.num_stage = len(depths)
        self.hidden_dim_first = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[0]
        self.hidden_dim_last = int(hidden_dim * 2 ** (self.num_stage - 1)) \
            if isinstance(hidden_dim, int) else hidden_dim[-1]
        if virtual_node and "long" not in edge_types:
            edge_types.append("long")
        if not virtual_node and "long" in edge_types:
            edge_types = [t for t in edge_types if t != "long"]

        num_relation = sum([self.edge_type2num_relation[t] for t in edge_types if t != "medium"])
        self.num_relations = [num_relation] * self.num_stage
        if "medium" in edge_types:
            for stage_id in medium_stages:
                self.num_relations[stage_id] += self.edge_type2num_relation["medium"]
        self.virtual_node = virtual_node

        self.ffn_ratio = ffn_ratio
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        self.patch_size = patch_size
        patch_resolution = [img_size // patch_size, img_size // patch_size]
        num_patch = patch_resolution[0] * patch_resolution[1]  
        self.patch_embed = PatchEmbed(patch_size=patch_size, input_dim=input_dim,
                                      hidden_dim=self.hidden_dim_first,
                                      norm_layer=norm_layer if self.patch_norm else None, is_stem=True)

        if self.ape:
            ## this is pretrained size, would be interpolated.
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim_first, patch_resolution[0], patch_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if drop_rate:
            self.pos_drop = nn.Dropout(drop_rate)
        else:
            self.pos_drop = None

        # graph construction models for all stages
        if isinstance(num_neighbors, int):
            num_neighbors = [num_neighbors] * self.num_stage

        self.patch_resolutions = [patch_resolution[0] // (2 ** stage_id) for stage_id in range(self.num_stage)]
        self.window_lengths = [window_length for _ in range(self.num_stage)]
        self.hidden_dims = [int(hidden_dim * 2 ** stage_id) for stage_id in range(self.num_stage)] \
            if isinstance(hidden_dim, int) else hidden_dim

        graph_construction_models = []
        for stage_id, (resolution, dim, k) in enumerate(zip(self.patch_resolutions, self.hidden_dims, num_neighbors)):
            node_layers = [ExtraVirtualNode(dim)] if virtual_node else None
            edge_layers = []
            for edge_type in edge_types:
                if edge_type == "short":
                    edge_layers.append(ShortRangeEdge(resolution))
                elif edge_type == "medium":
                    if stage_id in medium_stages:
                        edge_layers.append(MediumRangeEdge(self.hidden_dims[stage_id], resolution, k=k, 
                                                                window_length=self.window_lengths[stage_id],
                                                                dilation=dilation))
                elif edge_type == "long":
                    edge_layers.append(LongRangeEdge(resolution))
                else:
                    raise ValueError("Edge type `%s` is not defined." % edge_type)

            graph_construction_model = GraphConstruction(node_layers, edge_layers)
            graph_construction_models.append(graph_construction_model)

        # build model for all stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.stages = nn.ModuleList()
        for stage_id in range(self.num_stage):
            stage = EurNetStage(graph_construction_model=graph_construction_models[stage_id],
                              input_dim=self.hidden_dims[stage_id], output_dim=self.hidden_dims[min(stage_id + 1, self.num_stage - 1)],
                              num_relation=self.num_relations[stage_id],
                              depth=depths[stage_id], ffn_ratio=self.ffn_ratio, 
                              context_relation=context_relation, context_sizes=context_sizes, use_all_context=use_all_context,
                              drop_rate=drop_rate, drop_path_rate=dpr[sum(depths[:stage_id]):sum(depths[:stage_id + 1])],
                              norm_layer=norm_layer, downsample=PatchMerging if (stage_id < self.num_stage - 1) else None,)
            self.stages.append(stage)

        for i_layer in out_indices:
            layer = norm_layer(self.hidden_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.init_weights(pretrained=pretrained)
        self._freeze_stages()
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            if  os.path.isfile(pretrained):
                logger.info(f"checkpoint path {pretrained} is valid. loaded!")
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                logger.info(f"checkpoint path {pretrained} is invalid, we skip it and initialize net randomly")
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    def forward(self, x):
        x = self.patch_embed(x) #  [N, D, Wh, Ww]
        B, _, H, W = x.shape

        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            absolute_pos_embed = self.absolute_pos_embed.resize(1, self.patch_resolutions[0], 
                                        self.patch_resolutions[0], self.hidden_dim_first).permute(0,3,1,2)
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = x + absolute_pos_embed
        
        x = x.flatten(2).transpose(1, 2) # [N, Wh * Ww, D]
        if self.pos_drop is not None:
            x = self.pos_drop(x)

        outs = []
        for i, stage in enumerate(self.stages):
            virtual_x, x_out, H, W, x, Wh, Ww = stage(x, Wh, Ww) # input with size

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.hidden_dims[i]).permute(0,3,1,2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()
