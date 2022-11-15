import math

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from eurnet_seg.td_custom import layers


class GatedRelationalMessagePassing(layers.MessagePassingBase):
    """
    Gated Relational Message Passing layer.
    Inherit from the `MessagePassingBase` module of TorchDrug,
    which sequentially performs `message`, `aggregate` and `combine`.

    Args:
        input_dim (int): Dimension of input representations.
        output_dim (int): Dimension of output representations.
        num_relation (int): Number of all considered relations.
        context_relation (bool): Whether to add an extra global context relation.
        context_sizes (list of int): Sizes of context convolutional kernels.
        use_all_context (bool): Whether use all context outputs as relations.
        norm_layer (nn.Module or None): Normalization layer.
        act_layer (nn.Module or None): Activation layer.
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation,
                 context_relation=False, context_sizes=[3, 5], use_all_context=True, 
                 norm_layer=None, act_layer=None):
        super(GatedRelationalMessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.context_relation = context_relation
        self.use_all_context = use_all_context

        if norm_layer:
            self.norm = norm_layer(output_dim)
        else:
            self.norm = None
        if act_layer:
            self.activation = act_layer()
        else:
            self.activation = None

        if context_relation:
            self.context_layers = nn.ModuleList()
            for context_size in context_sizes:
                context_layer = nn.Sequential(
                    nn.Conv2d(output_dim, output_dim, kernel_size=context_size, stride=1,
                              groups=output_dim, padding=context_size // 2, bias=False),
                    act_layer()
                )
                self.context_layers.append(context_layer)


        self.self_loop = nn.Linear(input_dim, output_dim)
        self.message_linear = nn.Linear(input_dim, output_dim)
        num_context_relation = len(context_sizes) if use_all_context else 1
        num_weight = num_relation + num_context_relation if context_relation else num_relation
        self.weight_linear = nn.Linear(input_dim, num_weight)
        self.linear = nn.Conv1d(num_relation * output_dim, num_relation * output_dim,
                                kernel_size=1, groups=num_relation * output_dim, bias=False)
        self.gate_linear = nn.Linear(output_dim, output_dim)

    def message(self, graph, input):
        input, H, W = input
        B, L, C = input.shape
        message_input = self.message_linear(input)
        if L == (H * W + 1):
            message_input[:, 0, :] = message_input[:, 1:, :].mean(1)

        message_input = message_input.flatten(0, 1)

        node_in = graph.edge_list[:, 0]
        message = message_input[node_in]

        return [message, message_input, H, W]

    def aggregate(self, graph, message):
        message, message_input, H, W = message
        if graph.num_relation != self.num_relation:
            raise ValueError("Mismatch between the relation number of the graph and the model.")

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)

        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) / \
                     (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) + self.eps)
        update = update.view(graph.batch_size, -1, self.num_relation * self.output_dim).transpose(1, 2)  # [N, RC, L]
        update = self.activation(self.linear(update))
        update = update.transpose(1, 2).view(graph.batch_size, -1, self.num_relation, self.output_dim)  # [N, L, R, C]

        if self.context_relation:
            B, L, R, C = update.shape
            message_input = message_input.view(B, L, C)
            if L == (H * W + 1):
                context_update = message_input[:, 1:, :].transpose(1, 2).view(B, C, H, W)
            else:
                context_update = message_input.transpose(1, 2).view(B, C, H, W)
            context_updates = []
            for context_layer in self.context_layers:
                context_update = context_layer(context_update)
                _context_update = context_update.flatten(2).transpose(1, 2)
                if L == (H * W + 1):
                    _context_update = torch.cat([message_input[:, :1, :], _context_update], dim=1)
                context_updates.append(_context_update)
            if self.use_all_context:
                context_update_ = torch.stack(context_updates, dim=-2)
            else:
                context_update_ = context_updates[-1].unsqueeze(-2)
            update = torch.cat([update, context_update_], dim=-2)  # [N, L, R + R_context, C]

        return [update, H, W]

    def combine(self, input, update):
        update, H, W = update
        input, H, W = input
        B, L, C = input.shape
        relation_weight = self.weight_linear(input).unsqueeze(-1)  # [N, L, R, 1]
        update = (update * relation_weight).sum(dim=2)  # [N, L, C]
        gate = self.gate_linear(update)
        output = self.self_loop(input) * gate
        if self.norm:
            output = self.norm(output)

        return output
