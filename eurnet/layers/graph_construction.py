import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from eurnet.utils import position_embed


class ExtraVirtualNode(nn.Module):
    """
    Node construction model for adding extra virtual node.

    Parameters:
        hidden_dim (int): Hidden dimension.
    """

    def __init__(self, hidden_dim):
        super(ExtraVirtualNode, self).__init__()
        self.virtual_node_embed = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, graph):
        """
        Augment each graph with an extra virtual node.

        Parameters:
            graph (data.Batch): A batch of graphs.

        Returns:
            data.Batch: A batch of augmented graphs.
        """
        graph.num_nodes = graph.num_nodes + len(graph)
        virtual_node_batch = torch.arange(len(graph), device=graph.x.device).unsqueeze(-1)
        new_batch = torch.cat([virtual_node_batch, graph.batch.view(len(graph), -1)], dim=1).flatten()
        graph.batch = new_batch
        virtual_node_feature = self.virtual_node_embed.unsqueeze(0).repeat(len(graph), 1, 1)
        x = graph.x.view(len(graph), -1, graph.num_node_features)
        new_x = torch.cat([virtual_node_feature, x], dim=1).flatten(0, 1)
        graph.x = new_x

        return graph


class ShortRangeEdge(nn.Module):
    """
    Edge construction model for spatially adjacent patches (short-range edges).

    Parameters:
        resolution (int): Resolution of feature map.
    """

    shift2mask_dim = {1: -1, -1: 0}
    dims = [1, 2]
    shift_dim2relation = {(1, 1): 0, (-1, 1): 1, (1, 2): 2, (-1, 2): 3}
    shift_dim_pair2relation = {((1, 1), (1, 2)): 4, ((1, 1), (-1, 2)): 5, ((-1, 1), (1, 2)): 6, ((-1, 1), (-1, 2)): 7}

    def __init__(self, resolution):
        super(ShortRangeEdge, self).__init__()
        self.resolution = resolution
        self.num_patch = resolution ** 2

    def adjacent_edge_list(self, node_index, shift=1, dim=1):
        if shift not in self.shift2mask_dim:
            raise ValueError("`shift` must be either 1 or -1.")
        if dim not in self.dims:
            raise ValueError("`dim` must be either 1 or 2.")

        node_index_ = node_index.clone()
        if dim == 1:
            node_index_[:, self.shift2mask_dim[shift], :] = -1
        else:
            node_index_[:, :, self.shift2mask_dim[shift]] = -1
        node_index_ = torch.roll(node_index_, shift, dim)
        mask = node_index_.flatten() >= 0
        edge_type = torch.ones(mask.sum(), dtype=torch.long, device=node_index.device) \
                    * self.shift_dim2relation[(shift, dim)]
        edge_index = torch.stack([node_index_.flatten()[mask], node_index.flatten()[mask]], dim=0)

        return edge_index, edge_type

    def two_hop_edge_list(self, node_index, shift1=1, dim1=1, shift2=1, dim2=2):
        if shift1 not in self.shift2mask_dim:
            raise ValueError("`shift1` must be either 1 or -1.")
        if shift2 not in self.shift2mask_dim:
            raise ValueError("`shift2` must be either 1 or -1.")
        if dim1 not in self.dims:
            raise ValueError("`dim1` must be either 1 or 2.")
        if dim2 not in self.dims:
            raise ValueError("`dim2` must be either 1 or 2.")

        node_index_ = node_index.clone()
        for (shift, dim) in ((shift1, dim1), (shift2, dim2)):
            if dim == 1:
                node_index_[:, self.shift2mask_dim[shift], :] = -1
            else:
                node_index_[:, :, self.shift2mask_dim[shift]] = -1
            node_index_ = torch.roll(node_index_, shift, dim)
        mask = node_index_.flatten() >= 0
        relation = torch.ones(mask.sum(), dtype=torch.long, device=node_index.device) \
                   * self.shift_dim_pair2relation[((shift1, dim1), (shift2, dim2))]
        edge_list = torch.stack([node_index_.flatten()[mask], node_index.flatten()[mask], relation], dim=-1)

        return edge_list

    def forward(self, graph):
        """
        Return spatial edges connecting the nodes corresponding to adjacent patches.

        Parameters:
            graph (data.Batch): A batch of graphs.

        Returns:
            Tensor: Edges with shape (2, |E|).
            Tensor: Edge types with shape (|E|,).
            int: Number of relations.
        """
        per_graph_num_nodes = scatter_add(torch.ones_like(graph.batch), graph.batch, dim=0)
        if torch.all(per_graph_num_nodes == self.num_patch):  # without virtual node
            node_index = torch.arange(graph.num_nodes, device=graph.x.device)
            node_index = node_index.view(len(graph), self.resolution, self.resolution)
        elif torch.all(per_graph_num_nodes == (self.num_patch + 1)):  # with virtual node
            node_index = torch.arange(graph.num_nodes, device=graph.x.device)
            node_index = node_index.view(len(graph), -1)[:, 1:]
            node_index = node_index.view(len(graph), self.resolution, self.resolution)
        else:
            raise ValueError("The number of nodes in each graph should be either %d or %d."
                             % (self.num_patch, self.num_patch + 1))

        edge_index = []
        edge_type = []
        for (shift, dim) in self.shift_dim2relation.keys():
            _edge_index, _edge_type = self.adjacent_edge_list(node_index, shift, dim)
            edge_index.append(_edge_index)
            edge_type.append(_edge_type)

        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)
        num_relation = len(self.shift_dim2relation)
        return edge_index, edge_type, num_relation


class MediumRangeEdge(nn.Module):
    """
    Edge construction model for K-nearest semantic neighbors (medium-range edges).

    Parameters:
        dim (int): Dimension of hidden representations.
        resolution (int): Resolution of feature map.
        k (int): Number of neighbors.
    """

    inf = 1e5

    def __init__(self, dim, resolution, k=10):
        super(MediumRangeEdge, self).__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_patch = resolution ** 2
        self.k = k
        self.compute_spatial_neighbors()

        relative_pos = torch.from_numpy(np.float32(
            position_embed.get_2d_relative_pos_embed(dim, resolution))).unsqueeze(0)  # [1, N_patch, N_patch]
        self.relative_pos = nn.Parameter(-relative_pos, requires_grad=False)
        assert self.relative_pos.shape[1] == self.num_patch and self.relative_pos.shape[2] == self.num_patch

    def compute_spatial_neighbors(self):
        self.spatial_neighbors = {"left": [[], []], "right": [[], []], "upper": [[], []], "lower": [[], []],
                                  "upper_left": [[], []], "upper_right": [[], []], "lower_left": [[], []],
                                  "lower_right": [[], []]}
        for node_id in range(self.num_patch):
            if not node_id % self.resolution == 0:
                self.spatial_neighbors["left"][0].append(node_id)
                self.spatial_neighbors["left"][1].append(node_id - 1)
            if not node_id % self.resolution == (self.resolution - 1):
                self.spatial_neighbors["right"][0].append(node_id)
                self.spatial_neighbors["right"][1].append(node_id + 1)
            if node_id >= self.resolution:
                self.spatial_neighbors["upper"][0].append(node_id)
                self.spatial_neighbors["upper"][1].append(node_id - self.resolution)
            if node_id < self.resolution * (self.resolution - 1):
                self.spatial_neighbors["lower"][0].append(node_id)
                self.spatial_neighbors["lower"][1].append(node_id + self.resolution)
            if node_id >= self.resolution and not node_id % self.resolution == 0:
                self.spatial_neighbors["upper_left"][0].append(node_id)
                self.spatial_neighbors["upper_left"][1].append(node_id - self.resolution - 1)
            if node_id >= self.resolution and not node_id % self.resolution == (self.resolution - 1):
                self.spatial_neighbors["upper_right"][0].append(node_id)
                self.spatial_neighbors["upper_right"][1].append(node_id - self.resolution + 1)
            if node_id < self.resolution * (self.resolution - 1) and not node_id % self.resolution == 0:
                self.spatial_neighbors["lower_left"][0].append(node_id)
                self.spatial_neighbors["lower_left"][1].append(node_id + self.resolution - 1)
            if node_id < self.resolution * (self.resolution - 1) \
                    and not node_id % self.resolution == (self.resolution - 1):
                self.spatial_neighbors["lower_right"][0].append(node_id)
                self.spatial_neighbors["lower_right"][1].append(node_id + self.resolution + 1)

        for k, v in self.spatial_neighbors.items():
            self.spatial_neighbors[k] = [torch.tensor(v_) for v_ in v]

    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.

        Parameters:
            x (Tensor): point-wise features with shape (batch_size, num_points, feature_dims).

        Returns:
            Tensor: pairwise distances with shape (batch_size, num_points, num_points).
        """
        x_norm = (x ** 2).sum(-1).view(x.shape[0], -1, 1)
        y_norm = x_norm.view(x.shape[0], 1, -1)
        dist = x_norm + y_norm - 2.0 * (x @ x.transpose(1, 2))

        return dist

    def filter_pairwise_distance(self, dist, with_virtual=False):
        for neighbors in self.spatial_neighbors.values():
            if with_virtual:
                dist[:, neighbors[0] + 1, neighbors[1] + 1] += self.inf
            else:
                dist[:, neighbors[0], neighbors[1]] += self.inf

        return dist

    def forward(self, graph):
        """
        Return KNN edges constructed based on node features.

        Parameters:
            graph (data.Batch): A batch of graphs.

        Returns:
            Tensor: Edges with shape (2, |E|).
            Tensor: Edge types with shape (|E|,).
            int: Number of relations.
        """
        per_graph_num_nodes = scatter_add(torch.ones_like(graph.batch), graph.batch, dim=0)
        per_graph_num_cum_nodes = per_graph_num_nodes.cumsum(0)
        if torch.all(per_graph_num_nodes == self.num_patch):  # without virtual node
            x = graph.x.view(len(graph), -1, graph.num_node_features)
            x = F.normalize(x, p=2.0, dim=-1)
            dist = self.pairwise_distance(x)
            dist += self.relative_pos

            # build semantic KNN edges
            dist[:, range(dist.shape[1]), range(dist.shape[2])] += self.inf
            dist = self.filter_pairwise_distance(dist, with_virtual=False)
            _, topk = torch.topk(-dist, k=self.k, dim=-1)
            start = (per_graph_num_cum_nodes -
                     per_graph_num_nodes).repeat_interleave(per_graph_num_nodes * self.k).view(topk.shape)
            topk += start
            src_node_index = torch.arange(
                graph.num_nodes, device=graph.x.device).view(len(graph), -1, 1).repeat(1, 1, self.k)
            edge_index = torch.stack([topk, src_node_index], dim=0).flatten(1)
        elif torch.all(per_graph_num_nodes == (self.num_patch + 1)):  # with virtual node
            x = graph.x.view(len(graph), -1, graph.num_node_features)
            x = F.normalize(x, p=2.0, dim=-1)
            dist = self.pairwise_distance(x)
            dist[:, 1:, 1:] += self.relative_pos

            # build semantic KNN edges
            dist[:, range(dist.shape[1]), range(dist.shape[2])] += self.inf
            dist[:, :, 0] += self.inf
            dist = self.filter_pairwise_distance(dist, with_virtual=True)
            dist = dist[:, 1:, :]  # remove virtual nodes from query
            _, topk = torch.topk(-dist, k=self.k, dim=-1)
            start = (per_graph_num_cum_nodes -
                     per_graph_num_nodes).repeat_interleave((per_graph_num_nodes - 1) * self.k).view(topk.shape)
            topk += start
            src_node_index = torch.arange(
                graph.num_nodes, device=graph.x.device).view(len(graph), -1, 1).repeat(1, 1, self.k)[:, 1:, :]
            edge_index = torch.stack([topk, src_node_index], dim=0).flatten(1)
        else:
            raise ValueError("The number of nodes in each graph should be either %d or %d."
                             % (self.num_patch, self.num_patch + 1))

        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=graph.x.device)
        return edge_index, edge_type, 1


class LongRangeEdge(nn.Module):
    """
    Edge construction model for local-global connections (long-range edges).

    Parameters:
        resolution (int): Resolution of feature map.
    """

    def __init__(self, resolution):
        super(LongRangeEdge, self).__init__()
        self.resolution = resolution
        self.num_patch = resolution ** 2

    def forward(self, graph):
        """
        Return local-global edges connecting the virtual node with all patch nodes.

        Parameters:
            graph (data.Batch): A batch of graphs.

        Returns:
            Tensor: Edges with shape (2, |E|).
            Tensor: Edge types with shape (|E|,).
            int: Number of relations.
        """
        per_graph_num_nodes = scatter_add(torch.ones_like(graph.batch), graph.batch, dim=0)
        if not torch.all(per_graph_num_nodes == (self.num_patch + 1)):
            raise ValueError("The number of nodes in each graph should be %d." % (self.num_patch + 1))

        node_index = torch.arange(graph.num_nodes, device=graph.x.device).view(len(graph), -1)
        virtual_node_index = node_index[:, 0].unsqueeze(-1)
        node_index = node_index[:, 1:]
        virtual_node_index = virtual_node_index.repeat(1, node_index.shape[1])

        # global to local edges
        edge_index = torch.stack([virtual_node_index.flatten(), node_index.flatten()], dim=0)
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long, device=graph.x.device)
        return edge_index, edge_type, 1


class GraphConstruction(nn.Module):
    """
    Graph construction model.

    Parameters:
        node_layers (list of nn.Module): Node construction models.
        edge_layers (list of nn.Module): Edge construction models.
    """

    def __init__(self, node_layers=None, edge_layers=None):
        super(GraphConstruction, self).__init__()
        self.node_layers = nn.ModuleList() if node_layers is None else nn.ModuleList(node_layers)
        self.edge_layers = nn.ModuleList() if edge_layers is None else nn.ModuleList(edge_layers)

    def apply_node_layer(self, graph):
        for layer in self.node_layers:
            graph = layer(graph)
        return graph

    def apply_edge_layer(self, graph):
        if len(self.edge_layers) == 0:
            return graph

        edge_index = []
        edge_type = []
        num_edges = []
        num_relations = []
        for layer in self.edge_layers:
            _edge_index, _edge_type, _num_relation = layer(graph)
            edge_index.append(_edge_index)
            edge_type.append(_edge_type)
            num_edges.append(_edge_index.shape[1])
            num_relations.append(_num_relation)

        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)
        num_edges = torch.tensor(num_edges, device=graph.x.device)
        num_relations = torch.tensor(num_relations, device=graph.x.device)
        relation_offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_type += relation_offsets

        # reorder edges into a valid batch
        node_in, node_out = edge_index
        edge2graph = graph.batch[node_in]
        order = edge2graph.argsort()
        edge_index = torch.stack([node_in[order], node_out[order]], dim=0)
        edge_type = edge_type[order]
        edge_weight = torch.ones(edge_index.shape[1], device=graph.x.device)

        graph.edge_index = edge_index
        graph.edge_type = edge_type
        graph.edge_weight = edge_weight
        graph.num_relation = num_relations.sum().item()
        return graph

    def forward(self, graph):
        """
        Generate new graphs based on the input graphs and pre-defined node and edge layers.

        Parameters:
            graph (PackedGraph): A batch of graphs.

        Returns:
            PackedGraph: A batch of new graphs.
        """
        graph = self.apply_node_layer(graph)
        graph = self.apply_edge_layer(graph)

        return graph
