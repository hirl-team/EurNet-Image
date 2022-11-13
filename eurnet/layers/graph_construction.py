import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
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
            graph (PackedGraph): A batch of graphs.

        Returns:
            PackedGraph: A batch of augmented graphs.
        """
        graph.num_nodes = graph.num_nodes + 1
        graph.num_cum_nodes = graph.num_nodes.cumsum(0)
        graph.num_node = graph.num_nodes.sum()
        with graph.node():
            virtual_node_feature = self.virtual_node_embed.unsqueeze(0).repeat(graph.batch_size, 1, 1)
            node_feature = graph.node_feature.view(graph.batch_size, -1, graph.node_feature.shape[-1])
            new_node_feature = torch.cat([virtual_node_feature, node_feature], dim=1).flatten(0, 1)
            graph.node_feature = new_node_feature

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
        relation = torch.ones(mask.sum(), dtype=torch.long, device=node_index.device) \
                   * self.shift_dim2relation[(shift, dim)]
        edge_list = torch.stack([node_index_.flatten()[mask], node_index.flatten()[mask], relation], dim=-1)

        return edge_list

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
            graph (PackedGraph): A batch of graphs.

        Returns:
            Tensor: Edges with shape (|E|, 3).
            int: Number of relations.
        """
        if torch.all(graph.num_nodes == self.num_patch):  # without virtual node
            node_index = torch.arange(graph.num_node, device=graph.device)
            node_index = node_index.view(graph.batch_size, self.resolution, self.resolution)
        elif torch.all(graph.num_nodes == (self.num_patch + 1)):  # with virtual node
            node_index = torch.arange(graph.num_node, device=graph.device)
            node_index = node_index.view(graph.batch_size, -1)[:, 1:]
            node_index = node_index.view(graph.batch_size, self.resolution, self.resolution)
        else:
            raise ValueError("The number of nodes in each graph should be either %d or %d."
                             % (self.num_patch, self.num_patch + 1))

        edge_list = []
        for (shift, dim) in self.shift_dim2relation.keys():
            edges = self.adjacent_edge_list(node_index, shift, dim)
            edge_list.append(edges)

        edge_list = torch.cat(edge_list, dim=0)

        num_relation = len(self.shift_dim2relation)
        return edge_list, num_relation


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
            graph (PackedGraph): A batch of graphs.

        Returns:
            Tensor: Edges with shape (|E|, 3).
            int: Number of relations.
        """
        if torch.all(graph.num_nodes == self.num_patch):  # without virtual node
            node_feature = graph.node_feature.view(graph.batch_size, -1, graph.node_feature.shape[-1])
            node_feature = F.normalize(node_feature, p=2.0, dim=-1)
            dist = self.pairwise_distance(node_feature)
            dist += self.relative_pos

            # build semantic KNN edges
            dist[:, range(dist.shape[1]), range(dist.shape[2])] += self.inf
            dist = self.filter_pairwise_distance(dist, with_virtual=False)
            _, topk = torch.topk(-dist, k=self.k, dim=-1)
            start = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(graph.num_nodes * self.k).view(topk.shape)
            topk += start
            src_node_index = torch.arange(
                graph.num_node, device=graph.device).view(graph.batch_size, -1, 1).repeat(1, 1, self.k)
            edge_list = torch.stack([topk, src_node_index], dim=-1).flatten(0, -2)
        elif torch.all(graph.num_nodes == (self.num_patch + 1)):  # with virtual node
            node_feature = graph.node_feature.view(graph.batch_size, -1, graph.node_feature.shape[-1])
            node_feature = F.normalize(node_feature, p=2.0, dim=-1)
            dist = self.pairwise_distance(node_feature)
            dist[:, 1:, 1:] += self.relative_pos

            # build semantic KNN edges
            dist[:, range(dist.shape[1]), range(dist.shape[2])] += self.inf
            dist[:, :, 0] += self.inf
            dist = self.filter_pairwise_distance(dist, with_virtual=True)
            dist = dist[:, 1:, :]  # remove virtual nodes from query
            _, topk = torch.topk(-dist, k=self.k, dim=-1)
            start = (graph.num_cum_nodes -
                     graph.num_nodes).repeat_interleave((graph.num_nodes - 1) * self.k).view(topk.shape)
            topk += start
            src_node_index = torch.arange(
                graph.num_node, device=graph.device).view(graph.batch_size, -1, 1).repeat(1, 1, self.k)[:, 1:, :]
            edge_list = torch.stack([topk, src_node_index], dim=-1).flatten(0, -2)
        else:
            raise ValueError("The number of nodes in each graph should be either %d or %d."
                             % (self.num_patch, self.num_patch + 1))

        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        return edge_list, 1


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
            graph (PackedGraph): A batch of graphs.

        Returns:
            Tensor: Edges with shape (|E|, 3).
            int: Number of relations.
        """
        if not torch.all(graph.num_nodes == (self.num_patch + 1)):
            raise ValueError("The number of nodes in each graph should be %d." % (self.num_patch + 1))

        node_index = torch.arange(graph.num_node, device=graph.device).view(graph.batch_size, -1)
        virtual_node_index = node_index[:, 0].unsqueeze(-1)
        node_index = node_index[:, 1:]
        virtual_node_index = virtual_node_index.repeat(1, node_index.shape[1])

        # global to local edges
        relation = torch.zeros(node_index.flatten().shape[0], dtype=torch.long, device=graph.device)
        edge_list = torch.stack([virtual_node_index.flatten(), node_index.flatten(), relation], dim=-1)

        return edge_list, 1


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

        edge_list = []
        num_edges = []
        num_relations = []
        for layer in self.edge_layers:
            edges, num_relation = layer(graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(num_relation)

        edge_list = torch.cat(edge_list, dim=0)
        num_edges = torch.tensor(num_edges, device=graph.device)
        num_relations = torch.tensor(num_relations, device=graph.device)
        num_relation = num_relations.sum()
        relation_offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_list[:, 2] += relation_offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        edge_weight = torch.ones(len(edge_list), device=graph.device)
        num_edges = edge2graph.bincount(minlength=graph.batch_size)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        data_dict, meta_dict = graph.data_by_meta(include=("node", "node reference"))
        new_graph = type(graph)(edge_list, edge_weight=edge_weight, num_nodes=graph.num_nodes, num_edges=num_edges,
                                num_relation=num_relation, offsets=offsets, meta_dict=meta_dict, **data_dict)

        return new_graph

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
