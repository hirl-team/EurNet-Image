import math

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing


class GatedRelationalMessagePassing(MessagePassing):
    """
    Gated Relational Message Passing layer.
    Inherit from the `MessagePassing` module of PyTorch Geometric,
    which sequentially performs `message`, `aggregate` and `update` within `propagate`.

    Parameters:
        input_dim (int): Dimension of input representations.
        output_dim (int): Dimension of output representations.
        num_relation (int): Number of all considered relations.
        context_relation (bool): Whether to add an extra global context relation.
        context_sizes (list of int): Sizes of context convolutional kernels.
        norm_layer (nn.Module or None): Normalization layer.
        act_layer (nn.Module or None): Activation layer.
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation,
                 context_relation=False, context_sizes=[3, 5], norm_layer=None, act_layer=None):
        super(GatedRelationalMessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.context_relation = context_relation

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
        num_context_relation = 1
        num_weight = num_relation + num_context_relation if context_relation else num_relation
        self.weight_linear = nn.Linear(input_dim, num_weight)
        self.linear = nn.Conv1d(num_relation * output_dim, num_relation * output_dim,
                                kernel_size=1, groups=num_relation * output_dim, bias=False)
        self.gate_linear = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_type, edge_weight, size, batch_size, num_relation):
        output = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
                                size=size, batch_size=batch_size, num_relation=num_relation)

        return output

    def propagate(self, edge_index, size=None, **kwargs):
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index,
                                     size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        for hook in self._message_forward_pre_hooks.values():
            res = hook(self, (msg_kwargs,))
            if res is not None:
                msg_kwargs = res[0] if isinstance(res, tuple) else res
        out = self.message(**msg_kwargs)
        for hook in self._message_forward_hooks.values():
            res = hook(self, (msg_kwargs,), out)
            if res is not None:
                out = res

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        for hook in self._aggregate_forward_pre_hooks.values():
            res = hook(self, (aggr_kwargs,))
            if res is not None:
                aggr_kwargs = res[0] if isinstance(res, tuple) else res
        out = self.aggregate(out, **aggr_kwargs)
        for hook in self._aggregate_forward_hooks.values():
            res = hook(self, (aggr_kwargs,), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, x, edge_index_j):
        B, L, C = x.shape
        message_input = self.message_linear(x)
        if int(math.sqrt(L)) != math.sqrt(L):
            message_input[:, 0, :] = message_input[:, 1:, :].mean(1)
        message_input = message_input.flatten(0, 1)
        message = message_input[edge_index_j]

        return [message, message_input]

    def aggregate(self, input, edge_index_i, edge_type, edge_weight, size_i, batch_size, num_relation):
        message, message_input = input
        if num_relation != self.num_relation:
            raise ValueError("Mismatch between the relation number of the graph and the model.")

        node_out = edge_index_i * self.num_relation + edge_type
        edge_weight = edge_weight.unsqueeze(-1)

        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=size_i * self.num_relation) / \
                 (scatter_add(edge_weight, node_out, dim=0, dim_size=size_i * self.num_relation) + self.eps)
        update = update.view(batch_size, -1, self.num_relation * self.output_dim).transpose(1, 2)  # [N, RC, L]
        update = self.activation(self.linear(update))
        update = update.transpose(1, 2).view(batch_size, -1, self.num_relation, self.output_dim)  # [N, L, R, C]

        if self.context_relation:
            B, L, R, C = update.shape
            message_input = message_input.view(B, L, C)
            if int(math.sqrt(L)) != math.sqrt(L):
                context_update = message_input[:, 1:, :].transpose(1, 2).view(B, C, int(math.sqrt(L - 1)), -1)
            else:
                context_update = message_input.transpose(1, 2).view(B, C, int(math.sqrt(L)), -1)
            context_updates = []
            for context_layer in self.context_layers:
                context_update = context_layer(context_update)
                _context_update = context_update.flatten(2).transpose(1, 2)
                if int(math.sqrt(L)) != math.sqrt(L):
                    _context_update = torch.cat([message_input[:, :1, :], _context_update], dim=1)
                context_updates.append(_context_update)
            context_update_ = context_updates[-1].unsqueeze(-2)
            update = torch.cat([update, context_update_], dim=-2)  # [N, L, R + R_context, C]

        return update

    def update(self, update, x):
        B, L, C = x.shape
        relation_weight = self.weight_linear(x).unsqueeze(-1)  # [N, L, R, 1]
        update = (update * relation_weight).sum(dim=2)  # [N, L, C]
        gate = self.gate_linear(update)
        output = self.self_loop(x) * gate
        if self.norm:
            output = self.norm(output)

        return output
