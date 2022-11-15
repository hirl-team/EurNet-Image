"""
Mainly copy from https://github.com/DeepGraphLearning/torchdrug
"""
from torch import nn

class MessagePassingBase(nn.Module):
    """
    Base module for message passing.
    Any custom message passing module should be derived from this class.
    """
    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.
        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`
        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update):
        """
        Combine node input and node update.
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).
        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output