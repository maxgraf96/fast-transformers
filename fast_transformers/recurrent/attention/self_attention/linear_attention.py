#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the causally masked linear attention as a recurrent model."""
import numpy as np
import torch
from torch.nn import Module

from ....attention_registry import RecurrentAttentionRegistry, Optional, Int, \
    Callable, EventDispatcherInstance
from ....events import EventDispatcher
from ....feature_maps import elu_feature_map
from ..._utils import check_state


class RecurrentLinearAttention(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        # self.resmulouter = torch.zeros((1, 8, query_dimensions, query_dimensions)).cuda()


    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)

        # If this is a new sequence reinitialize the feature map
        if state is None:
            self.feature_map.new_feature_map(query.device)

        # Apply the feature map to the query and key
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))
        else:
            Si, Zi = state

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        test = torch.zeros((64, 64)).cuda()
        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K

            for i in range(1):
                # for j in range(K.shape[1]):
                for j in range(8):
                    # asdf = K[i, j] * value[i, j]
                    # test = torch.cat([test, K[i, j]], dim=1)
                    test2 = K[i, j].unsqueeze(1) * value[i, j]
                    test = torch.cat([test, test2], dim=0)
                    # Si[i, j] += asdf

            asdf = torch.reshape(test[64:], (1, 8, 64, 64))
            # final = torch.stack([asdf, test], dim = 0)
            Si = Si + asdf
            # Si = Si + torch.einsum("nhd,nhm->nhdm", K, value)
        else:
            Zi += K
            # resmul = torch.einsum("nhd,nhm->nhdm", K, value)

            for i in range(1):
                # for j in range(K.shape[1]):
                for j in range(8):
                    # asdf = K[i, j] * value[i, j]
                    # test = torch.cat([test, K[i, j]], dim=1)
                    test2 = K[i, j].unsqueeze(1) * value[i, j]
                    test = torch.cat([test, test2], dim=0)
                    # Si[i, j] += asdf

            asdf = torch.reshape(test[64:], (1, 8, 64, 64))
            # final = torch.stack([asdf, test], dim = 0)
            Si = Si + asdf

            # s = torch.allclose(self.resmulouter, res)
            # Si += res
            # Si += self.resmulouter

        # Compute the output
        # Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        resmul = torch.sum(Q * Zi, dim=2)
        Z = 1. / (resmul + self.eps)

        # V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)
        V = torch.sum(Q[..., None] * Si, dim=-2) * Z[..., None]

        # V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Zmul)
        # same = torch.allclose(res, V)

        return V, [Si, Zi]


# Register the attention implementation so that it becomes available in our
# builders
RecurrentAttentionRegistry.register(
    "linear", RecurrentLinearAttention,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
RecurrentAttentionRegistry.register(
    "causal-linear", RecurrentLinearAttention,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
