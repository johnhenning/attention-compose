"""Standalone test oracles: no attention_compose imports, classes, or inheritance.

These intentionally spell out linear projections, grouped heads, visibility, and
softmax in one function. They do not call SDPA or the library's projection, cache,
retention, mask, or kernel helpers. Python loops make mask semantics explicit.
"""

from collections.abc import Mapping, Sequence
from math import sqrt

import torch
from torch import Tensor


def direct_attention(
    queries: Tensor,
    weights: Mapping[str, Tensor],
    query_heads: int,
    kv_heads: int,
    *,
    keys: Tensor | None = None,
    query_positions: Sequence[int] | None = None,
    key_positions: Sequence[int] | None = None,
    causal: bool = True,
    window: int | None = None,
    global_positions: Sequence[int] = (),
    global_queries: bool = False,
) -> Tensor:
    """Direct MHA/GQA/MQA, local, and local/global attention equations.

    A separate raw `keys` sequence permits an independent streaming oracle:
    tests retain raw inputs, then recompute K/V instead of using any KV cache.
    """
    source = queries if keys is None else keys
    batch, query_length, width = queries.shape
    key_length = source.shape[1]
    dim = width // query_heads
    q = queries @ weights["q_proj.weight"].T
    k = source @ weights["k_proj.weight"].T
    v = source @ weights["v_proj.weight"].T
    if "q_proj.bias" in weights:
        q = q + weights["q_proj.bias"]
        k = k + weights["k_proj.bias"]
        v = v + weights["v_proj.bias"]
    q = q.reshape(batch, query_length, query_heads, dim).permute(0, 2, 1, 3)
    k = k.reshape(batch, key_length, kv_heads, dim).permute(0, 2, 1, 3)
    v = v.reshape(batch, key_length, kv_heads, dim).permute(0, 2, 1, 3)
    # Index each query head's shared KV head without the production expansion helper.
    kv_index = [head // (query_heads // kv_heads) for head in range(query_heads)]
    k, v = k[:, kv_index], v[:, kv_index]
    q_positions = list(range(query_length)) if query_positions is None else query_positions
    k_positions = list(range(key_length)) if key_positions is None else key_positions
    visible = torch.tensor(
        [
            [
                (not causal or key <= query)
                and (
                    window is None
                    or abs(query - key) < window
                    or key in global_positions
                    or (global_queries and query in global_positions)
                )
                for key in k_positions
            ]
            for query in q_positions
        ],
        dtype=torch.bool,
        device=queries.device,
    )
    scores = (q @ k.transpose(-1, -2)) / sqrt(dim)
    scores = scores.masked_fill(~visible, float("-inf"))
    # Independent zero-row handling: give empty rows a temporary uniform softmax,
    # then remove all probabilities with their original visibility mask.
    for row in range(query_length):
        if not bool(visible[row].any()):
            scores[:, :, row, :] = 0
    probability = torch.softmax(scores, dim=-1) * visible.to(scores.dtype)
    attended = (probability @ v).permute(0, 2, 1, 3).reshape(batch, query_length, width)
    output = attended @ weights["out_proj.weight"].T
    if "out_proj.bias" in weights:
        output = output + weights["out_proj.bias"]
    return output
