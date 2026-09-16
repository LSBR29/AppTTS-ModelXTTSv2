"""Pre-allocated-buffer KV-cache layer (E018, 2026-08-29): avoids the per-decode-step
`torch.cat`-based reallocation `transformers.cache_utils.DynamicLayer.update()` performs by
default. Measured via cProfile on a real 255-token generation: that `torch.cat` call accounted
for ~9.6% of total wall time (1.667s / 17.317s), 13263 calls (~52/token = 25 layers x key+value).

Only the Cache's internal K/V STORAGE mechanism changes here -- `attention_mask` handling (in
`GenerationMixin._update_model_kwargs_for_generation`) is untouched and keeps growing
incrementally exactly as before. This is what makes this safe where a full
`cache_implementation="static"` was NOT (see E014, abandoned: HF's automatic static-cache
creation also changes attention_mask to a fixed-size, padded-from-the-start tensor, which broke
`GPT2InferenceModel.forward`'s custom position calculation, `attention_mask.shape[1] - (prefix_len
+ 1)`, which assumes exactly-one-column-per-token incremental growth). Passing an
already-constructed Cache instance via `past_key_values=` bypasses HF's cache-implementation
selection logic entirely (confirmed via `GenerationMixin._prepare_cache_for_generation` source:
"Quick escape route 1" returns immediately if the cache is already user-provided), so
attention_mask's code path is never touched by this change.

Verified bit-identical to the default DynamicCache-backed generation for the same seed (see
.claude-work/experiments/E018_preallocated_kv_cache/EXPERIMENT.md) -- the returned key/value
tensors are numerically identical, only the storage mechanism differs.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class PreallocatedDynamicLayer(DynamicLayer):
    def __init__(self, max_cache_len):
        self.max_cache_len = max_cache_len
        self._valid_len = 0
        self._buf_keys = None
        self._buf_values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        batch, heads, _, head_dim = key_states.shape
        self._buf_keys = torch.zeros(batch, heads, self.max_cache_len, head_dim, dtype=self.dtype, device=self.device)
        self._buf_values = torch.zeros(batch, heads, self.max_cache_len, head_dim, dtype=self.dtype, device=self.device)
        self._valid_len = 0
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        new_len = key_states.shape[-2]
        start, end = self._valid_len, self._valid_len + new_len
        if end > self.max_cache_len:
            raise RuntimeError(
                f"PreallocatedDynamicLayer overflow: need {end} but max_cache_len={self.max_cache_len}"
            )
        self._buf_keys[:, :, start:end, :] = key_states
        self._buf_values[:, :, start:end, :] = value_states
        self._valid_len = end
        return self._buf_keys[:, :, :end, :], self._buf_values[:, :, :end, :]

    @property
    def keys(self):
        if not self.is_initialized:
            return torch.tensor([])
        return self._buf_keys[:, :, : self._valid_len, :]

    @keys.setter
    def keys(self, value):
        pass  # base class may assign during construction; storage is buffer-backed, ignore

    @property
    def values(self):
        if not self.is_initialized:
            return torch.tensor([])
        return self._buf_values[:, :, : self._valid_len, :]

    @values.setter
    def values(self, value):
        pass


def build_preallocated_cache(config, max_cache_len):
    """A DynamicCache whose .layers are all PreallocatedDynamicLayer instead of DynamicLayer --
    same public Cache API (update/get_seq_length/etc via delegation to the layer), different
    internal storage."""
    cache = DynamicCache(config=config)
    num_layers = len(cache.layers)
    cache.layers = [PreallocatedDynamicLayer(max_cache_len) for _ in range(num_layers)]
    return cache
