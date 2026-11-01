# ModernBERT-large / ModernBERT-Large-Instruct: architecture reference for a from-scratch Rust forward pass

Sources used (all fetched 2026-11-01):

- `https://huggingface.co/answerdotai/ModernBERT-Large-Instruct/raw/main/config.json`
- `https://huggingface.co/answerdotai/ModernBERT-large/raw/main/config.json`
- HF transformers `v4.48.0` `src/transformers/models/modernbert/modeling_modernbert.py` (the version that wrote these checkpoints; `transformers_version: 4.48.0` in the Instruct config)
- HF transformers `main` `src/transformers/models/modernbert/{modeling,configuration}_modernbert.py`, `masking_utils.py`, `modeling_flash_attention_utils.py` (post-refactor; used to confirm the off-by-one)
- HF transformers `v4.48.0` `src/transformers/activations.py`
- Paper arXiv 2412.13663v2
- `pylate-rs` Rust reference: `https://docs.rs/pylate-rs/latest/src/pylate_rs/modernbert.rs.html`

Local copies of every file are in `dl/` next to this document.

**Transformers is the authority below. Where the paper is vaguer, that is noted.**

---

## 1. Core constants

The two configs (`ModernBERT-large` and `ModernBERT-Large-Instruct`) are **architecturally identical**. Only `torch_dtype` differs (`float32` vs `bfloat16`), plus the Instruct config adds `sparse_prediction`, `repad_logits_with_grad` and `reference_compile`, and omits the explicit `tie_word_embeddings` (which defaults to `True`).

| Constant | Value | Source |
|---|---|---|
| `num_hidden_layers` | **28** | config |
| `hidden_size` | **1024** | config |
| `intermediate_size` | **2624** | config |
| `num_attention_heads` | **16** | config |
| `head_dim` | **64** | derived: `hidden_size // num_attention_heads` |
| `vocab_size` | **50368** | config (50,285 real + 83 unused padding tokens, per paper) |
| `max_position_embeddings` | **8192** | config |
| `norm_eps` | **1e-5** | config (`norm_eps`; `layer_norm_eps` is vestigial, see traps) |
| `norm_bias` | **false** | config |
| `attention_bias` | **false** | config |
| `mlp_bias` | **false** | config |
| `decoder_bias` | **true** | config |
| `classifier_bias` | **false** | config (this is the bias of `head.dense`) |
| `hidden_activation` | `"gelu"` -> exact erf GELU | config + `activations.py` |
| `classifier_activation` | `"gelu"` -> exact erf GELU | config + `activations.py` |
| `global_attn_every_n_layers` | **3** | config |
| `local_attention` | **128** (TOTAL window) | config |
| `global_rope_theta` | **160000.0** | config |
| `local_rope_theta` | **10000.0** | config |
| `tie_word_embeddings` | **true** | `ModernBertConfig` default; checkpoint has no `decoder.weight` |
| `classifier_pooling` | `"mean"` | config (only matters for the SequenceClassification head) |
| GLU expansion width (`Wi` out) | **5248** = `2 * 2624` | code |
| Special tokens | `cls/bos=50281`, `sep/eos=50282`, `pad=50283` | config |

Dropouts are all `0.0` (`attention_dropout`, `mlp_dropout`, `embedding_dropout`, `classifier_dropout`), so inference ignores them entirely.

---

## 2. Global / local attention schedule

`global_attn_every_n_layers = 3`. A layer is **global (full) attention when `layer_id % 3 == 0`**, local otherwise.

v4.48 `ModernBertAttention.__init__`:

```python
if layer_id % config.global_attn_every_n_layers != 0:
    self.local_attention = (config.local_attention // 2, config.local_attention // 2)
else:
    self.local_attention = (-1, -1)
```

`main` branch, same rule expressed in `ModernBertConfig.__post_init__`:

```python
global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)
if self.layer_types is None:
    self.layer_types = [
        "sliding_attention" if bool(i % global_attn_every_n_layers) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]
```

**Layer 0 is GLOBAL.** (`0 % 3 == 0` -> `full_attention`.)

For 28 layers:

- **Global layer indices (10 layers): 0, 3, 6, 9, 12, 15, 18, 21, 24, 27**
- Local layer indices (18 layers): all others (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26)

Note the last layer (27) is global, which is convenient: the final representation is not window-limited.

`pylate-rs` agrees: `let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;`

---

## 3. Sliding-window size: TOTAL, so 64 each side

`local_attention = 128` is the **total** window. The half-window is 64 **each side**. Transformers states this explicitly in `configuration_modernbert.py` (`main`):

```python
@property
def sliding_window(self):
    """Half-window size: `local_attention` is the total window, so we divide by 2."""
    return self.local_attention // 2
```

The v4.48 mask builder, `ModernBertModel._update_attention_mask` (this is the exact code you asked to be quoted):

```python
global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

# Create position indices
rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
# Calculate distance between positions
distance = torch.abs(rows - rows.T)

# Create sliding window mask (1 for positions within window, 0 outside)
window_mask = (
    (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
)
# Combine with existing mask
sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)

return global_attention_mask, sliding_window_mask
```

So the predicate is **`abs(i - j) <= 64`, inclusive**. Each query therefore sees **129 keys** (itself + 64 left + 64 right), not 128. This is intentional and consistent across every code path:

- v4.48 eager/sdpa: `distance <= local_attention // 2` -> `<= 64`.
- v4.48 FA2: `window_size=self.local_attention` = `(64, 64)`, and flash-attn's `window_size` is inclusive on each side -> 64 + 64 + self.
- `main` eager/sdpa: `create_bidirectional_sliding_window_mask` reads `config.sliding_window` (= 64) and applies, in `masking_utils.py`:

  ```python
  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
      """A token can attend to any other token if their absolute distance is within
      the (inclusive) sliding window size (distance <= sliding_window)."""
      return abs(q_idx - kv_idx) <= sliding_window
  ```

- `main` FA2: the attention module stores `self.sliding_window = config.sliding_window + 1` = **65**, with the comment

  ```python
  if config.layer_types[layer_idx] == "sliding_attention":
      # config.sliding_window = local_attention // 2 (half-window size, e.g. 64 for local_attention=128)
      # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
      self.sliding_window = config.sliding_window + 1
  ```

  and `modeling_flash_attention_utils.py` then converts it back: `flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)` -> `(64, 64)`.

  **Do not copy the `+1` into your Rust code.** It exists only to cancel a `-1` inside the flash-attention wrapper. The semantic window is 64/64.

`pylate-rs` agrees: `get_local_attention_mask(seq_len, self.local_attention_size / 2, ...)` with `if (j - i).abs() > max_distance { NEG_INFINITY } else { 0. }` — strict `>` for masking, i.e. inclusive `<=` for keeping.

The paper only says "the remaining layers use a 128 token, local sliding window attention"; it does **not** disambiguate total vs per-side. The code does: total.

---

## 4. RoPE

### Different theta per layer type: YES, confirmed

v4.48 `ModernBertAttention.__init__`:

```python
rope_theta = config.global_rope_theta
max_position_embeddings = config.max_position_embeddings
if self.local_attention != (-1, -1):
    if config.local_rope_theta is not None:
        rope_theta = config.local_rope_theta
    max_position_embeddings = config.local_attention
...
self.rotary_emb = ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)
```

- Global layers (0, 3, 6, ...): `theta = 160000.0`
- Local layers: `theta = 10000.0`

You need **exactly two** cos/sin tables. `main` builds exactly two and selects by `layer_type`.

The `max_position_embeddings = config.local_attention` on the local branch only sizes the *precomputed cache* for the FA2 unpadded variant. It does **not** change position indices: local layers still use the true absolute positions `0..seq_len-1`, not positions wrapped into `0..127`.

### Applied to the full head_dim

`dim=self.head_dim` = 64, i.e. **all 64 dims are rotated**; there is no partial-rotary fraction. `inv_freq` has 32 entries:

```python
inv_freq, self.attention_scaling = self.rope_init_fn(None, device, dim=dim, base=base)
```
with the default `compute_default_rope_parameters`, i.e. `inv_freq[i] = 1 / base**(2i/dim)` for `i in 0..32`, and `attention_scaling = 1.0` for `rope_type="default"` (no YaRN/NTK scaling in these configs).

### Convention: half-split (`rotate_half`), NOT interleaved

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

and the cos/sin tables are the duplicated-half form:

```python
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()
```

So, with `d = 64` and `h = 32`: pair dim `i` with dim `i + 32`, and freq index `i % 32`. Concretely
`out[i]   = x[i]*cos[i] - x[i+32]*sin[i]` and `out[i+32] = x[i+32]*cos[i] + x[i]*sin[i]`, for `i in 0..32`.

The FA2 unpadded path is also non-interleaved, explicitly:

```python
super().__init__(dim=dim, base=base, pos_idx_in_fp32=True, device=device, interleaved=False)
```

Also note: cos/sin are computed in **float32** under `torch.autocast(..., enabled=False)` and only then cast back to the activation dtype.

### Position ids

```python
if position_ids is None:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
```

Plain `0..seq_len-1`, identical for every layer, and **not** offset by padding. RoPE is applied to Q and K only, never to V.

---

## 5. LayerNorm: type, eps, bias, exact placement

- **True LayerNorm** (mean-subtract + variance normalise), not RMSNorm. The paper: "We also disable all bias terms in Layer Norms." The config docstring for `norm_eps` mistakenly says "rms normalization layers" — the code is `nn.LayerNorm`.
- `eps = norm_eps = 1e-5`.
- `bias = norm_bias = false` -> every norm has **weight only, no bias**. Confirmed by the checkpoint: there is no `*.norm.bias` / `*_norm.bias` tensor anywhere.

### Placement

**Norm after the embeddings** (this is the "norm before the first layer" you asked about):

```python
self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
...
hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)))
```

Paper: "we add a LayerNorm after the embedding layer."

**Layer 0 skips its attention norm: CONFIRMED, in both code and checkpoint.**

```python
class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
```

The safetensors file has `model.layers.0.mlp_norm.weight` but **no** `model.layers.0.attn_norm.weight`; layers 1..27 have both. Paper: "To avoid repetition, we remove the first LayerNorm in the first attention layer."

**Residuals sit outside the norms (standard pre-norm), and there is NO post-attention output norm:**

```python
attn_outputs = self.attn(
    self.attn_norm(hidden_states),
    ...
)
hidden_states = hidden_states + attn_outputs[0]
mlp_output = (
    self.compiled_mlp(hidden_states)
    if self.config.reference_compile
    else self.mlp(self.mlp_norm(hidden_states))
)
hidden_states = hidden_states + mlp_output
```

i.e. per layer:

```
h = h + Attn(attn_norm(h))        # attn_norm = Identity for layer 0
h = h + MLP(mlp_norm(h))
```

There are exactly **two** norms per layer (one for layer 0), both *before* their sub-block, and the residual is added to the **un-normed** input. No norm is applied to the attention output and no norm sits on the residual stream.

**Final norm after the last layer: YES.**

```python
self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
...
hidden_states = self.final_norm(hidden_states)
```
applied after the loop, so `last_hidden_state` is already normalised.

Full norm inventory: `embeddings.norm`, `layers.{1..27}.attn_norm`, `layers.{0..27}.mlp_norm`, `final_norm`, `head.norm` = 1 + 27 + 28 + 1 + 1 = **58 norms**, matching 58 `*norm*.weight` tensors in the checkpoint.

---

## 6. Feed-forward: GeGLU

Confirmed GeGLU, with the **exact erf GELU**, and the split is **`chunk(2)` (contiguous halves), with the activation on the FIRST half**:

```python
class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))
```

Identical in `main` (line 89).

- `Wi`: `[5248, 1024]`. **Rows 0..2623 = the activated branch (`input`), rows 2624..5247 = the multiplicative branch (`gate`).** Contiguous halves, **not** interleaved.
- Note the variable naming is the reverse of most GeGLU/SwiGLU code: the half called `gate` is the one *without* the activation. So: `y = Wo( gelu(Wi_first_half @ x) * (Wi_second_half @ x) )`.
- `Wo`: `[1024, 2624]`, no bias.

### Which GELU

`hidden_activation = "gelu"` -> `ACT2CLS["gelu"] = GELUActivation` (`activations.py`):

```python
ACT2CLS = {
    "gelu": GELUActivation,
    ...
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
```

```python
class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
```

`nn.functional.gelu` with no `approximate=` argument is the **exact erf GELU**: `x * 0.5 * (1 + erf(x / sqrt(2)))`.

**It is NOT the tanh approximation.** `"gelu_pytorch_tanh"` and `"gelu_new"` are the tanh variants and are not what this config selects. `pylate-rs` independently uses `gelu_erf()`.

In Candle terms: `gelu_erf()`, not `gelu()` (Candle's bare `gelu()` is the tanh approximation).

---

## 7. Attention mechanics

| Item | Value |
|---|---|
| QKV bias | **No** (`attention_bias: false`); no `Wqkv.bias` in the checkpoint |
| `Wo` bias | **No** (same flag); no `attn.Wo.bias` in the checkpoint |
| Fused projection | **Yes, a single `Wqkv`** of shape `[3072, 1024]` |
| Scaling | `head_dim ** -0.5` = `64 ** -0.5` = **0.125** |
| Heads | 16 heads x 64 dims, no GQA/MQA (Q, K, V all 16 heads) |
| Causal | No, fully bidirectional |

```python
self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)
...
self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
```

```python
qkv = self.Wqkv(hidden_states)
bs = hidden_states.shape[0]
if self.config._attn_implementation == "flash_attention_2":
    qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
else:
    qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)
```

So the 3072 output features are laid out as `[Q (1024) | K (1024) | V (1024)]`, and within each 1024 block as `[head0 (64) | head1 (64) | ... | head15 (64)]`. **Split `Wqkv` rows into three contiguous 1024-row blocks**, not interleaved per head.

```python
cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
query, key, value = qkv.transpose(3, 1).unbind(dim=2)
# query, key, value: [batch_size, heads, seq_len, head_dim]
query, key = apply_rotary_pos_emb(query, key, cos, sin)

scale = module.head_dim**-0.5
attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

if local_attention != (-1, -1):
    attention_mask = sliding_window_mask

attn_weights = attn_weights + attention_mask

# upcast attention to fp32
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
attn_output = torch.matmul(attn_weights, value)
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(bs, -1, dim)
```

Order of operations: `Wqkv` -> reshape to heads -> RoPE on Q,K -> scaled dot product -> **additive** mask -> softmax **in fp32** -> `@V` -> merge heads -> `Wo`.

The mask is additive with `torch.finfo(dtype).min` (not `-inf`), from `_prepare_4d_attention_mask(attention_mask, self.dtype)`: shape `[bs, 1, 1, seq]` broadcast over queries, masking padded **keys**. The local mask additionally fills `abs(i-j) > 64` with `finfo.min`, giving a `[bs, 1, seq, seq]` tensor.

---

## 8. MLM head

```python
class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))
```

```python
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
```

```python
logits = (
    self.compiled_head(last_hidden_state)
    if self.config.reference_compile
    else self.decoder(self.head(last_hidden_state))
)
```

Full pipeline:

```
x = final_norm(encoder_output)          # already applied inside ModernBertModel
x = head.dense(x)                       # [1024 -> 1024], NO bias (classifier_bias=false)
x = gelu_erf(x)
x = head.norm(x)                        # LayerNorm, eps 1e-5, weight only, no bias
logits = x @ tok_embeddings.weight.T + decoder.bias     # [1024 -> 50368]
```

- **Order is dense -> activation -> norm.** Not dense -> norm -> activation (which is what BERT's own `BertPredictionHeadTransform` does: `dense -> act -> LayerNorm` -- actually the same, but worth not guessing).
- **Decoder weight IS tied to the input embeddings.** `tie_word_embeddings` is `True` (explicit in the plain `ModernBERT-large` config; the Instruct config omits it and the `ModernBertConfig` default is `tie_word_embeddings: bool = True`). The checkpoint confirms it: **`decoder.weight` is absent from the safetensors file.** You must reuse `model.embeddings.tok_embeddings.weight` as the decoder matrix.
- **There IS a decoder bias.** `decoder_bias: true`, and `decoder.bias` of shape `[50368]` is present in the checkpoint. This is the one bias the paper keeps: "we disable bias terms in all linear layers except for the final decoder linear layer."
- `head.dense` has **no** bias and `head.norm` has **no** bias.

---

## 9. Exact safetensors parameter names

Single file `model.safetensors` (no `model.safetensors.index.json` -- that URL returns `Entry not found`; the names below came from reading the safetensors header directly via an HTTP range request). **173 tensors**, metadata `{"format": "pt"}`.

Top-level / non-layer tensors:

```
model.embeddings.tok_embeddings.weight      [50368, 1024]
model.embeddings.norm.weight                [1024]
model.final_norm.weight                     [1024]
head.dense.weight                           [1024, 1024]
head.norm.weight                            [1024]
decoder.bias                                [50368]
```

Per layer, for `N` in `0..=27`:

```
model.layers.{N}.attn_norm.weight           [1024]        # ABSENT for N == 0
model.layers.{N}.attn.Wqkv.weight           [3072, 1024]
model.layers.{N}.attn.Wo.weight             [1024, 1024]
model.layers.{N}.mlp_norm.weight            [1024]
model.layers.{N}.mlp.Wi.weight              [5248, 1024]
model.layers.{N}.mlp.Wo.weight              [1024, 2624]
```

Count check: 6 top-level + 28*6 - 1 (missing `layers.0.attn_norm.weight`) = 6 + 168 - 1 = **173**. Matches.

Notes for the loader:

- There is **no** `decoder.weight`, **no** `*.bias` anywhere except `decoder.bias`, and **no** position-embedding tensor.
- Shapes are PyTorch `nn.Linear` convention `[out_features, in_features]`, so a row-major read gives you `W` such that `y = x @ W.T`. Transpose (or use a transposed matmul) for `y = W @ x`.
- The `ModernBertModel`-only checkpoints (e.g. a `ModernBertModel` save) drop the `model.` prefix and the `head.*` / `decoder.*` tensors. The Instruct checkpoint is `ModernBertForMaskedLM`, so the encoder tensors carry the `model.` prefix. `pylate-rs` loads the un-prefixed names (`embeddings.tok_embeddings`, `layers.{i}.attn.Wqkv`, ...) because it strips/roots the prefix.
- dtype in the Instruct file is **BF16** for every tensor.

---

## 10. Size on disk and parameter count

| | `ModernBERT-Large-Instruct` | `ModernBERT-large` |
|---|---|---|
| Parameters | **395,881,664** (all BF16) | **395,881,664** (all F32) |
| `model.safetensors` | **791,781,544 bytes** (755.1 MiB), bf16 | **1,583,544,840 bytes** (1.475 GiB), fp32 |
| `pytorch_model.bin` | not present | 1,583,581,182 bytes |

- fp32 payload is exactly `4 * 395,881,664 = 1,583,526,656` bytes; the remaining ~18 KB is the safetensors header (8-byte length + 18,208-byte JSON).
- The **fp32 figure you asked for is ~1.47 GiB**. Be aware the Instruct checkpoint itself ships as **bf16 (755 MiB)**; if you want fp32 in memory you upcast on load, you do not download fp32.
- The 395.9M total counts `tok_embeddings` **once** (51,576,832 params = 13% of the model). Because the decoder is tied, there is no second copy.
- Hand-verified breakdown summing to 395,881,664: embeddings 51,576,832 + embeddings.norm 1,024 + 28 layers at 12,257,280 each minus 1,024 for layer 0's missing `attn_norm` (= 343,202,816) + final_norm 1,024 + head.dense 1,048,576 + head.norm 1,024 + decoder.bias 50,368.
- Paper agrees: "a total parameter count of 149 and 395 million, respectively".

---

## 11. Traps: what a naive reimplementation gets wrong

Ordered roughly by how likely each is to bite and how hard it is to notice.

1. **`local_attention: 128` is the TOTAL window, so 64 each side, and the comparison is inclusive -> 129 keys per query.** Three separate off-by-ones live here. Using 128 each side doubles the receptive field of 18 of 28 layers; using `< 64` instead of `<= 64` silently clips one key on each side. Also do **not** copy `main`'s `self.sliding_window = config.sliding_window + 1` (= 65): that `+1` only cancels a `-1` inside the flash-attention wrapper. Correct predicate: `(i as i64 - j as i64).abs() <= 64`.

2. **GELU is the exact erf form, not the tanh approximation.** `nn.functional.gelu` with no `approximate=` arg. In Candle, that is `gelu_erf()`; Candle's bare `gelu()` is the tanh approximation. The difference is ~1e-3 absolute per activation, which compounds across 28 GeGLU layers and 28 attention blocks and will visibly move MLM logits and change argmax on close calls. This bites in **two** places: the MLP and `head.act`.

3. **The GeGLU halves are named backwards relative to most implementations.** `input, gate = Wi(x).chunk(2, dim=-1)` then `act(input) * gate`. The **first** half gets the GELU; the second half is the plain linear multiplier. If you follow the LLaMA/SwiGLU habit of activating the "gate" (= second half), you get the halves swapped and the output is wrong but not obviously NaN-y. Contiguous `chunk(2)`, i.e. rows `0..2624` and `2624..5248` of `Wi` — **not** interleaved.

4. **Layer 0 has no `attn_norm`.** The tensor is simply missing from the checkpoint. A loader that `expect()`s it on every layer fails loudly (fine), but a loader that defaults a missing norm weight to ones applies a LayerNorm that should not be there (normalising the already-normalised embedding output a second time) and is silently wrong. Make it `Option` and skip the whole LayerNorm, not just the scale.

5. **Local and global layers use genuinely different RoPE theta: 10,000 vs 160,000.** Build two cos/sin tables. Using one theta for all layers is the single easiest way to produce plausible-looking but wrong embeddings. Which theta is which: **local = 10,000, global = 160,000** (the long-range layers get the large base). Note the counterintuitive pairing — the layers with the *small* window get the *small* theta.

6. **Layer 0 is GLOBAL, not local.** `layer_id % 3 == 0`. It is tempting to read "every third layer is global" as 2, 5, 8, ... Global set is `{0, 3, 6, 9, 12, 15, 18, 21, 24, 27}`. Layer 27 (the last) is global too.

7. **RoPE is the half-split `rotate_half` convention, not interleaved.** Pair dim `i` with dim `i+32`, frequency index `i % 32`, cos/sin tables built as `cat((freqs, freqs), -1)`. An interleaved implementation (pair `2i` with `2i+1`) produces garbage that still has the right norms and will not obviously crash. The FA2 path spells it out: `interleaved=False`.

8. **RoPE covers the full 64-dim head.** There is no partial-rotary fraction and no `rope_pct`/`partial_rotary_factor` in this model. Don't carry over a 25%/50% rotary habit from GPT-NeoX/Phi.

9. **Local layers use true absolute positions.** The `max_position_embeddings = config.local_attention` on the local branch only sizes a precomputed cache for the FA2 kernel. Positions are **not** wrapped into `0..127` and are not window-relative. Both layer types see `position_ids = 0..seq_len-1`.

10. **There is a LayerNorm right after the token embeddings.** Easy to miss because it has no analogue in vanilla BERT's pre-norm variants. `embeddings.norm` is a real trained tensor; skipping it is a large error at the input.

11. **There is a `final_norm` after the last layer**, and the MLM head then applies `head.norm` after its own dense+act. So on the MLM path you pass through two more LayerNorms after layer 27. Forgetting `final_norm` is a classic.

12. **No absolute position embeddings exist, despite `"position_embedding_type": "absolute"` in the config.** That field is a vestigial copy-paste; the code never reads it and there is no position-embedding tensor in the checkpoint. Position information comes solely from RoPE. Don't go looking for `position_embeddings.weight`.

13. **`layer_norm_eps` in the config is also vestigial.** The code reads `norm_eps`. Both happen to be `1e-5` here, so it costs nothing, but read `norm_eps`.

14. **Norms are true LayerNorm, not RMSNorm**, despite the `ModernBertConfig` docstring saying "The epsilon used by the rms normalization layers". You must subtract the mean. The paper is explicit that ModernBERT uses LayerNorm. (And `norm_bias: false` means weight-only LayerNorm, which looks a lot like RMSNorm in a state dict — one tensor per norm — so the state dict shape does not disambiguate it for you.)

15. **Residuals add to the un-normed input.** `h = h + Attn(attn_norm(h))`, then `h = h + MLP(mlp_norm(h))`. A post-norm or "sandwich" arrangement, or adding the normed value into the residual, is wrong. There is **no** post-attention output norm and no norm on the residual stream.

16. **`decoder.weight` does not exist; it is tied to `tok_embeddings.weight`.** `tie_word_embeddings` is absent from the Instruct config and defaults to `True`. If your loader requires every linear to have a weight, this fails. If it defaults missing weights to anything, you get noise logits.

17. **`decoder.bias` DOES exist** (shape `[50368]`) even though literally every other bias in the model is disabled. Dropping it shifts all MLM logits per-token-type and changes the argmax. This matters a lot for ModernBERT-Large-Instruct, whose whole inference protocol is reading the top-k at a `[MASK]` position.

18. **`head.dense` has no bias** (`classifier_bias: false`), unlike BERT's `cls.predictions.transform.dense` which does. Don't add one.

19. **Attention scale is applied, and only once.** `head_dim ** -0.5 = 0.125`. HF scales the logits after the matmul; `pylate-rs` scales `q` before. Equivalent in exact arithmetic; pick one and don't do both.

20. **Softmax is computed in fp32 and then cast back.** `softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`. If you compute in bf16 you will see a few-1e-3 drift per layer. Likewise cos/sin are built in fp32 under `autocast(enabled=False)`. For a reference-matching Rust implementation, do your accumulations in f32 even if weights are bf16.

21. **`Wqkv` splits into three contiguous 1024-row blocks `[Q | K | V]`, head-major within each.** The reshape is `view(bs, seq, 3, n_heads, head_dim)`, so the "3" is the *outer* of (3, heads, dim). A per-head-interleaved `[q0 k0 v0 q1 k1 v1 ...]` split (which some fused-QKV checkpoints use) is wrong here.

22. **The padding mask uses `finfo(dtype).min`, not `-inf`**, and is `[bs, 1, 1, seq]` — it masks padded **keys** only, never padded **queries**. So rows of the attention matrix belonging to pad positions are *not* masked and produce finite junk that flows through the whole network. That is harmless as long as you ignore the outputs at pad positions, but if you ever reduce over the sequence (mean pooling, or a `classifier_pooling: "mean"` head) you must exclude pad positions yourself. Using true `-inf` risks NaN if a row ever ends up fully masked.

23. **Local and global layers get *different mask tensors*, and the local mask already includes the padding mask.** In HF, `sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), min)`. If you build the window mask alone and forget to OR in the padding mask for local layers, padding leaks into local attention. Build both masks once outside the layer loop and select by layer type.

24. **`model.` prefix.** The Instruct checkpoint is a `ModernBertForMaskedLM`, so encoder tensors are `model.layers.*`, `model.embeddings.*`, `model.final_norm.*`, while the head tensors are *not* under `model.` (`head.dense.weight`, `decoder.bias`). A `ModernBertModel`-only checkpoint drops the `model.` prefix entirely. Handle both or pin to the one you load.

25. **The intermediate size is 2624, an odd-looking non-multiple of 1024,** and `Wi` is `5248 = 2 * 2624`. Don't "round" it or assume `4 * hidden_size`. The GLU doubling is why 2624 appears twice in the shapes (`Wi: [5248, 1024]`, `mlp.Wo: [1024, 2624]`).

26. **`vocab_size = 50368` is larger than the tokenizer's real vocabulary** (50,285 real tokens plus 83 unused, padded to a multiple of 64 for GPU efficiency). Logits for the unused ids are meaningless; if you argmax over the full 50,368 you can in principle return a token the tokenizer cannot decode. Worth masking for MLM decode.

27. **`tok_embeddings` is constructed with `padding_idx=config.pad_token_id`.** `padding_idx` in PyTorch only zeroes *gradients*; it does **not** zero the embedding row at inference. The row in the checkpoint holds whatever it was trained to. Do not zero it in your loader.

28. **No bias terms means no bias tensors to find, which can mask a loading bug.** With `attention_bias`, `mlp_bias`, `norm_bias` and `classifier_bias` all `false`, the only bias in the whole 173-tensor file is `decoder.bias`. If your loader silently substitutes zeros for missing biases, every bias is correctly zero *except* the one that matters (trap 17).

29. **`reference_compile: true` in the Instruct config is a `torch.compile` flag, not architecture.** It changes nothing semantically. Likewise `deterministic_flash_attn`, `sparse_prediction`, `repad_logits_with_grad`, `gradient_checkpointing`, and the `*_dropout` fields (all 0.0) are inference no-ops.

30. **`classifier_pooling: "mean"`** applies only to `ModernBertForSequenceClassification`, and the HF docstring warns why it is set that way: "In local attention layers, the CLS token doesn't attend to all tokens on long sequences." Irrelevant to the MLM path you are building, but it tells you not to trust CLS as a whole-sequence summary in this model.

### Could not be determined

- `model.safetensors.index.json` does not exist for `ModernBERT-Large-Instruct` (single-shard model); the URL returns `Entry not found`. Parameter names above were read from the safetensors header directly, which is authoritative.
- The paper does not state whether the 128-token window is total or per-side; only the code does (total).
