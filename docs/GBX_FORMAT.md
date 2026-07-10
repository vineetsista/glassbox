# GBX weight format (version GBX1)

The single-file bundle consumed by the C++ engine (native and WASM). One
fetch gives the browser everything: model weights, tokenizer, and SAE.
Produced by `scripts/export_gbx.py`; parsed by `engine/src/gbx.hpp`.

All integers and floats are **little-endian**. Tensor data is **fp32**,
row-major, in the exact layout of the PyTorch `state_dict` tensors.

## Layout

| offset | size | content |
|---|---|---|
| 0 | 4 | magic bytes `GBX1` |
| 4 | 4 | `u32 header_len` — byte length of the JSON header |
| 8 | header_len | header JSON (utf-8, no BOM) |
| 8 + header_len | 0-63 | zero padding to the next 64-byte boundary |
| aligned | ... | tensor data region; each tensor starts 64-byte aligned |

Tensor `offset` fields in the manifest are **relative to the start of the
tensor data region** (i.e. after the post-header padding), not the file.

## Header JSON

```json
{
  "config": {
    "vocab_size": 4096, "d_model": 192, "n_layers": 4, "n_heads": 6,
    "d_mlp": 768, "ctx_len": 256, "rope_base": 10000.0, "rms_eps": 1e-05
  },
  "tokenizer": { "merges": [[116, 104], ...], "vocab_size": 4096 },
  "sae": {
    "d_in": 192, "expansion": 8, "k": 32, "n_features": 1536, "hook_layer": 2
  },
  "tensors": [
    { "name": "wte", "shape": [4096, 192], "offset": 0, "nbytes": 3145728 },
    ...
  ]
}
```

`sae` is `null` when the bundle carries no autoencoder (steering disabled).

## Tensor names

| name | shape | note |
|---|---|---|
| `wte` | `[vocab, d_model]` | tied embedding / unembedding |
| `blocks.{i}.norm1` | `[d_model]` | RMSNorm gain (attention) |
| `blocks.{i}.w_q` `w_k` `w_v` `w_o` | `[d_model, d_model]` | torch Linear layout: `y = W @ x` per row |
| `blocks.{i}.norm2` | `[d_model]` | RMSNorm gain (MLP) |
| `blocks.{i}.w_in` | `[d_mlp, d_model]` | |
| `blocks.{i}.w_out` | `[d_model, d_mlp]` | |
| `final_norm` | `[d_model]` | |
| `sae.w_enc` | `[n_features, d_in]` | |
| `sae.b_enc` | `[n_features]` | |
| `sae.w_dec` | `[d_in, n_features]` | unit-norm columns |
| `sae.b_dec` | `[d_in]` | |

## Tokenizer

Byte-level BPE: ids 0-255 are raw bytes, id `256 + r` is merge rank `r`,
id `vocab_size - 1` is `<|endoftext|>` (never produced by encoding). The
merge list fully determines the vocabulary; runtimes rebuild token byte
strings by replaying merges.

## Design notes

- fp32 everywhere: at 2.6M params the bundle is ~12MB, small enough that
  quantization is not worth the parity risk for this project.
- 64-byte alignment lets the native loader hand out raw pointers safely and
  keeps SIMD loads aligned.
- The header is JSON so the Python exporter, C++ engine, and any future JS
  inspector all parse the same source of truth. The C++ side uses the
  hand-rolled parser in `engine/src/json.hpp` (zero-dependency rule).
