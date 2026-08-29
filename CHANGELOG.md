# Changelog

## 0.1.0 — 2026-08-29

First release.

- Kronos tokenizer (encode and decode) and autoregressive model, ported to TorchSharp
- `Safetensors` reader — checkpoints load without Python at build or run time
- `IKronosCheckpoint`: weights as streams, from embedded resources or a directory
- `Kronos.Forecasting.Weights.Small` and `.Mini`, checkpoints embedded at pinned revisions
- Verified stage by stage against the reference on CPU and Metal; see README

Known limitations, all deliberate and documented in the README: no key-value cache, so
attention is recomputed per decode step to preserve checkable parity; results are not
bit-identical across devices; the sampling stream diverges from the reference in order to
be batch-invariant, which the reference is not.
