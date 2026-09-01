# Changelog

## 0.1.0 — 2026-08-30

First release of the family.

Supersedes `Kronos.Forecasting` 0.1.0, which was published under the previous name and
is deprecated in favour of `Tsfm.Forecasting.Kronos`. The Kronos code is unchanged; only
the package identity and namespace moved.

### Tsfm.Forecasting (MIT)

- `Safetensors` reader — checkpoints load without Python at build or run time
- `ICheckpoint`: weights as streams, from embedded resources or a directory

### Tsfm.Forecasting.Kronos (MIT)

- Kronos tokenizer (encode and decode) and autoregressive model, ported to TorchSharp
- `Tsfm.Forecasting.Kronos.Weights.Small` and `.Mini`, checkpoints embedded at pinned
  revisions
- Verified stage by stage against the reference on CPU and Metal

### Tsfm.Forecasting.TimesFm (Apache-2.0)

- TimesFM 3.0 inference: patch embedding, the mixing transformer stack (sequence
  attention, variate attention, feed-forward), quantile head, and the surrounding
  preprocessing — cumulative running statistics, RevIN, future-covariate rolling
- OHLCVA supplied as variates; one forward pass yields 64 steps at nine quantiles
- Verified stage by stage against the reference on CPU and Metal
- **No weights, and none can be published.** The checkpoints are non-commercial; see
  README and NOTICE

Licences differ per package because the ports derive from differently licensed sources.
Neural modules are deliberately not shared between the two adapters: RmsNorm, RoPE and
attention differ in exactly the details that parity checking exposed, and unifying them
would put a verified port at risk to save little.

Known limitations, all deliberate and documented in the README: neither model has a
key-value cache, so attention is recomputed per decode step to preserve checkable
parity; results are not bit-identical across devices; and Kronos's sampling stream
diverges from its reference in order to be batch-invariant, which the reference is not.
