# Changelog

## 0.1.1 — 2026-09-02

### Projected paths

- `KronosForecaster.InferPath` writes the projected OHLCVA as `rows x horizon x 6`,
  sized by `PathLength`. The decode always produced all six channels; only the close
  was surfaced.
- Rollouts are reduced per channel per step by a caller-supplied `RolloutAggregator`
  (`ReadOnlySpan<float> -> float`), which is the shape the reference uses internally
  (`np.mean(preds, axis=1)`). `Mean` is the default, matching it; `Median`,
  `AtQuantile` and `Rollout` are provided. Samples are ordered by rollout index.
- Paths are frequently not well-formed candles, and this is the model rather than the
  port: channels decode independently with nothing tying `high` to `close`. A single
  rollout broke ordering in roughly 30% to 70% of candles depending on the checkpoint,
  smaller ones being worse; averaging across rollouts reduces that severalfold, cancelling
  the noise rather than enforcing anything.
  The reference does no post-processing either, and its headline example runs
  `sample_count=1`. Clamp if you need valid candles.

### Documentation reaches consumers

- `GenerateDocumentationFile` was never set, so 0.1.0 shipped no `.xml` and gave
  consumers no IntelliSense beyond signatures. All five packages now carry it.
- `Infer` documents every parameter, including which channels are filled rather than
  ignored — the model reads six and has no mask, and the reference derives amount as
  `volume * mean(open, high, low, close)`.
- Both forecasters record that inference is not thread safe.

### Checkpoint limits

- `ICheckpoint.MaxContext` states the longest context a checkpoint can attend over —
  2048 for Kronos-mini, 512 for Kronos-small and -base. It varies by checkpoint, so it
  is not a constant of the architecture.
- `Infer` refuses a context beyond that limit rather than accepting it. Upstream
  truncates silently, which reads as a working call that quietly ignored the excess.
- `KronosForecaster.Channels` names the six-channel layout that was previously a bare
  literal at every call site, and documents the row-major order with an example.

### Allocation and API shape

- Per-window scratch is rented once and passed in pre-sliced rather than allocated per
  window, and the per-batch `ToArray()` in the sampler is gone.
- **Breaking:** `WriteStamp` takes `Span<float>` rather than `float[]`. Source-compatible
  for array callers, binary-breaking against 0.1.0.

The Kronos and TimesFM inference paths are otherwise unchanged and still parity-verified.

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
