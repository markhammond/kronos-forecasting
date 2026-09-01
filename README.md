# Tsfm.Forecasting

Native .NET inference for time-series foundation models, on CPU or GPU via
TorchSharp. Currently [Kronos](https://github.com/shiyu-coder/Kronos) and [TimesFM 3.0](https://github.com/google-research/timesfm).

[![ci](https://github.com/markhammond/tsfm-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/markhammond/tsfm-forecasting/actions/workflows/ci.yml)

Targets .NET 10

NuGet packages:

| Package | Licence | |
|---|---|---|
| Tsfm.Forecasting [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.svg)](https://www.nuget.org/packages/Tsfm.Forecasting) | MIT | safetensors reader, checkpoint abstraction |
| Tsfm.Forecasting.Kronos [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos) | MIT | tokenizer, model, forecaster |
| Tsfm.Forecasting.Kronos.Weights.Small [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.Weights.Small.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos.Weights.Small) | MIT | Kronos-small (24.7M) + Tokenizer-base |
| Tsfm.Forecasting.Kronos.Weights.Mini [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.Weights.Mini.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos.Weights.Mini) | MIT | Kronos-mini (4.1M) + Tokenizer-2k |
| Tsfm.Forecasting.TimesFm [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.TimesFm.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.TimesFm) | **Apache-2.0** | TimesFM 3.0 — **no weights** |

Refer to respective package for license and conditions of use (if any).

## Why this exists

Neither model is a text LLM, so existing .NET inference runtimes cannot load them.
Kronos pairs a Binary Spherical Quantization autoencoder with a decoder that has a
hierarchical two-subtoken embedding, a dependency-aware cross-attention layer and two
conditional heads. GGUF has no representation for any of that, which rules out
llama.cpp-based runtimes; the alternatives that *do* reach Apple's GPU load GGUF only.
The architectures had to be ported.

## Kronos — batteries included

```
dotnet add package Tsfm.Forecasting.Kronos
dotnet add package Tsfm.Forecasting.Kronos.Weights.Small   # or .Mini
dotnet add package TorchSharp-cpu                          # or a CUDA backend
```

```csharp
using Tsfm.Forecasting.Kronos;
using Tsfm.Forecasting.Kronos.Weights;
using static TorchSharp.torch;

using var forecaster = KronosForecaster.Load(KronosSmall.Instance, new Device("mps"));

var rows = KronosForecaster.OutputCount(barCount, contextBars: 384);
Span<float> lean = new float[rows];
Span<int>   upCount = new int[rows];

forecaster.Infer(
    ohlcva,          // row-major [L x 6]: open, high, low, close, volume, amount
    barTimeMs,       // [L] Unix ms, one per bar
    lean, upCount, dispersion: default,
    contextBars: 384, horizon: 1, rollouts: 30,
    greedy: false, temperature: 1f, topP: 1f);
```

### Supplying fewer than six channels

The model always reads six channels and has no mask, so an absent one is **filled, not
ignored** — whatever you supply is read as data. There is a single entry point rather
than `ohlc`/`ohlcv` overloads, because choosing the filler is a modelling decision and
belongs at the call site. These are the fills the reference implementation uses:

| You have | volume | amount |
|---|---|---|
| OHLC | `0` | `0` |
| OHLCV | as given | `volume * mean(open, high, low, close)` |
| OHLCVA | as given | as given |

Note the mean of all four prices, not the close. The reference rejects NaN in any of the
six rather than treating it as absent, so fill explicitly.

Checkpoints are embedded as assembly resources, so nothing resolves through
configuration or the filesystem — which is what lets this sit behind a consumer
forbidden from reading its environment. Buffers are caller-supplied and sized from
`OutputCount`; a mismatch throws with the expected count rather than silently
misaligning every row.

## TimesFM — bring your own weights

> **The weights are not open source and are not distributed here.** TimesFM 3.0
> checkpoints are published under the **TimesFM Non-Commercial License v1.0**: research
> and evaluation only, with revenue-generating activity and production deployment
> expressly forbidden. This package's Apache-2.0 licence grants no rights in them
> whatsoever. Commercial use needs terms from Google.

```
dotnet add package Tsfm.Forecasting.TimesFm
dotnet add package TorchSharp-cpu
./scripts/fetch-timesfm-checkpoint.sh    # ~1.2 GB — read the licence first
```

```csharp
using Tsfm.Forecasting.TimesFm;

var forecaster = TimesFmForecaster.Load("checkpoints/timesfm-3.0-pytorch", new Device("mps"));
double[,] q = forecaster.Forecast(ohlcva, horizon: 4);   // [step, quantile]
```

One forward pass yields 64 steps at all nine quantiles, so a full predictive interval
costs no more than a point forecast. Every OHLCVA channel is supplied as a variate and
marked a target: preprocessing masks future covariate slots only for target variates, so
a channel left non-target would be handed its own future values.

## Choose your own backend

Neither model package propagates a native runtime, because the right one is platform-
and accelerator-specific. `TorchSharp-cpu` resolves it per RID; see
[TorchSharp's download guidance](https://github.com/dotnet/TorchSharp#download) for CUDA.

## Things that will bite you

**`DisposeScope` is mandatory.** TorchSharp has no refcounting. Every forward pass these
libraries perform is scoped; if you write your own, wrap it in
`using var _ = torch.NewDisposeScope()`. Without it the Metal allocator thrashes and you
will measure roughly a 10x slowdown that looks like a backend problem and is not.

**Cached tensors need `DetachFromDisposeScope()`, not `MoveToOuterDisposeScope()`.** The
latter hands ownership to the caller's scope, which frees it on exit — so a cache is
released after its first use.

**Metal is present in the "cpu" backend.** On Apple Silicon "cpu" means *not CUDA*. Probe
by placing a tensor rather than trusting the name.

**macOS needs Homebrew's libomp.** `libtorch_cpu.dylib` links
`/opt/homebrew/opt/libomp/lib/libomp.dylib` by absolute path, so the copy shipped inside
the NuGet package is never used. Without it, loading fails with a message claiming the
backend reference is missing — which it is not.

**Neither model has a key-value cache.** Attention is recomputed over the whole context
at every decode step, matching both references. A cache would change the arithmetic and
forfeit checkable parity. For Kronos this costs `horizon` full passes per rollout, so
long horizons are expensive; TimesFM emits its whole horizon in one pass and is
unaffected.

## Parity

Both ports are verified stage by stage against their references, on CPU and Metal,
rather than judged on final outputs — agreement at the end cannot distinguish a wrong
attention scale from a misplaced norm, since both merely shift the result.

**Kronos.** The tokenizer's embedding and first norm are bit-exact; relative error
thereafter is 1e-7 to 1e-6, ordinary float32 accumulation. It is *not* bit-identical and
does not claim to be: cross-implementation token disagreement extrapolates to near 1 in
15,000. Sampling differs deliberately — per-bar uniforms from a SplitMix64 stream seeded
by the bar's own timestamp, selected by inverse CDF, so a draw does not depend on how
bars were grouped into batches. The reference seeds one stream per batch and is not
batch-invariant. Distributionally identical; the stream is not.

**TimesFM.** Relative error is ~1e-7 through preprocessing and the layer stack, ~2e-6 at
the logits. Two faults the harness caught, neither of which throws and both of which
produce entirely plausible forecasts: `scaled_dot_product_attention` ignores `is_casual`
once `attn_mask` is supplied, so supplying a patch mask silently disabled causal masking
and let every position attend to its own future; and the reference leaves
`rescale_logits` false, making its logits `QK^T * sqrt(d)` rather than the conventional
`QK^T / sqrt(d)`.

```bash
./scripts/fetch-checkpoints.sh                       # Kronos
dotnet test tests/Tsfm.Forecasting.Kronos.Tests -c Release

./scripts/fetch-timesfm-checkpoint.sh                # TimesFM, non-commercial
./scripts/fetch-timesfm-reference.sh
python3 -m venv .venv && ./.venv/bin/pip install torch safetensors numpy huggingface_hub
./.venv/bin/python reference/dump_timesfm_parity.py
dotnet run --project tests/Tsfm.Forecasting.TimesFm.Parity -c Release
```

## Attribution

**Kronos.** Model architecture, BSQ tokenizer and reference implementation by
[ShiYu](https://github.com/shiyu-coder/Kronos) (MIT). Published checkpoints are by
NeoQuasar (MIT), redistributed unmodified in the `Tsfm.Forecasting.Kronos.Weights.*`
packages, which record the revision each was built from.

**TimesFM.** Derived from
[google-research/timesfm](https://github.com/google-research/timesfm), Copyright 2026
Google LLC, Apache-2.0. No Google source files and no weights are redistributed here;
both are fetched at pinned revisions by the scripts above.

This project is an independent port and is not affiliated with or endorsed by either.
See NOTICE for the full attribution and the statement of changes.
