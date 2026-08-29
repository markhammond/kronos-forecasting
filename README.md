# Kronos.Net

Native .NET inference for [Kronos](https://github.com/shiyu-coder/Kronos), a pre-trained
autoregressive model over K-line (OHLCVA) sequences. Runs on CPU or Apple GPU via
TorchSharp.

## Why this exists

Kronos is not a text LLM, so the existing .NET inference runtimes cannot load it. It pairs
a Binary Spherical Quantization autoencoder with a decoder that has a hierarchical
two-subtoken embedding, a dependency-aware cross-attention layer and two conditional
heads. GGUF has no representation for any of that, which rules out llama.cpp-based
runtimes; and the alternatives that *do* reach Apple's GPU only load GGUF. The
architecture had to be ported.

## Install

```
dotnet add package Kronos.Net
dotnet add package Kronos.Net.Weights.Small     # or .Mini
dotnet add package TorchSharp-cpu               # or a CUDA backend
```

The weights ship separately so that choosing a model is a package reference rather than a
rebuild, and so this package stays small enough to be a reasonable dependency.

**Choose your own backend.** This package depends on TorchSharp but does not propagate a
native runtime, which is platform- and accelerator-specific. `TorchSharp-cpu` resolves the
right one per RID; see [TorchSharp's
download guidance](https://github.com/dotnet/TorchSharp#download) for CUDA.

## Use

```csharp
using Kronos.Net;
using Kronos.Net.Weights;
using static TorchSharp.torch;

using var forecaster = KronosForecaster.Load(KronosSmall.Instance, new Device("mps"));
// Hosts typically probe: try "cuda", then "mps", then fall back to "cpu". Probe by
// placing a tensor and catching — the package name does not tell you what is available.

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

Buffers are caller-supplied and sized from `OutputCount`; a mismatch throws with the
expected count rather than silently misaligning every row. `dispersion` may be empty,
in which case the per-window standard deviation is not computed at all.

`IKronosCheckpoint` is the weights abstraction. `EmbeddedCheckpoint` (used by the weights
packages) reads from assembly resources, so nothing resolves through configuration or the
filesystem — which is what lets this sit behind a consumer forbidden from reading its
environment. `DirectoryCheckpoint` loads a published snapshot layout for development.

## Performance

Parity with PyTorch when using CPU and GPU (on Apple Silicon).

## Things that will bite you

**`DisposeScope` is mandatory.** TorchSharp has no refcounting. Every forward pass this
library performs is scoped; if you write your own, wrap it in
`using var _ = torch.NewDisposeScope()`. Without it the Metal allocator thrashes and you
will measure roughly a 10x slowdown that looks like a backend problem and is not.

**Cached tensors need `DetachFromDisposeScope()`, not `MoveToOuterDisposeScope()`.** The
latter hands ownership to the caller's scope, which frees it on exit — so a cache is
released after its first use.

**Metal is present in the "cpu" backend.** On Apple Silicon "cpu" means *not CUDA*. Probe
by placing a tensor rather than trusting the name.

**No key-value cache.** Attention is recomputed over the whole context at every decode
step, matching the reference implementation. A cache would change the arithmetic and
forfeit checkable parity; it costs `horizon` full passes per rollout, which is affordable
only for short horizons.

## Parity

Verified stage by stage against the reference implementation on CPU and Metal. The
tokenizer's embedding and first norm are bit-exact; relative error thereafter is 1e-7 to
1e-6, ordinary float32 accumulation.

It is *not* bit-identical, and does not claim to be: cross-implementation token
disagreement extrapolates to near 1 in 15,000. Within one implementation the result is
deterministic, which is usually the property that matters.

Sampling differs deliberately: this library draws per-bar uniforms from a SplitMix64
stream seeded by the bar's own timestamp and samples by inverse CDF, so a draw does not
depend on how bars were grouped into batches. The reference seeds one stream per batch and
is therefore not batch-invariant. Distributionally identical; the stream is not.

## Attribution

The model architecture, tokenizer and reference implementation are the work of
[ShiYu](https://github.com/shiyu-coder/Kronos) (MIT). Published checkpoints are by
NeoQuasar (MIT) and are redistributed unmodified in the `Kronos.Net.Weights.*` packages,
which record the revision each was built from. This project is an independent port and is
not affiliated with or endorsed by either.
