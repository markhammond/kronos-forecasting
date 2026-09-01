using static TorchSharp.torch;
using Tsfm.Forecasting;
using Tsfm.Forecasting.Kronos;

// Parity harness. Each stage is compared against tensors exported from the reference,
// and — where a stage feeds a discrete decision — the margin to the nearest flip is
// reported too, because agreement on a wide margin says nothing about faithfulness.
static string Env(string name)
    => Environment.GetEnvironmentVariable(name)
       ?? throw new InvalidOperationException(
           $"{name} is unset. Required: KRONOS_TOKENIZER_DIR, KRONOS_MODEL_DIR, KRONOS_FIXTURE; "
           + "optional: KRONOS_DEVICE (default cpu).");

var tokDir = Env("KRONOS_TOKENIZER_DIR");
var mdlDir = Env("KRONOS_MODEL_DIR");
var fixture = Env("KRONOS_FIXTURE");
var device = new Device(Environment.GetEnvironmentVariable("KRONOS_DEVICE") ?? "cpu");

using var _ = no_grad();
var checkpoint = new DirectoryCheckpoint(mdlDir, tokDir);
var tok = KronosTokenizerEncoder.FromCheckpoint(checkpoint, device);
var mdl = KronosModel.FromCheckpoint(checkpoint, device);
using var fx = Safetensors.Load(fixture);
Tensor E(string n) => fx.Get(n).to(device);

static (float abs, float rel) Cmp(Tensor a, Tensor b)
    => ((a - b).abs().max().item<float>(),
        ((a - b).abs().max() / b.abs().max().clamp(min: 1e-12)).item<float>());

Console.WriteLine($"device={device.type}");
Console.WriteLine($"{"stage",-26} {"max|Δ|",12} {"rel",12}  note");

var s1 = E("s1"); var s2 = E("s2"); var stamp = E("stamp");
var ok = true;

// 1. tokenizer decode — indices back to continuous bars
var recon = tok.Decode(s1, s2);
var (ra, rr) = Cmp(recon, E("recon"));
Console.WriteLine($"{"tokenizer decode",-26} {ra,12:E3} {rr,12:E3}");

// 2. the transformer stack
var (logits, ctx) = mdl.DecodeS1(s1, s2, stamp);
var (ca, cr) = Cmp(ctx, E("context"));
var (la, lr) = Cmp(logits, E("s1_logits"));
Console.WriteLine($"{"decode_s1 context",-26} {ca,12:E3} {cr,12:E3}");
Console.WriteLine($"{"decode_s1 logits",-26} {la,12:E3} {lr,12:E3}");

// The logits only matter through the token they select, so compare the argmax and
// report how far ahead the winner was.
var expL = E("s1_logits");
var gotArg = logits.argmax(-1); var expArg = expL.argmax(-1);
var argMatch = gotArg.eq(expArg).sum().item<long>();
var topTwo = expL.topk(2, dim: -1).values;
var gap = (topTwo.select(-1, 0) - topTwo.select(-1, 1)).min().item<float>();
Console.WriteLine($"{"  -> argmax agreement",-26} {argMatch,12}/{expArg.numel(),-11} smallest top-1/top-2 gap {gap:F3}");
ok &= argMatch == expArg.numel();

// 3. the conditional head
var s2Logits = mdl.DecodeS2(ctx, E("samp"));
var (sa, sr) = Cmp(s2Logits, E("s2_logits"));
var g2 = s2Logits.argmax(-1); var e2 = E("s2_logits").argmax(-1);
var m2 = g2.eq(e2).sum().item<long>();
Console.WriteLine($"{"decode_s2 logits",-26} {sa,12:E3} {sr,12:E3}");
Console.WriteLine($"{"  -> argmax agreement",-26} {m2,12}/{e2.numel(),-11}");
ok &= m2 == e2.numel();

// 4. end-to-end: raw bars -> per-window lean, greedy, against the reference pipeline
var engineFixture = Environment.GetEnvironmentVariable("KRONOS_ENGINE_FIXTURE");
if (engineFixture is not null)
{
    using var ef = Safetensors.Load(engineFixture);
    var raw = ef.Get("ohlcva");
    var times = ef.Get("bar_time_ms");
    var expLean = ef.Get("lean");

    using var engine = KronosForecaster.Load(checkpoint, device);
    var bars = raw.reshape(-1).to(ScalarType.Float32).data<float>().ToArray();
    var ts = times.data<long>().ToArray();
    var rows = KronosForecaster.OutputCount(ts.Length, 384);

    var leanBuf = new float[rows];
    var upBuf = new int[rows];
    var dispBuf = new float[rows];

    // The published-factor path passes an EMPTY dispersion span, so exercise that shape
    // here — it is the one the indicator actually uses, and a contract mismatch on it
    // degrades every bar to the neutral state rather than throwing anywhere visible.
    var okHot = engine.Infer(bars, ts, leanBuf, upBuf, Span<float>.Empty,
        contextBars: 384, horizon: 1, rollouts: 1, greedy: true, temperature: 1f, topP: 1f, batch: 8);
    var hotLean = (float[])leanBuf.Clone();

    // ...then the diagnostic shape, which must agree with it bar for bar.
    var okTrace = engine.Infer(bars, ts, leanBuf, upBuf, dispBuf,
        contextBars: 384, horizon: 1, rollouts: 1, greedy: true, temperature: 1f, topP: 1f, batch: 8);

    var shapesAgree = okHot && okTrace;
    for (var i = 0; i < rows; i++) shapesAgree &= hotLean[i].Equals(leanBuf[i]);
    Console.WriteLine($"{"  -> empty vs full dispersion",-26} {(shapesAgree ? "identical" : "DIVERGED"),12}");
    ok &= shapesAgree;

    var res = new { Lean = leanBuf };
    var n = (int)expLean.numel();
    var e = expLean.data<float>().ToArray();
    float worstAbs = 0, worstRel = 0;
    var signAgree = 0;
    for (var i = 0; i < n; i++)
    {
        var d = Math.Abs(res.Lean[i] - e[i]);
        worstAbs = Math.Max(worstAbs, d);
        if (Math.Abs(e[i]) > 1e-9f) worstRel = Math.Max(worstRel, d / Math.Abs(e[i]));
        if (Math.Sign(res.Lean[i]) == Math.Sign(e[i])) signAgree++;
    }
    // Relative error is not the right yardstick: it is dominated by the leans nearest
    // zero and says nothing about whether a band could move. The decision scale is the
    // dead-band the caller cuts on, so measure headroom against that.
    const float deadBand = 5e-4f;   // IndicatorNearTermLean.FlatBand
    Console.WriteLine();
    Console.WriteLine($"{"engine end-to-end",-26} {worstAbs,12:E3} {worstRel,12:E3}  n={n}");
    Console.WriteLine($"{"  -> sign agreement",-26} {signAgree,12}/{n,-11}");
    Console.WriteLine($"{"  -> headroom vs dead-band",-26} {deadBand / Math.Max(worstAbs, 1e-30f),12:N0}x");
    ok &= signAgree == n && worstAbs < deadBand / 100f;
}

Console.WriteLine(ok ? "\nFULL PARITY OK" : "\n*** PARITY FAILED ***");
return ok ? 0 : 2;
