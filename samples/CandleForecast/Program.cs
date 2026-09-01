// Projects a candle path from OHLC history, which is the question the library's span-based
// API answers least obviously. Mirrors prediction_example.py in the upstream repository.
//
//   dotnet run --project samples/CandleForecast

using TorchSharp;

using Tsfm.Forecasting.Kronos;
using Tsfm.Forecasting.Kronos.Weights;

using static TorchSharp.torch;

// Lookback matches the upstream example; the checkpoint states its own ceiling, so no
// context length is hard-coded here.
const int lookback = 400, horizon = 12, rollouts = 30;

// A caller's own candle type, to show the conversion rather than assume a layout.
var history = BuildHistory(lookback);

Device device;
try { device = new Device(DeviceType.MPS); using (ones(1).to(device)) { } }
catch { device = new Device(DeviceType.CPU); }

var checkpoint = KronosMini.Instance;
var context = Math.Min(history.Count, checkpoint.MaxContext);

using var forecaster = KronosForecaster.Load(checkpoint, device);
Console.WriteLine($"{forecaster.CheckpointName} on {device.type}, "
                  + $"{context} of {checkpoint.MaxContext} context bars\n");

var ohlcva = new float[history.Count * KronosForecaster.Channels];
var timeMs = new long[history.Count];
for (var i = 0; i < history.Count; i++)
{
    var c = history[i];
    WriteBar(ohlcva, timeMs, i,
        open: c.Open, high: c.High, low: c.Low, close: c.Close, volume: c.Volume,
        amount: c.Volume * (c.Open + c.High + c.Low + c.Close) / 4f,
        closeTimeMs: c.CloseTime.ToUnixTimeMilliseconds());
}

var rows = KronosForecaster.OutputCount(history.Count, context);   // 1
var lean = new float[rows];
var upCount = new int[rows];
var path = new float[KronosForecaster.PathLength(history.Count, context, horizon)];

forecaster.InferPath(ohlcva, timeMs, lean, upCount, dispersion: default, path,
    context, horizon, rollouts, greedy: false, temperature: 1f, topP: 1f);

Console.WriteLine($"lean {lean[0] * 1e4:+0.0;-0.0}bp over {horizon} bars, "
                  + $"{upCount[0]}/{rollouts} rollouts positive\n");

// Channels are decoded independently, so ordering is not guaranteed. Clamp for anything
// that will be plotted or treated as a real candle.
Console.WriteLine("step         open       high        low      close   out of order");
var last = history[^1].CloseTime;
for (var j = 0; j < horizon; j++)
{
    var b = j * KronosForecaster.Channels;
    float o = path[b + 0], h = path[b + 1], l = path[b + 2], c = path[b + 3];
    var outOfOrder = h < MathF.Max(o, c) || l > MathF.Min(o, c);
    Console.WriteLine($"{last.AddMinutes(5 * (j + 1)):HH:mm}  {o,10:F2} {h,10:F2} {l,10:F2} {c,10:F2}"
                      + (outOfOrder ? "   yes" : ""));
}

static void WriteBar(Span<float> ohlcva, Span<long> timeMs, int index,
    float open, float high, float low, float close, float volume, float amount,
    long closeTimeMs)
{
    var o = index * KronosForecaster.Channels;
    ohlcva[o + 0] = open;
    ohlcva[o + 1] = high;
    ohlcva[o + 2] = low;
    ohlcva[o + 3] = close;
    ohlcva[o + 4] = volume;
    ohlcva[o + 5] = amount;
    timeMs[index] = closeTimeMs;
}

static List<Candle> BuildHistory(int n)
{
    var t0 = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
    var bars = new List<Candle>(n);
    for (var i = 0; i < n; i++)
    {
        var mid = 60000f + 400f * MathF.Sin(i / 19f) + 3f * i;
        float o = mid - 5f, c = mid + 5f;
        bars.Add(new Candle(t0.AddMinutes(5 * i), o, MathF.Max(o, c) + 12f,
            MathF.Min(o, c) - 12f, c, 1000f + 40f * (i % 7)));
    }
    return bars;
}

/// <summary>A caller's own candle type. The library takes spans rather than a candle
/// abstraction, so this exists to show the conversion, not because it is required.</summary>
internal readonly record struct Candle(
    DateTimeOffset CloseTime, float Open, float High, float Low, float Close, float Volume);
