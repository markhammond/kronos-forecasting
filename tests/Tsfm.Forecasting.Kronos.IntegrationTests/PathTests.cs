using Tsfm.Forecasting.Kronos.Weights;

using Xunit;
using Xunit.Abstractions;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Kronos.IntegrationTests;

/// <summary>
/// Exercises <see cref="KronosForecaster.InferPath"/> against real weights. Kronos-mini
/// is used throughout: the path contract does not vary by checkpoint, and Mini costs a
/// fraction of Small to fetch.
/// </summary>
public class PathTests(ITestOutputHelper output) : IDisposable
{
    private const int Bars = 420, Context = 384, Horizon = 8, Rollouts = 16;

    private readonly KronosForecaster _forecaster =
        KronosForecaster.Load(KronosMini.Instance, new Device("cpu"));

    public void Dispose() => _forecaster.Dispose();

    /// <summary>Kronos-mini attends over 2048 bars; asking for more must be refused rather
    /// than silently truncated, which is what upstream does.</summary>
    [Fact]
    public void ContextBeyondTheCheckpointLimitIsRefused()
    {
        Assert.Equal(2048, _forecaster.MaxContext);

        var over = _forecaster.MaxContext + 1;
        var (ohlcva, times) = Synthetic(over + 1);
        var rows = KronosForecaster.OutputCount(over + 1, over);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => _forecaster.Infer(
            ohlcva, times, new float[rows], new int[rows], default,
            over, Horizon, Rollouts, greedy: true, 1f, 1f));
        Assert.Contains(_forecaster.MaxContext.ToString(), ex.Message);
    }

    [Fact]
    public void PathLengthMatchesWhatInferPathWrites()
    {
        var rows = KronosForecaster.OutputCount(Bars, Context);
        Assert.Equal(rows * Horizon * 6, KronosForecaster.PathLength(Bars, Context, Horizon));
    }

    [Fact]
    public void WrongSizedPathThrowsWithTheExpectedCount()
    {
        var (ohlcva, times) = Synthetic();
        var rows = KronosForecaster.OutputCount(Bars, Context);
        var ex = Assert.Throws<ArgumentException>(() => _forecaster.InferPath(
            ohlcva, times, new float[rows], new int[rows], default, new float[7],
            Context, Horizon, Rollouts, greedy: true, 1f, 1f));
        Assert.Contains(KronosForecaster.PathLength(Bars, Context, Horizon).ToString(), ex.Message);
    }

    /// <summary>The close channel of the path must reproduce the directional lean, which
    /// is computed from an independent read of the same decode. Greedy removes sampling so
    /// the two are directly comparable.</summary>
    [Fact]
    public void PathCloseChannelReconcilesWithLean()
    {
        var (ohlcva, times) = Synthetic();
        var rows = KronosForecaster.OutputCount(Bars, Context);
        var lean = new float[rows];
        var path = new float[KronosForecaster.PathLength(Bars, Context, Horizon)];

        Assert.True(_forecaster.InferPath(ohlcva, times, lean, new int[rows], default, path,
            Context, Horizon, Rollouts, greedy: true, 1f, 1f));

        var anchorClose = ohlcva[(Context - 1) * 6 + 3];
        double sum = 0;
        for (var j = 0; j < Horizon; j++) sum += path[j * 6 + 3];
        var fromPath = sum / Horizon / anchorClose - 1.0;

        output.WriteLine($"lean {lean[0]:F9}  from path {fromPath:F9}");
        Assert.Equal(lean[0], fromPath, 6);
    }

    [Fact]
    public void EveryProjectedValueIsFinite()
    {
        var (ohlcva, times) = Synthetic();
        var rows = KronosForecaster.OutputCount(Bars, Context);
        var path = new float[KronosForecaster.PathLength(Bars, Context, Horizon)];

        Assert.True(_forecaster.InferPath(ohlcva, times, new float[rows], new int[rows],
            default, path, Context, Horizon, Rollouts, greedy: false, 1f, 1f));
        Assert.All(path, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void AggregatorChoiceChangesTheResult()
    {
        var (ohlcva, times) = Synthetic();
        var rows = KronosForecaster.OutputCount(Bars, Context);
        var mean = new float[KronosForecaster.PathLength(Bars, Context, Horizon)];
        var high = new float[mean.Length];

        _forecaster.InferPath(ohlcva, times, new float[rows], new int[rows], default, mean,
            Context, Horizon, Rollouts, false, 1f, 1f, RolloutAggregators.Mean);
        _forecaster.InferPath(ohlcva, times, new float[rows], new int[rows], default, high,
            Context, Horizon, Rollouts, false, 1f, 1f, RolloutAggregators.AtQuantile(0.9));

        // A high quantile must sit above the mean somewhere, or the reducer is being ignored.
        Assert.Contains(Enumerable.Range(0, mean.Length), i => high[i] > mean[i]);
    }

    /// <summary>Records how often independently decoded channels break OHLC ordering. This
    /// is the model's behaviour, not the port's, and the README documents it — the test
    /// exists so a change in that behaviour is noticed rather than assumed.</summary>
    [Fact]
    public void OhlcOrderingIsNotGuaranteed()
    {
        var (ohlcva, times) = Synthetic();
        var rows = KronosForecaster.OutputCount(Bars, Context);
        var path = new float[KronosForecaster.PathLength(Bars, Context, Horizon)];

        foreach (var (name, agg) in new (string, RolloutAggregator)[]
                 { ("mean", RolloutAggregators.Mean), ("single rollout", RolloutAggregators.Rollout(0)) })
        {
            _forecaster.InferPath(ohlcva, times, new float[rows], new int[rows], default, path,
                Context, Horizon, Rollouts, greedy: false, 1f, 1f, agg);

            int bad = 0, total = rows * Horizon;
            for (var i = 0; i < total; i++)
            {
                var b = i * 6;
                if (path[b + 1] < MathF.Max(path[b + 0], path[b + 3]) ||
                    path[b + 2] > MathF.Min(path[b + 0], path[b + 3])) bad++;
            }
            output.WriteLine($"{name,-14} {bad}/{total} candles break OHLC ordering ({100.0 * bad / total:F1}%)");
        }
    }

    private static (float[] Ohlcva, long[] TimeMs) Synthetic(int bars = Bars)
    {
        var ohlcva = new float[bars * 6];
        var times = new long[bars];
        var t0 = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();
        for (var i = 0; i < bars; i++)
        {
            var mid = 60000f + 400f * MathF.Sin(i / 19f) + 3f * i;
            float o = mid - 5f, c = mid + 5f;
            ohlcva[i * 6 + 0] = o;
            ohlcva[i * 6 + 1] = MathF.Max(o, c) + 12f;
            ohlcva[i * 6 + 2] = MathF.Min(o, c) - 12f;
            ohlcva[i * 6 + 3] = c;
            ohlcva[i * 6 + 4] = 1000f + 40f * (i % 7);
            ohlcva[i * 6 + 5] = ohlcva[i * 6 + 4] * mid;
            times[i] = t0 + i * 300_000L;
        }
        return (ohlcva, times);
    }
}
