using Xunit;
using Xunit.Abstractions;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.TimesFm.IntegrationTests;

/// <summary>
/// Covers <see cref="TimesFmForecaster.ForecastJoint"/>.
///
/// <para>Gated on the checkpoint being present and therefore vacuous in CI: the TimesFM
/// weights are ~1.2 GB and non-commercially licensed, so they are never fetched there.
/// Fetch them with scripts/fetch-timesfm-checkpoint.sh to run these locally.</para>
/// </summary>
public class JointForecastTests(ITestOutputHelper output)
{
    private const int Patch = 32, Variates = 3, Horizon = 8;

    private static string CheckpointDir => Path.Combine(
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../..")),
        "checkpoints", "timesfm-3.0-pytorch");

    private static bool Available => File.Exists(Path.Combine(CheckpointDir, "model.safetensors"));

    private TimesFmForecaster? Load()
    {
        if (Available) return TimesFmForecaster.Load(CheckpointDir, new Device("cpu"));
        output.WriteLine("checkpoint absent — vacuous");
        return null;
    }

    /// <summary>The whole point of the feature: values supplied for a known-future variate
    /// must reach the model. If they did not, the two forecasts would coincide.</summary>
    [Fact]
    public void KnownFutureValuesChangeTheForecast()
    {
        var f = Load(); if (f is null) return;

        var (rising, falling) = (Series(driverFuture: +40f), Series(driverFuture: -40f));
        var a = f.ForecastJoint(rising, Variates, Patch, Horizon, [false, true, true]);
        var b = f.ForecastJoint(falling, Variates, Patch, Horizon, [false, true, true]);

        var mid = a.GetLength(1) / 2;
        var moved = Enumerable.Range(0, Horizon).Count(h => Math.Abs(a[h, mid] - b[h, mid]) > 1e-4);
        output.WriteLine($"median differs on {moved}/{Horizon} steps; "
                         + $"step 0: {a[0, mid]:F3} vs {b[0, mid]:F3}");
        Assert.True(moved > 0, "a known-future variate that changes should move the forecast");
    }

    /// <summary>Marking a variate known but leaving its future unchanged must not move the
    /// forecast — otherwise the previous test could pass on noise alone.</summary>
    [Fact]
    public void IdenticalInputGivesIdenticalForecast()
    {
        var f = Load(); if (f is null) return;

        var s = Series(driverFuture: +40f);
        var a = f.ForecastJoint(s, Variates, Patch, Horizon, [false, true, true]);
        var b = f.ForecastJoint(s, Variates, Patch, Horizon, [false, true, true]);

        for (var h = 0; h < Horizon; h++)
        for (var q = 0; q < a.GetLength(1); q++)
            Assert.Equal(a[h, q], b[h, q], 6);
    }

    [Fact]
    public void QuantilesAreOrderedAndFinite()
    {
        var f = Load(); if (f is null) return;

        var q = f.ForecastJoint(Series(0f), Variates, Patch, Horizon, [false, true, true]);
        for (var h = 0; h < Horizon; h++)
        {
            for (var i = 0; i < q.GetLength(1); i++) Assert.True(double.IsFinite(q[h, i]));
            for (var i = 1; i < q.GetLength(1); i++)
                Assert.True(q[h, i] >= q[h, i - 1] - 1e-3,
                    $"quantiles must not invert at step {h}: {q[h, i - 1]:F4} then {q[h, i]:F4}");
        }
    }

    /// <summary>The limit is the checkpoint's own declaration, not a constant of the
    /// architecture — so assert it comes from the config rather than a literal.</summary>
    [Fact]
    public void MaxVariatesComesFromTheCheckpoint()
    {
        var f = Load(); if (f is null) return;
        Assert.Equal(32, f.MaxVariates);          // what TimesFM 3.0 declares
        Assert.Equal(f.Config.MaxVariates, f.MaxVariates);

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(
            new float[(f.MaxVariates + 1) * Patch * 2], f.MaxVariates + 1, Patch, Horizon));
        Assert.Contains(f.MaxVariates.ToString(), ex.Message);
    }

    [Fact]
    public void TargetCannotBeMarkedKnownIntoTheFuture()
    {
        var f = Load(); if (f is null) return;
        var ex = Assert.Throws<ArgumentException>(() => f.ForecastJoint(
            Series(0f), Variates, Patch, Horizon, [true, true, true]));
        Assert.Contains("target", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ContextMustFillWholePatchesAndLeaveRoom()
    {
        var f = Load(); if (f is null) return;
        // Not a multiple of the patch length.
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(
            Series(0f), Variates, Patch - 1, Horizon, [false, true, true]));
        // Leaves no patch to roll the known future from.
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(
            Series(0f), Variates, Patch * 2, Horizon, [false, true, true]));
    }

    /// <summary>Variate 0 is the target; 1 is a driver whose future is supplied; 2 is a
    /// constant. The driver's future is what the tests vary.</summary>
    private static float[] Series(float driverFuture)
    {
        const int total = Patch * 2;
        var s = new float[Variates * total];
        for (var t = 0; t < total; t++)
        {
            var observed = t < Patch;
            s[0 * total + t] = observed ? 100f + 5f * MathF.Sin(t / 4f) : 0f;
            s[1 * total + t] = observed ? 20f + 2f * MathF.Sin(t / 4f) : 20f + driverFuture;
            s[2 * total + t] = 1f;
        }
        return s;
    }
}
