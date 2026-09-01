namespace Tsfm.Forecasting.Kronos;

/// <summary>Reduces one channel's rollout samples at one horizon step to a single value.
/// Samples are ordered by rollout index.</summary>
public delegate float RolloutAggregator(ReadOnlySpan<float> samples);

/// <summary>
/// Ready-made reducers for <see cref="KronosForecaster.InferPath"/>.
///
/// <para>A span-taking delegate rather than a transducer protocol: there is one reduction
/// over a short, already-contiguous sample set, so there is no pipeline to fuse and nothing
/// for composition machinery to save. Callers compose by writing a delegate that calls
/// others.</para>
/// </summary>
public static class RolloutAggregators
{
    /// <summary>Middle sample. Used for the directional lean, where the mean drifts toward
    /// the greedy projection and outliers drag it. Not the path default: the reference
    /// averages, and matching it keeps outputs comparable.</summary>
    public static float Median(ReadOnlySpan<float> samples) => Quantile(samples, 0.5);

    /// <summary>Arithmetic mean. The path default, matching the reference implementation's
    /// per-element average across samples.</summary>
    public static float Mean(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var v in samples) sum += v;
        return (float)(sum / samples.Length);
    }

    /// <summary>Linear-interpolated quantile, <paramref name="p"/> in [0, 1].</summary>
    public static float Quantile(ReadOnlySpan<float> samples, double p)
    {
        Span<float> sorted = samples.Length <= 64 ? stackalloc float[samples.Length] : new float[samples.Length];
        samples.CopyTo(sorted);
        sorted.Sort();
        var pos = p * (sorted.Length - 1);
        var lo = (int)Math.Floor(pos);
        var hi = Math.Min(lo + 1, sorted.Length - 1);
        return (float)(sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]));
    }

    /// <summary>A quantile reducer bound to <paramref name="p"/>.</summary>
    public static RolloutAggregator AtQuantile(double p)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(p);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(p, 1.0);
        return s => Quantile(s, p);
    }

    /// <summary>Takes rollout <paramref name="index"/> verbatim — one sampled path, not a
    /// blend. Note this does NOT give better-formed candles — it gives markedly worse ones,
    /// breaking OHLC ordering in roughly 30% to 70% of candles depending on the checkpoint,
    /// because the model decodes each channel independently and averaging across rollouts
    /// cancels the noise that causes the disorder.</summary>
    public static RolloutAggregator Rollout(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        return s => s[index];
    }
}
