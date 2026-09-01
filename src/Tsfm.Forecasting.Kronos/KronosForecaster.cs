using System.Buffers;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.Kronos;

/// <summary>
/// End-to-end inference: raw bars in, per-window summaries out.
///
/// <para>Tensors do not cross this boundary — callers pass spans and receive spans.</para>
///
/// <para><b>No key-value cache</b>, matching the reference: a cache changes the arithmetic
/// and forfeits checkable parity. Costs one full pass per decode step per rollout, so it
/// is affordable only at short horizons.</para>
/// </summary>
public sealed class KronosForecaster : IDisposable
{
    private readonly KronosTokenizerEncoder _tokenizer;
    private readonly KronosModel _model;
    private readonly Device _device;

    /// <summary>Rows this slice produces, <c>L − K + 1</c>. Size buffers from this; a
    /// wrong count misaligns every row.</summary>
    public static int OutputCount(int barCount, int contextBars) => barCount - contextBars + 1;

    private KronosForecaster(KronosTokenizerEncoder tokenizer, KronosModel model, Device device)
        => (_tokenizer, _model, _device) = (tokenizer, model, device);

    public string CheckpointName { get; private init; } = string.Empty;

    public static KronosForecaster Load(ICheckpoint checkpoint, Device device)
        => new(KronosTokenizerEncoder.FromCheckpoint(checkpoint, device),
               KronosModel.FromCheckpoint(checkpoint, device), device)
           { CheckpointName = checkpoint.Name };

    /// <summary>
    /// One inference over a contiguous slice, re-windowed internally so the payload is
    /// O(L) rather than O(L·K).
    /// </summary>
    /// <returns><c>L − K + 1</c> rows, the last anchored on the final bar. Forward
    /// timestamps are extrapolated from the bar interval, so no trailing bars are consumed.</returns>
    public bool Infer(ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> barTimeMs,
        Span<float> lean, Span<int> upCount, Span<float> dispersion,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, int batch = 8)
    {
        var k = contextBars;
        var h = horizon;
        var length = barTimeMs.Length;
        if (ohlcva.Length != length * 6)
            throw new ArgumentException("ohlcva must be L x 6", nameof(ohlcva));

        var count = OutputCount(length, k);
        if (count <= 0) return false;
        if (lean.Length != count || upCount.Length != count)
            throw new ArgumentException(
                $"lean and upCount must each hold exactly {count} rows (= L - K + 1); " +
                $"got {lean.Length}/{upCount.Length}. Use {nameof(OutputCount)}.");

        // Dispersion is diagnostic. An empty span skips the standard deviation entirely.
        if (!dispersion.IsEmpty && dispersion.Length != count)
            throw new ArgumentException(
                $"dispersion must hold exactly {count} rows or be empty; got {dispersion.Length}.");

        lean.Fill(float.NaN);
        if (!dispersion.IsEmpty) dispersion.Fill(float.NaN);
        upCount.Clear();

        var stamps = ArrayPool<float>.Shared.Rent(length * 5);
        // Size scratch for a full batch and reuse it across the slice; per-batch
        // allocation dominates a materialisation-sized run.
        var maxRows = batch * rollouts;
        var x = ArrayPool<float>.Shared.Rent(maxRows * k * 6);
        var stamp = ArrayPool<float>.Shared.Rent(maxRows * (k + h) * 5);
        var mean = ArrayPool<float>.Shared.Rent(batch * 6);
        var scale = ArrayPool<float>.Shared.Rent(batch * 6);
        var anchor = ArrayPool<float>.Shared.Rent(batch);
        var uniforms = ArrayPool<float>.Shared.Rent(batch * rollouts * h * 2);
        var pick = ArrayPool<float>.Shared.Rent(maxRows);

        try
        {
            BuildStamps(barTimeMs, stamps);
            var stepMs = length > 1 ? barTimeMs[1] - barTimeMs[0] : 0L;

            using var _ = no_grad();
            for (var start = 0; start < count; start += batch)
            {
                var take = Math.Min(batch, count - start);
                EvaluateBatch(ohlcva, barTimeMs, stamps, stepMs, start, take,
                    k, h, rollouts, greedy, temperature, topP,
                    lean, upCount, dispersion, x, stamp, mean, scale, anchor, uniforms, pick);
            }
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(stamps);
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(stamp);
            ArrayPool<float>.Shared.Return(mean);
            ArrayPool<float>.Shared.Return(scale);
            ArrayPool<float>.Shared.Return(anchor);
            ArrayPool<float>.Shared.Return(uniforms);
            ArrayPool<float>.Shared.Return(pick);
        }
    }

    private void EvaluateBatch(
        ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> barTimeMs, float[] stamps, long stepMs,
        int start, int take, int k, int h, int rollouts, bool greedy, float temperature, float topP,
        Span<float> lean, Span<int> upCount, Span<float> dispersion,
        float[] x, float[] stamp, float[] mean, float[] scale, float[] anchor,
        float[] uniforms, float[] pick)
    {
        using var scope = NewDisposeScope();
        var bn = take * rollouts;

        // Normalisation is window-local and causal: statistics come from the same K bars
        // the model reads.
        for (var w = 0; w < take; w++)
        {
            var origin = start + w;
            NormaliseWindow(ohlcva, origin, k, mean.AsSpan(w * 6, 6), scale.AsSpan(w * 6, 6));
            anchor[w] = ohlcva[(origin + k - 1) * 6 + 3];               // close of the anchor bar

            for (var r = 0; r < rollouts; r++)
            {
                var row = w * rollouts + r;
                for (var t = 0; t < k; t++)
                for (var c = 0; c < 6; c++)
                {
                    var v = (ohlcva[(origin + t) * 6 + c] - mean[w * 6 + c]) / (scale[w * 6 + c] + 1e-5f);
                    x[(row * k + t) * 6 + c] = Math.Clamp(v, -5f, 5f);
                }
                Array.Copy(stamps, origin * 5, stamp, row * (k + h) * 5, k * 5);
                for (var j = 0; j < h; j++)                              // forward stamps: extrapolated
                    WriteStamp(stamp, row * (k + h) + k + j, barTimeMs[origin + k - 1] + stepMs * (j + 1));
            }
        }

        // Slice explicitly. ArrayPool returns arrays longer than requested and the final
        // batch fills fewer rows; reshaping the backing array reinterprets stale values.
        var xt = tensor(new Memory<float>(x, 0, bn * k * 6), [bn, (long)k, 6L]).to(_device);
        var st = tensor(new Memory<float>(stamp, 0, bn * (k + h) * 5), [bn, k + h, 5L]).to(_device);

        var (s1, s2) = _tokenizer.forward(xt);
        var pre = cat([s1, zeros(bn, h, ScalarType.Int64, _device)], dim: 1);
        var post = cat([s2, zeros(bn, h, ScalarType.Int64, _device)], dim: 1);

        // Draw before the loop: sampling must be a pure function of the bar, not of how
        // windows were batched.
        if (!greedy) DrawUniforms(barTimeMs, start, take, k, rollouts, h, uniforms);

        for (var step = 0; step < h; step++)
        {
            var len = k + step;
            var (logits, context) = _model.DecodeS1(
                pre.narrow(1, 0, len), post.narrow(1, 0, len), st.narrow(1, 0, len).contiguous());
            var a = Pick(logits.select(1, len - 1), greedy, temperature, topP, uniforms, step * 2, bn, rollouts, h, pick);
            var l2 = _model.DecodeS2(context, a);
            var b = Pick(l2.select(1, len - 1), greedy, temperature, topP, uniforms, step * 2 + 1, bn, rollouts, h, pick);
            pre.select(1, len).copy_(a.squeeze(-1));
            post.select(1, len).copy_(b.squeeze(-1));
        }

        // Decode the whole sequence, keep the generated tail, and read the close channel.
        var decoded = _tokenizer.Decode(pre, post).narrow(1, k, h);
        var closes = decoded.select(-1, 3).to(ScalarType.Float32).cpu().data<float>().ToArray();

        for (var w = 0; w < take; w++)
        {
            var mu = mean[w * 6 + 3];
            var sd = scale[w * 6 + 3];
            var c0 = anchor[w];
            var per = new double[rollouts];
            for (var r = 0; r < rollouts; r++)
            {
                double sum = 0;
                for (var j = 0; j < h; j++)
                    sum += closes[(w * rollouts + r) * h + j] * (sd + 1e-5f) + mu;
                per[r] = sum / h / c0 - 1.0;
            }
            Array.Sort(per);
            var up = 0;
            foreach (var v in per) if (v > 0) up++;
            // Median, not mean: the mean converges to the greedy projection and outlier
            // rollouts drag it.
            lean[start + w] = (float)(rollouts % 2 == 1
                ? per[rollouts / 2]
                : 0.5 * (per[rollouts / 2 - 1] + per[rollouts / 2]));
            upCount[start + w] = up;
            if (!dispersion.IsEmpty) dispersion[start + w] = rollouts > 1 ? (float)StdDev(per) : 0f;
        }
    }

    private static double StdDev(double[] v)
    {
        double m = 0;
        foreach (var t in v) m += t;
        m /= v.Length;
        double acc = 0;
        foreach (var t in v) acc += (t - m) * (t - m);
        return Math.Sqrt(acc / (v.Length - 1));
    }

    /// <summary>Per-channel mean and population standard deviation, ddof = 0 as
    /// <c>np.std</c>.</summary>
    private static void NormaliseWindow(
        ReadOnlySpan<float> ohlcva, int origin, int k, Span<float> mean, Span<float> sd)
    {
        for (var c = 0; c < 6; c++)
        {
            double sum = 0;
            for (var t = 0; t < k; t++) sum += ohlcva[(origin + t) * 6 + c];
            var m = sum / k;
            double acc = 0;
            for (var t = 0; t < k; t++)
            {
                var d = ohlcva[(origin + t) * 6 + c] - m;
                acc += d * d;
            }
            mean[c] = (float)m;
            sd[c] = (float)Math.Sqrt(acc / k);
        }
    }

    /// <summary>One uniform per (window, rollout, decode step, head), seeded from the
    /// window's anchor bar. SplitMix64, so the stream cannot shift with a framework
    /// version.</summary>
    private static void DrawUniforms(
        ReadOnlySpan<long> barTimeMs, int start, int take, int k, int rollouts, int h, float[] u)
    {
        var perWindow = rollouts * h * 2;
        for (var w = 0; w < take; w++)
        {
            var state = (ulong)SeedFor(barTimeMs[start + w + k - 1]);
            for (var i = 0; i < perWindow; i++)
            {
                state += 0x9E3779B97F4A7C15UL;
                var z = state;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                z ^= z >> 31;
                u[w * perWindow + i] = (float)((z >> 11) * (1.0 / 9007199254740992.0));
            }
        }
    }

    /// <summary>Select a token. Greedy takes the argmax; sampling uses inverse CDF against
    /// a pre-drawn uniform, making the draw independent of batch grouping. Distributionally
    /// identical to the reference; the RNG stream deliberately is not.</summary>
    private static Tensor Pick(Tensor logits, bool greedy, float temperature, float topP,
        float[] uniforms, int slot, int bn, int rollouts, int h, float[] pick)
    {
        using var scope = NewDisposeScope();
        if (greedy) return logits.argmax(-1, keepdim: true).MoveToOuterDisposeScope();

        var scaled = logits / Math.Max(temperature, 1e-6f);
        var probs = F.softmax(scaled, -1);

        if (topP < 1f)
        {
            var (sorted, indices) = probs.sort(dim: -1, descending: true);
            var cum = sorted.cumsum(-1);
            var keep = (cum - sorted).le(topP);
            sorted = sorted * keep.to(sorted.dtype);
            probs = zeros_like(probs).scatter_(-1, indices, sorted);
        }

        probs = probs / probs.sum(-1, keepdim: true);
        var cdf = probs.cumsum(-1);

        // Window-major layout: the row index divides out to its window.
        var perWindow = rollouts * h * 2;
        for (var i = 0; i < bn; i++) pick[i] = uniforms[(i / rollouts) * perWindow + (i % rollouts) * h * 2 + slot];
        var u = from_array(pick.AsSpan(0, bn).ToArray(), ScalarType.Float32)
            .reshape(bn, 1).to(logits.device);

        var idx = searchsorted(cdf, u);
        return idx.clamp(0, cdf.shape[^1] - 1).MoveToOuterDisposeScope();
    }

    /// <summary>Calendar features: minute, hour, weekday, day-of-month, month.
    /// <b>Weekday is Monday-zero</b>, as pandas. <see cref="DayOfWeek"/> is Sunday-zero;
    /// using it directly shifts every weekday embedding by one, silently.</summary>
    private static void BuildStamps(ReadOnlySpan<long> barTimeMs, float[] stamps)
    {
        for (var i = 0; i < barTimeMs.Length; i++) WriteStamp(stamps, i, barTimeMs[i]);
    }

    public static void WriteStamp(float[] dst, int row, long unixMs)
    {
        var t = DateTimeOffset.FromUnixTimeMilliseconds(unixMs).UtcDateTime;
        var o = row * 5;
        dst[o + 0] = t.Minute;
        dst[o + 1] = t.Hour;
        dst[o + 2] = ((int)t.DayOfWeek + 6) % 7;   // Sunday-zero -> Monday-zero
        dst[o + 3] = t.Day;
        dst[o + 4] = t.Month;
    }

    /// <summary>
    ///     FNV-1a over the bar's timestamp. Never carried between bars, so a sliding
    ///     pass and a one-shot pass agree.</summary>
    /// <remarks>
    ///     Deterministically seeded from the timestamp
    /// </remarks>
    public static long SeedFor(long closeTimeMs)
    {
        var hash = 0xcbf29ce484222325UL;
        for (var i = 0; i < 8; i++)
        {
            hash ^= (byte)(closeTimeMs >> (i * 8));
            hash *= 0x100000001b3UL;
        }
        return (long)(hash & 0x7FFFFFFFFFFFFFFFUL);
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _model.Dispose();
    }
}
