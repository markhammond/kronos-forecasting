using static TorchSharp.torch;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.TimesFm;

/// <summary>
/// The normalisation and patching that surround the transformer.
///
/// <para>Statistics are cumulative and causal: patch i is normalised by the mean and
/// standard deviation of every unmasked point in patches 0..i. That is what makes the
/// model's own scale window-relative, and is the analogue of Kronos's window-local
/// normalisation.</para>
/// </summary>
public static class TimesFmPreprocess
{
    private const double Tolerance = 1e-6;

    /// <summary>Cumulative per-patch statistics, each shaped (b, v, n).</summary>
    /// <remarks>
    /// Combines each patch into the running total with Chan's parallel formula rather than
    /// re-scanning the prefix, so cost is linear in patches. Naively accumulating sums of
    /// squares loses precision once the running mean is far from zero.
    /// </remarks>
    public static (Tensor N, Tensor Mu, Tensor Sigma) RunningStats(Tensor values, Tensor masks)
    {
        using var scope = NewDisposeScope();
        var (b, v, n) = (values.shape[0], values.shape[1], values.shape[2]);
        var dev = values.device;

        var curN = zeros([b, v], ScalarType.Float32, dev);
        var curMu = zeros([b, v], ScalarType.Float32, dev);
        var curSigma = zeros([b, v], ScalarType.Float32, dev);

        var outN = new Tensor[n];
        var outMu = new Tensor[n];
        var outSigma = new Tensor[n];

        for (var i = 0; i < n; i++)
        {
            var x = values.select(2, i);                       // (b, v, p)
            var m = masks.select(2, i);
            var legit = ~m;
            var legitF = legit.to(ScalarType.Float32);

            var incN = legitF.sum([-1L]);
            var incSum = where(legit, x, zeros_like(x)).sum([-1L]);
            var incMu = where(incN == 0, zeros_like(incSum), incSum / incN);
            var diffSq = where(legit, (x - incMu.unsqueeze(-1)).pow(2), zeros_like(x));
            var incVar = where(incN == 0, zeros_like(incSum), diffSq.sum([-1L]) / incN);
            var incSigma = incVar.sqrt();

            var newN = curN + incN;
            var newMu = where(newN == 0, zeros_like(curMu),
                (curN * curMu + incMu * incN) / newN);
            var newSigma = where(newN == 0, zeros_like(curSigma),
                (curN * curSigma * curSigma
                 + incN * incSigma * incSigma
                 + curN * (curMu - newMu).pow(2)
                 + incN * (incMu - newMu).pow(2)) / newN).sqrt();

            (curN, curMu, curSigma) = (newN, newMu, newSigma);
            outN[i] = newN; outMu[i] = newMu; outSigma[i] = newSigma;
        }

        return (stack(outN, 2).MoveToOuterDisposeScope(),
                stack(outMu, 2).MoveToOuterDisposeScope(),
                stack(outSigma, 2).MoveToOuterDisposeScope());
    }

    /// <summary>Normalise (or invert) by the running statistics. A near-zero sigma divides
    /// by one rather than exploding.</summary>
    public static Tensor Revin(Tensor x, Tensor mu, Tensor sigma, bool reverse = false)
    {
        using var scope = NewDisposeScope();
        var m = mu; var s = sigma;
        if (mu.Dimensions == x.Dimensions - 1) { m = mu.unsqueeze(-1); s = sigma.unsqueeze(-1); }
        else if (mu.Dimensions == x.Dimensions - 2)
        {
            m = mu.unsqueeze(-1).unsqueeze(-1); s = sigma.unsqueeze(-1).unsqueeze(-1);
        }
        var safe = where(s < Tolerance, ones_like(s), s);
        return (reverse ? x * s + m : (x - m) / safe).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// For each patch, the <paramref name="rolls"/> patches that follow it, flattened —
    /// the "future covariate" slots. Also returns the mask marking positions that wrapped
    /// past the end of the sequence and therefore carry no real future.
    /// </summary>
    public static (Tensor Values, Tensor WrapMask) OutputPatchViaRoll(Tensor x, int rolls)
    {
        using var scope = NewDisposeScope();
        var (b, v, n, p) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);

        var shifted = new Tensor[rolls];
        var cur = x;
        for (var i = 0; i < rolls; i++) { cur = cur.roll(-1, 2); shifted[i] = cur; }
        var result = stack(shifted, 3).reshape(b, v, n, rolls * p);

        var patchIdx = arange(n, ScalarType.Int64, x.device).unsqueeze(-1);
        var pointIdx = arange(rolls * p, ScalarType.Int64, x.device).unsqueeze(0);
        var sourcePatch = patchIdx + 1 + pointIdx / p;
        var wrap = (sourcePatch >= n).unsqueeze(0).unsqueeze(0);

        return (result.MoveToOuterDisposeScope(), wrap.MoveToOuterDisposeScope());
    }
}
