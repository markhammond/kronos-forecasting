// Quantile directional-signal bakeoff: Kronos against TimesFM over one week of 5-minute
// bars, 384 bars of context, horizons 1 and 4.
//
// Both models are reduced to the same statistic — an implied P(return > 0) — so the
// comparison does not reward one model's output format over the other's:
//
//   Kronos   samples rollouts, so P(up) is the empirical share of positive draws.
//   TimesFM  emits trained quantiles, so P(up) is where zero falls among them.
//
// Both predict the PATH MEAN over the horizon, because that is what Kronos's `lean`
// already is (mean of horizon closes against the anchor). Scoring a terminal return
// against a path-mean forecast would penalise Kronos for a definition it never used.

using Tsfm.Forecasting.Kronos;
using Tsfm.Forecasting.Kronos.Weights;
using Tsfm.Forecasting.TimesFm;

using static TorchSharp.torch;

const int Context = 384;
const int Rollouts = 30;
int[] horizons = [1, 4];

var csv = args.Length > 0 ? args[0] : Path.Combine(Path.GetTempPath(), "bakeoff-5m.csv");
var maxAnchors = args.Length > 1 ? int.Parse(args[1]) : int.MaxValue;
if (!File.Exists(csv)) { Console.Error.WriteLine($"missing {csv}"); return 1; }

var rows = File.ReadAllLines(csv).Skip(1)
    .Select(l => l.Split(','))
    .Select(f => (
        Time: DateTime.Parse(f[0], null, System.Globalization.DateTimeStyles.RoundtripKind),
        O: float.Parse(f[1]), H: float.Parse(f[2]), L: float.Parse(f[3]),
        C: float.Parse(f[4]), V: float.Parse(f[5]), A: float.Parse(f[6]),
        Anchor: f[7] == "1"))
    .ToArray();

var n = rows.Length;
var ohlcva = new float[n * 6];
var timesMs = new long[n];
for (var i = 0; i < n; i++)
{
    ohlcva[i * 6 + 0] = rows[i].O; ohlcva[i * 6 + 1] = rows[i].H; ohlcva[i * 6 + 2] = rows[i].L;
    ohlcva[i * 6 + 3] = rows[i].C; ohlcva[i * 6 + 4] = rows[i].V; ohlcva[i * 6 + 5] = rows[i].A;
    timesMs[i] = new DateTimeOffset(rows[i].Time, TimeSpan.Zero).ToUnixTimeMilliseconds();
}

Device device;
try { device = new Device("mps"); using (ones(1).to(device)) { } }
catch { device = new Device("cpu"); }

Console.WriteLine($"device      : {device.type}");
Console.WriteLine($"bars        : {n}, anchors {rows.Count(r => r.Anchor)}");
Console.WriteLine($"span        : {rows[0].Time:u} .. {rows[^1].Time:u}");
Console.WriteLine($"context 384, rollouts {Rollouts}, horizons {string.Join(",", horizons)}\n");

using var kronos = KronosForecaster.Load(KronosSmall.Instance, device);
var timesfm = TimesFmForecaster.Load(
    Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../..")),
        "checkpoints", "timesfm-3.0-pytorch"), device);
var quantiles = timesfm.Config.Quantiles;

foreach (var h in horizons)
{
    Console.WriteLine($"── horizon {h} ──────────────────────────────────────────");

    // Realised path-mean return over the h bars after each anchor.
    var anchorIdx = new List<int>();
    for (var i = 0; i < n; i++) if (rows[i].Anchor && i + h < n) anchorIdx.Add(i);
    if (anchorIdx.Count > maxAnchors) anchorIdx = anchorIdx.Take(maxAnchors).ToList();

    var realised = new double[anchorIdx.Count];
    for (var a = 0; a < anchorIdx.Count; a++)
    {
        var i = anchorIdx[a];
        double sum = 0;
        for (var j = 1; j <= h; j++) sum += rows[i + j].C;
        realised[a] = sum / h / rows[i].C - 1.0;
    }

    // ── Kronos: one batched call covers every window in the slice ──────────
    var first = anchorIdx[0] - Context + 1;
    var lastBar = anchorIdx[^1];
    var sliceLen = lastBar - first + 1;
    var rowsOut = KronosForecaster.OutputCount(sliceLen, Context);
    var lean = new float[rowsOut]; var up = new int[rowsOut]; var disp = new float[rowsOut];

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var ok = kronos.Infer(ohlcva.AsSpan(first * 6, sliceLen * 6), timesMs.AsSpan(first, sliceLen),
        lean, up, disp, Context, h, Rollouts, greedy: false, temperature: 1f, topP: 1f);
    sw.Stop();
    Console.WriteLine($"  kronos   : {rowsOut} windows in {sw.Elapsed.TotalSeconds:F0}s  ok={ok}");

    var kProb = new double[anchorIdx.Count];
    for (var a = 0; a < anchorIdx.Count; a++) kProb[a] = up[a] / (double)Rollouts;

    // ── TimesFM: one forward pass per anchor yields every horizon at once ──
    var tProb = new double[anchorIdx.Count];
    sw.Restart();
    for (var a = 0; a < anchorIdx.Count; a++)
    {
        var i = anchorIdx[a];
        var q = timesfm.Forecast(ohlcva.AsSpan((i - Context + 1) * 6, Context * 6), h);
        // Path mean per quantile, then the implied P(up).
        var mean = new double[quantiles.Length];
        for (var qi = 0; qi < quantiles.Length; qi++)
        {
            double s = 0;
            for (var step = 0; step < h; step++) s += q[step, qi];
            mean[qi] = s / h;
        }
        tProb[a] = ProbAboveZero(mean, quantiles);
        if (a == 0)
            Console.WriteLine("  sample fan : "
                + string.Join(" ", mean.Select((m, qi) => $"q{quantiles[qi]*100:F0}={m * 1e4:+0.0;-0.0}bp"))
                + $"  -> P(up) {tProb[a]:P1}");
    }
    sw.Stop();
    Console.WriteLine($"  timesfm  : {anchorIdx.Count} anchors in {sw.Elapsed.TotalSeconds:F0}s "
                      + $"({sw.Elapsed.TotalMilliseconds / anchorIdx.Count:F0} ms/anchor)\n");

    var prior = new double[anchorIdx.Count];
    for (var a = 0; a < anchorIdx.Count; a++)
    {
        var i = anchorIdx[a];
        prior[a] = i > 0 ? rows[i].C / (double)rows[i - 1].C - 1.0 : 0.0;
    }

    Report("kronos  P(up)", kProb, realised, prior, h);
    Report("timesfm P(up)", tProb, realised, prior, h);
    Console.WriteLine($"  {"leak sentinel",-14} IC vs ALREADY-SEEN bar: "
                      + $"kronos {Spearman(kProb, prior):+0.000}  timesfm {Spearman(tProb, prior):+0.000}"
                      + "   (>> forward IC would mean the window is misaligned)");
    Console.WriteLine();
}
return 0;

// Where zero sits in the predictive distribution, by linear interpolation between the
// bracketing quantile levels. Returns P(return > 0).
static double ProbAboveZero(double[] sortedQuantileValues, double[] levels)
{
    if (sortedQuantileValues[0] > 0) return 1.0;                       // whole interval positive
    if (sortedQuantileValues[^1] < 0) return 0.0;                      // whole interval negative
    for (var i = 0; i < sortedQuantileValues.Length - 1; i++)
    {
        var (lo, hi) = (sortedQuantileValues[i], sortedQuantileValues[i + 1]);
        if (lo <= 0 && 0 <= hi)
        {
            var t = hi > lo ? (0 - lo) / (hi - lo) : 0.5;
            return 1.0 - (levels[i] + t * (levels[i + 1] - levels[i]));
        }
    }
    return 0.5;
}

static void Report(string label, double[] signal, double[] realised, double[] prior, int horizon)
{
    var rho = Spearman(signal, realised);

    // Threshold at the signal's OWN median, not 0.5. A model whose median forecast is
    // persistently positive is otherwise scored as always-long, and its hit rate simply
    // reports the base rate of up bars rather than any directional skill.
    var mid = Median(signal);
    int taken = 0, correct = 0;
    for (var i = 0; i < signal.Length; i++)
    {
        var side = Math.Sign(signal[i] - mid);
        if (side == 0 || realised[i] == 0) continue;
        taken++;
        if (side == Math.Sign(realised[i])) correct++;
    }
    var acc = taken > 0 ? correct / (double)taken : double.NaN;

    // Most of the raw IC is an echo of the last observed bar. What matters is whether
    // anything survives once that echo is removed from BOTH sides.
    var partial = Spearman(Residual(Rank(signal), Rank(prior)), Residual(Rank(realised), Rank(prior)));

    var t = HacT(Rank(realised), Rank(signal), Math.Max(1, horizon));
    var tp = HacT(Residual(Rank(realised), Rank(prior)), Residual(Rank(signal), Rank(prior)),
                  Math.Max(1, horizon));
    Console.WriteLine($"  {label,-14} IC {rho,+7:0.000} |t| {t,5:F2}   "
                      + $"net of last bar {partial,+7:0.000} |t| {tp,5:F2}   hit {acc,6:P1} (n={taken})");
}

static double Median(double[] v)
{
    var c = (double[])v.Clone();
    Array.Sort(c);
    return c.Length % 2 == 1 ? c[c.Length / 2] : 0.5 * (c[c.Length / 2 - 1] + c[c.Length / 2]);
}

static double[] Residual(double[] y, double[] x)
{
    double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0;
    for (var i = 0; i < x.Length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    var b = sxx > 0 ? sxy / sxx : 0;
    return y.Select((t, i) => t - my - b * (x[i] - mx)).ToArray();
}

static double Spearman(double[] a, double[] b)
{
    double[] ra = Rank(a), rb = Rank(b);
    double ma = ra.Average(), mb = rb.Average(), num = 0, da = 0, db = 0;
    for (var i = 0; i < ra.Length; i++)
    {
        num += (ra[i] - ma) * (rb[i] - mb);
        da += (ra[i] - ma) * (ra[i] - ma);
        db += (rb[i] - mb) * (rb[i] - mb);
    }
    return da > 0 && db > 0 ? num / Math.Sqrt(da * db) : double.NaN;
}

static double[] Rank(double[] v)
{
    var idx = Enumerable.Range(0, v.Length).OrderBy(i => v[i]).ToArray();
    var rk = new double[v.Length];
    for (var i = 0; i < idx.Length;)
    {
        var j = i;
        while (j + 1 < idx.Length && v[idx[j + 1]] == v[idx[i]]) j++;
        var avg = (i + j) / 2.0;
        for (var k = i; k <= j; k++) rk[idx[k]] = avg;
        i = j + 1;
    }
    return rk;
}

static double HacT(double[] y, double[] x, int lag)
{
    var n = x.Length;
    double mx = x.Average(), my = y.Average(), sxy = 0, sxx = 0;
    for (var i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    if (sxx <= 0) return double.NaN;
    var beta = sxy / sxx;
    var u = new double[n];
    for (var i = 0; i < n; i++) u[i] = (x[i] - mx) * (y[i] - my - beta * (x[i] - mx));
    double sum = 0;
    for (var i = 0; i < n; i++) sum += u[i] * u[i];
    for (var L = 1; L <= lag; L++)
    {
        double g = 0;
        for (var i = L; i < n; i++) g += u[i] * u[i - L];
        sum += 2.0 * (1.0 - L / (double)(lag + 1)) * g;
    }
    return Math.Abs(beta) / (Math.Sqrt(Math.Max(1e-30, sum)) / sxx);
}
