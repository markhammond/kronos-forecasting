using static TorchSharp.torch;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.TimesFm;

/// <summary>
/// Quantile forecasts of forward return from OHLCVA bars, with the six channels supplied
/// as variates.
///
/// <para>Every variate is marked a target. That is not cosmetic: the preprocessing masks
/// future covariate slots only for target variates, so a channel left non-target is handed
/// its own future values. Marking close alone would leak tomorrow's high, low and volume
/// into today's forecast and flatter the model enormously.</para>
/// </summary>
public sealed class TimesFmForecaster(TimesFmModel model, TimesFmConfig config, Device device)
{
    private const int CloseChannel = 3;    // open, high, low, close, volume, amount
    private const int Channels = 6;

    public TimesFmConfig Config => config;

    public static TimesFmForecaster Load(string checkpointDir, Device device)
    {
        var cfg = TimesFmConfig.Load(Path.Combine(checkpointDir, "config.json"));
        return new TimesFmForecaster(TimesFmModel.Load(checkpointDir, device), cfg, device);
    }

    /// <summary>
    /// Forward-return quantiles for each step up to <paramref name="horizon"/>.
    /// </summary>
    /// <param name="ohlcva">Row-major [L x 6]; L must be a multiple of the patch length.</param>
    /// <returns>[horizon, numQuantiles] of cumulative return relative to the anchor close.</returns>
    public double[,] Forecast(ReadOnlySpan<float> ohlcva, int horizon)
    {
        var p = config.InputPatchLen;
        var bars = ohlcva.Length / Channels;
        if (bars % p != 0)
            throw new ArgumentException($"bar count {bars} must be a multiple of the patch length {p}");
        if (horizon < 1 || horizon > config.OutputPatchLen)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"horizon must lie in 1..{config.OutputPatchLen}");

        var n = bars / p;
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        // (bars, 6) -> (1, 6, n, p): variate-major, then patched along time.
        var flat = zeros(bars * Channels, ScalarType.Float32);
        var dst = flat.data<float>();
        for (var i = 0; i < bars * Channels; i++) dst[i] = ohlcva[i];
        var values = flat.view(bars, Channels).t().reshape(1, Channels, n, p).contiguous().to(device);

        var masks = zeros([1, Channels, n, p], ScalarType.Bool, device);
        var isTarget = ones([1, Channels, n], ScalarType.Bool, device);

        var logits = model.forward(values, masks, isTarget)
            .view(1, Channels, n, config.OutputPatchLen, config.NumQuantiles);

        // The final patch carries the forecast; its running statistics undo the
        // normalisation the model produced its output in.
        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(
            nan_to_num(values, 0.0).clamp(-config.ValueClip, config.ValueClip), masks);
        var m = mu[0, CloseChannel, n - 1].item<float>();
        var s = sigma[0, CloseChannel, n - 1].item<float>();

        var head = logits[0, CloseChannel, n - 1].to(ScalarType.Float32).cpu();
        var acc = head.data<float>();

        var anchorClose = ohlcva[(bars - 1) * Channels + CloseChannel];
        var result = new double[horizon, config.NumQuantiles];
        for (var h = 0; h < horizon; h++)
        for (var q = 0; q < config.NumQuantiles; q++)
        {
            var price = acc[h * config.NumQuantiles + q] * s + m;   // reverse revin
            result[h, q] = price / anchorClose - 1.0;
        }
        return result;
    }
}
