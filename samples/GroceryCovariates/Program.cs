// The grocery-store toy from Google's TimesFM covariates notebook: forecast next week's
// daily sales for ice cream and sunscreen, given this week's sales plus temperature and
// promotion schedules that are known for both weeks.
//
// NOT a reproduction of that notebook's numbers. It uses forecast_with_covariates, the
// XReg path, which forecasts the base series and then regresses covariates on the
// residuals. TimesFM 3.0 instead attends over the covariates as variates, so the
// mechanism differs and the outputs should not be expected to match.
//
//   dotnet run --project samples/GroceryCovariates

using TorchSharp;

using Tsfm.Forecasting.TimesFm;

using static TorchSharp.torch;

const int observed = 7, horizon = 7;                 // one week seen, one week ahead

// Known for both weeks — the point of the example.
float[] temperature = [31.0f, 24.3f, 19.4f, 26.2f, 24.6f, 30.0f, 31.1f,
                       32.4f, 30.9f, 26.0f, 25.0f, 27.8f, 29.5f, 31.2f];
float[] weekday     = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6];

var products = new (string Name, float[] Sales, float[] Promotion)[]
{
    ("ice cream", [30, 30, 4, 5, 7, 8, 10], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
    ("sunscreen", [5, 7, 12, 13, 5, 6, 10],  [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
};

Device device;
try { device = new Device(DeviceType.MPS); using (ones(1).to(device)) { } }
catch { device = new Device(DeviceType.CPU); }

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
var forecaster = TimesFmForecaster.Load(
    Path.Combine(root, "checkpoints", "timesfm-3.0-pytorch"), device);

// The model patches 32 steps at a time and reads the known future by rolling whole
// patches forward, so the context must fill one patch and the horizon must sit in the
// next. Seven days is far below that granularity: the observed week is left-padded and
// masked to fill a patch, and the forecast week occupies the one after it.
var patch = forecaster.Config.InputPatchLen;
var context = patch;                       // one whole patch of context
var total = context + patch;               // one further patch holds the horizon
var pad = context - observed;              // left padding, masked as absent

var quantiles = forecaster.Config.Quantiles;
Console.WriteLine($"{device.type}: {observed} days observed, {horizon} forecast; "
                  + $"padded to {context} context + {patch} horizon patch (patch length {patch})\n");

foreach (var (name, sales, promotion) in products)
{
    // Variate 0 is the target; the rest are known through the horizon. Order is
    // arbitrary — the model has no variate identity and infers relations from values.
    const int variates = 4;
    var series = new float[variates * total];
    for (var t = 0; t < observed + horizon; t++)
    {
        var i = pad + t;
        series[0 * total + i] = t < observed ? sales[t] : 0f;
        series[1 * total + i] = temperature[t];
        series[2 * total + i] = promotion[t];
        series[3 * total + i] = weekday[t];
    }

    var q = forecaster.ForecastJoint(series, variates, context, horizon,
        knownFuture: [false, true, true, true], targetVariate: 0);

    Console.WriteLine($"{name}  (observed: {string.Join(", ", sales)})");
    Console.WriteLine($"  day  promo  temp    p10     p50     p90");
    for (var h = 0; h < horizon; h++)
        Console.WriteLine($"  {h + 1,3}  {(promotion[observed + h] > 0 ? "yes" : "  -"),5}"
                          + $"  {temperature[observed + h],4:F1}"
                          + $"  {q[h, 0],6:F1}  {q[h, quantiles.Length / 2],6:F1}  {q[h, quantiles.Length - 1],6:F1}");
    Console.WriteLine();
}
