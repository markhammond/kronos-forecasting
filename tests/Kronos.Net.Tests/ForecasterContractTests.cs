using System;
using Xunit;

namespace Kronos.Net.Tests;

/// <summary>
/// The parts of the forecaster that hold without weights loaded: the row arithmetic every
/// caller sizes its buffers from, the calendar features, and the seeding rule.
/// </summary>
public class ForecasterContractTests
{
    [Theory]
    [InlineData(385, 384, 2)]
    [InlineData(384, 384, 1)]
    [InlineData(420, 384, 37)]
    [InlineData(100, 384, -283)]   // negative: caller must treat <= 0 as "no output"
    public void OutputCountIsBarsMinusContextPlusOne(int bars, int context, int expected)
        => Assert.Equal(expected, KronosForecaster.OutputCount(bars, context));

    [Fact]
    public void WeekdayIsMondayZero()
    {
        // pandas .dt.weekday is Monday-zero; DayOfWeek is Sunday-zero. Taking the latter
        // directly shifts every weekday embedding by one with no other symptom, so this
        // pins the conversion rather than the calendar.
        var monday = new DateTimeOffset(2026, 3, 2, 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();
        var sunday = new DateTimeOffset(2026, 3, 8, 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();

        var stamps = new float[2 * 5];
        KronosForecaster.WriteStamp(stamps, 0, monday);
        KronosForecaster.WriteStamp(stamps, 1, sunday);

        Assert.Equal(0f, stamps[2]);   // Monday
        Assert.Equal(6f, stamps[7]);   // Sunday
    }

    [Fact]
    public void StampCarriesMinuteHourDayMonth()
    {
        var t = new DateTimeOffset(2026, 7, 14, 23, 55, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();
        var stamps = new float[5];
        KronosForecaster.WriteStamp(stamps, 0, t);

        Assert.Equal(55f, stamps[0]);  // minute
        Assert.Equal(23f, stamps[1]);  // hour
        Assert.Equal(14f, stamps[3]);  // day of month
        Assert.Equal(7f, stamps[4]);   // month
    }

    [Fact]
    public void SeedIsAPureFunctionOfTheBar()
    {
        // Sampling must not depend on which bars were grouped into a batch, so the seed
        // is derived from the bar alone and never carried between bars.
        const long bar = 1_772_000_000_000;
        Assert.Equal(KronosForecaster.SeedFor(bar), KronosForecaster.SeedFor(bar));
        Assert.NotEqual(KronosForecaster.SeedFor(bar), KronosForecaster.SeedFor(bar + 300_000));
        Assert.True(KronosForecaster.SeedFor(bar) >= 0, "seed feeds a generator that rejects negatives");
    }

    [Fact]
    public void AdjacentBarsGetWellSeparatedSeeds()
    {
        // Consecutive bars differ by one interval; a weak mixer would give them adjacent
        // streams and correlate their draws.
        var a = KronosForecaster.SeedFor(1_772_000_000_000);
        var b = KronosForecaster.SeedFor(1_772_000_300_000);
        Assert.True(Math.Abs(a - b) > 1L << 32, $"seeds too close: {a} vs {b}");
    }
}
