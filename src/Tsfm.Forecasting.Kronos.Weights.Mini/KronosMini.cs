using System.Reflection;

using Tsfm.Forecasting;
using Tsfm.Forecasting.Kronos;

namespace Tsfm.Forecasting.Kronos.Weights;

/// <summary>
/// Kronos-mini (4.1M parameters) with Kronos-Tokenizer-2k. 2048-token context, though a consumer capped below that gains nothing from the extra reach.
///
/// <para>Pinned revisions: model <c>f4e68697d9d5aed55cef5c96aabc3376bcad9f81</c>, tokenizer <c>26966d0035065a0cae0ebad7af8ece35bc1fb51c</c>. They are
/// embedded rather than fetched, so the weights a build was compiled against cannot
/// drift underneath it — which matters wherever a published result is content-addressed
/// over the code but not over the model that produced it.</para>
/// </summary>
public sealed class KronosMini : EmbeddedCheckpoint
{
    public static readonly KronosMini Instance = new();

    public override string Name => "NeoQuasar--Kronos-mini@f4e68697d9d5aed55cef5c96aabc3376bcad9f81";

    /// <inheritdoc/>
    public override int MaxContext => 2048;

    protected override Assembly Host => typeof(KronosMini).Assembly;
    protected override string ModelResource => "kronos.model";
    protected override string ModelConfigResource => "kronos.model.config";
    protected override string TokenizerResource => "kronos.tokenizer";
    protected override string TokenizerConfigResource => "kronos.tokenizer.config";
}
