using System.Reflection;

namespace Kronos.Net.Weights;

/// <summary>
/// Kronos-small (24.7M parameters) with Kronos-Tokenizer-base. 512-token context.
///
/// <para>Pinned revisions: model <c>901c26c1332695a2a8f243eb2f37243a37bea320</c>, tokenizer <c>0e0117387f39004a9016484a186a908917e22426</c>. They are
/// embedded rather than fetched, so the weights a build was compiled against cannot
/// drift underneath it — which matters wherever a published result is content-addressed
/// over the code but not over the model that produced it.</para>
/// </summary>
public sealed class KronosSmall : EmbeddedCheckpoint
{
    public static readonly KronosSmall Instance = new();

    public override string Name => "NeoQuasar--Kronos-small@901c26c1332695a2a8f243eb2f37243a37bea320";

    protected override Assembly Host => typeof(KronosSmall).Assembly;
    protected override string ModelResource => "kronos.model";
    protected override string ModelConfigResource => "kronos.model.config";
    protected override string TokenizerResource => "kronos.tokenizer";
    protected override string TokenizerConfigResource => "kronos.tokenizer.config";
}
