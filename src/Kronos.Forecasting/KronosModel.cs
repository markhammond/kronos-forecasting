using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Kronos.Forecasting;

/// <summary>
/// The autoregressive model over subtoken pairs. The two decode passes differ in cost:
/// <see cref="DecodeS1"/> traverses the full stack, <see cref="DecodeS2"/> reuses its
/// hidden states for one cross-attention and a projection.
/// </summary>
public sealed class KronosModel : nn.Module
{
    private readonly HierarchicalEmbedding embedding;
    private readonly TemporalEmbedding time_emb;
    private readonly ModuleList<KronosTransformerBlock> transformer;
    private readonly RmsNorm norm;
    private readonly DualHead head;
    private readonly DependencyAwareLayer dep_layer;

    public KronosModel(long s1Bits, long s2Bits, long dModel, long nHeads, long ffDim, long nLayers)
        : base(nameof(KronosModel))
    {
        embedding = new HierarchicalEmbedding(s1Bits, s2Bits, dModel);
        time_emb = new TemporalEmbedding(dModel);
        transformer = nn.ModuleList(Enumerable.Range(0, (int)nLayers)
            .Select(_ => new KronosTransformerBlock(dModel, nHeads, ffDim)).ToArray());
        norm = new RmsNorm(dModel);
        head = new DualHead(s1Bits, s2Bits, dModel);
        dep_layer = new DependencyAwareLayer(dModel, nHeads: 4);
        RegisterComponents();
    }

    /// <summary>Coarse-subtoken logits and the hidden states they came from; the fine
    /// head needs the states.</summary>
    public (Tensor logits, Tensor context) DecodeS1(Tensor s1Ids, Tensor s2Ids, Tensor stamp)
    {
        using var scope = NewDisposeScope();
        var x = embedding.forward(s1Ids, s2Ids) + time_emb.forward(stamp);
        foreach (var block in transformer) x = block.forward(x);
        x = norm.forward(x);
        return (head.forward(x).MoveToOuterDisposeScope(), x.MoveToOuterDisposeScope());
    }

    /// <summary>Fine-subtoken logits, conditioned on the sampled coarse token. The sibling
    /// query is the <b>raw, unscaled</b> s1 embedding, not the fused sequence embedding —
    /// a distinction the tensor names hide.</summary>
    public Tensor DecodeS2(Tensor context, Tensor s1Ids)
    {
        using var scope = NewDisposeScope();
        var sibling = embedding.EmbedS1Raw(s1Ids);
        var conditioned = dep_layer.forward(context, sibling);
        return head.CondForward(conditioned).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st)
    {
        embedding.Load(st, "embedding");
        time_emb.Load(st, "time_emb");
        norm.weight.set_(st.Get("norm.weight"));
        head.Load(st, "head");
        dep_layer.Load(st, "dep_layer");
        for (var i = 0; i < transformer.Count; i++) transformer[i].Load(st, $"transformer.{i}");
    }

    public static KronosModel FromCheckpoint(IKronosCheckpoint checkpoint, Device device)
    {
        var cfg = System.Text.Json.JsonDocument.Parse(checkpoint.ModelConfigJson).RootElement;
        long G(string k) => cfg.GetProperty(k).GetInt64();
        using var stream = checkpoint.OpenModel();
        var st = Safetensors.Load(stream, $"{checkpoint.Name}/model");
        var model = new KronosModel(G("s1_bits"), G("s2_bits"), G("d_model"), G("n_heads"), G("ff_dim"), G("n_layers"));
        using (no_grad()) model.Load(st);
        model.to(device);
        model.eval();
        return model;
    }
}
