using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Kronos.Forecasting;

/// <summary>
/// The BSQ tokenizer: continuous OHLCVA bars to two 10-bit subtoken streams, and back.
///
/// <para>Everything after <c>quant_embed</c> is sign-preserving — L2 normalisation
/// divides by a positive norm, the BSQ scale is positive — so the emitted index reduces
/// to <c>bit_j = quant_embed_out_j &gt; 0</c>. Parity fails only within float epsilon of
/// zero, and does not accumulate.</para>
/// </summary>
public sealed class KronosTokenizerEncoder : nn.Module<Tensor, (Tensor s1, Tensor s2)>
{
    private readonly Linear embed, quant_embed, post_quant_embed, head;
    private readonly ModuleList<KronosTransformerBlock> encoder, decoder;
    private readonly long _s1Bits, _s2Bits;
    private readonly double _qScale;

    public KronosTokenizerEncoder(
        long dIn, long dModel, long nHeads, long ffDim, long nEncLayers, long s1Bits, long s2Bits)
        : base(nameof(KronosTokenizerEncoder))
    {
        _s1Bits = s1Bits; _s2Bits = s2Bits;
        embed = nn.Linear(dIn, dModel);
        // The reference builds n_enc_layers - 1 blocks; config disagrees with the tensor
        // names, so derive the count.
        encoder = nn.ModuleList(Enumerable.Range(0, (int)nEncLayers - 1)
            .Select(_ => new KronosTransformerBlock(dModel, nHeads, ffDim)).ToArray());
        quant_embed = nn.Linear(dModel, s1Bits + s2Bits);
        post_quant_embed = nn.Linear(s1Bits + s2Bits, dModel);
        decoder = nn.ModuleList(Enumerable.Range(0, (int)nEncLayers - 1)
            .Select(_ => new KronosTransformerBlock(dModel, nHeads, ffDim)).ToArray());
        head = nn.Linear(dModel, dIn);
        _qScale = 1.0 / Math.Sqrt(s1Bits + s2Bits);
        RegisterComponents();
    }

    /// <summary>Pre-quantisation projection, whose sign determines every emitted bit.
    /// Exposed so parity can be measured before the discrete output hides the margin.</summary>
    public Tensor Project(Tensor x)
    {
        using var scope = NewDisposeScope();
        var z = embed.forward(x);
        foreach (var block in encoder) z = block.forward(z);
        return quant_embed.forward(z).MoveToOuterDisposeScope();
    }

    /// <summary>Intermediates under the reference's export names. Diagnostic: a
    /// whole-stack comparison cannot separate a faithful port from a lucky sign.</summary>
    public IEnumerable<(string name, Tensor value)> Probe(Tensor x)
    {
        var z = embed.forward(x);
        yield return ("after_embed", z);
        for (var i = 0; i < encoder.Count; i++)
        {
            var (norm1, attn, outp) = encoder[i].Probe(z);
            yield return ($"blk{i}_norm1", norm1);
            yield return ($"blk{i}_attn", attn);
            yield return ($"blk{i}_out", outp);
            z = outp;
        }
        yield return ("z", quant_embed.forward(z));
    }

    public override (Tensor s1, Tensor s2) forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var z = Project(x);

        // bits_to_indices over the sign pattern; positional weights are 2^j ascending.
        var bits = z.gt(0).to(ScalarType.Int64);
        var pre = bits.narrow(-1, 0, _s1Bits);
        var post = bits.narrow(-1, _s1Bits, _s2Bits);
        var w1 = pow(2, arange(_s1Bits, ScalarType.Int64, z.device));
        var w2 = pow(2, arange(_s2Bits, ScalarType.Int64, z.device));

        var s1 = (pre * w1).sum([-1L]);
        var s2 = (post * w2).sum([-1L]);
        return (s1.MoveToOuterDisposeScope(), s2.MoveToOuterDisposeScope());
    }

    /// <summary>Indices back to continuous bars. Unpack to ascending bit positions, map
    /// to bipolar ±1, apply the <c>1/sqrt(codebook)</c> scale the quantiser used inbound;
    /// without it the decoder sees the wrong magnitudes.</summary>
    public Tensor Decode(Tensor s1, Tensor s2)
    {
        using var scope = NewDisposeScope();
        var bits = cat([Unpack(s1, _s1Bits), Unpack(s2, _s2Bits)], dim: -1);
        var q = (bits.to(ScalarType.Float32) * 2 - 1) * _qScale;

        var z = post_quant_embed.forward(q);
        foreach (var block in decoder) z = block.forward(z);
        return head.forward(z).MoveToOuterDisposeScope();
    }

    private static Tensor Unpack(Tensor indices, long bits)
    {
        var mask = pow(2, arange(bits, ScalarType.Int64, indices.device));
        return indices.unsqueeze(-1).bitwise_and(mask).ne(0);
    }

    public void Load(Safetensors st)
    {
        embed.weight!.set_(st.Get("embed.weight"));
        embed.bias!.set_(st.Get("embed.bias"));
        quant_embed.weight!.set_(st.Get("quant_embed.weight"));
        quant_embed.bias!.set_(st.Get("quant_embed.bias"));
        post_quant_embed.weight!.set_(st.Get("post_quant_embed.weight"));
        post_quant_embed.bias!.set_(st.Get("post_quant_embed.bias"));
        head.weight!.set_(st.Get("head.weight"));
        head.bias!.set_(st.Get("head.bias"));
        for (var i = 0; i < encoder.Count; i++) encoder[i].Load(st, $"encoder.{i}");
        for (var i = 0; i < decoder.Count; i++) decoder[i].Load(st, $"decoder.{i}");
    }

    public static KronosTokenizerEncoder FromCheckpoint(IKronosCheckpoint checkpoint, Device device)
    {
        var cfg = System.Text.Json.JsonDocument.Parse(checkpoint.TokenizerConfigJson).RootElement;
        long G(string k) => cfg.GetProperty(k).GetInt64();
        using var stream = checkpoint.OpenTokenizer();
        var st = Safetensors.Load(stream, $"{checkpoint.Name}/tokenizer");

        // Count blocks from the tensor names; config's n_enc_layers includes a stage the
        // reference does not materialise.
        var blocks = st.Tensors.Keys
            .Where(k => k.StartsWith("encoder.", StringComparison.Ordinal))
            .Select(k => int.Parse(k.Split('.')[1])).DefaultIfEmpty(-1).Max() + 1;

        var enc = new KronosTokenizerEncoder(
            G("d_in"), G("d_model"), G("n_heads"), G("ff_dim"), blocks + 1, G("s1_bits"), G("s2_bits"));
        using (no_grad()) enc.Load(st);
        enc.to(device);
        enc.eval();
        return enc;
    }
}
