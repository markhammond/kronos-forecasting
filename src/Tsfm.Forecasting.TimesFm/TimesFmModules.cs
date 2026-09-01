using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.TimesFm;

/// <summary>Root-mean-square norm over the last dimension, weight only.</summary>
public sealed class RmsNorm(long dims, double eps = 1e-6) : Module<Tensor, Tensor>(nameof(RmsNorm))
{
    public Parameter weight = new(ones(dims));

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var norm = x * rsqrt(x.pow(2).mean([-1L], keepdim: true) + eps);
        return (norm * weight).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Learned per-dimension query scale, Pax style:
/// <c>x * 1.442695041 / sqrt(d) * softplus(s)</c>.
///
/// <para>This REPLACES the usual 1/sqrt(d) attention scaling. Attention must therefore be
/// invoked with a scale of 1, or every logit is scaled twice.</para>
/// </summary>
public sealed class PerDimScale(long dims) : Module<Tensor, Tensor>(nameof(PerDimScale))
{
    private const double ReciprocalOfSoftplus0 = 1.442695041;

    public Parameter per_dim_scale = new(zeros(dims));

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var s = ReciprocalOfSoftplus0 / Math.Sqrt(dims);
        return (x * s * functional.softplus(per_dim_scale)).MoveToOuterDisposeScope();
    }
}

/// <summary>Rotary embedding over the patch axis. Cached tables are detached: handing them
/// to the caller's scope frees them after the first use.</summary>
public sealed class RotaryEmbedding : Module<Tensor, Tensor, Tensor>
{
    private readonly Tensor _invFreq;

    public RotaryEmbedding(long headDim, double theta = 10000.0) : base(nameof(RotaryEmbedding))
    {
        var idx = arange(0, headDim, 2, ScalarType.Float32);
        _invFreq = (1.0 / pow(theta, idx / headDim)).DetachFromDisposeScope();
    }

    /// <param name="x">(b, n, h, hd)</param>
    /// <param name="position">(b, n)</param>
    public override Tensor forward(Tensor x, Tensor position)
    {
        using var scope = NewDisposeScope();
        var freqs = position.to(ScalarType.Float32).unsqueeze(-1) * _invFreq;  // (b, n, hd/2)
        var cos = freqs.cos().unsqueeze(2);                                    // (b, n, 1, hd/2)
        var sin = freqs.sin().unsqueeze(2);

        var half = x.shape[^1] / 2;
        var x1 = x.narrow(-1, 0, half);
        var x2 = x.narrow(-1, half, half);
        var r1 = x1 * cos - x2 * sin;
        var r2 = x2 * cos + x1 * sin;
        return cat([r1, r2], -1).MoveToOuterDisposeScope();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _invFreq.Dispose();
        base.Dispose(disposing);
    }
}

/// <summary>
/// Multi-head attention with RoPE, QK norm and a learned per-dimension query scale.
///
/// <para>Order matters and is not the usual one: RoPE is applied BEFORE the QK norm, and
/// the query scale is learned rather than 1/sqrt(d).</para>
/// </summary>
public sealed class MultiHeadAttention : Module<Tensor, Tensor, Tensor?, Tensor>
{
    private readonly long _heads, _headDim;
    private readonly bool _causal;

    public Linear query_proj, key_proj, value_proj, out_proj;
    public RmsNorm query_ln, key_ln;
    public PerDimScale per_dim_scale;
    private readonly RotaryEmbedding? _rope;

    public MultiHeadAttention(long modelDims, long heads, bool useRope, bool causal)
        : base(nameof(MultiHeadAttention))
    {
        (_heads, _headDim, _causal) = (heads, modelDims / heads, causal);
        query_proj = Linear(modelDims, modelDims, hasBias: false);
        key_proj = Linear(modelDims, modelDims, hasBias: false);
        value_proj = Linear(modelDims, modelDims, hasBias: false);
        out_proj = Linear(modelDims, modelDims, hasBias: false);
        query_ln = new RmsNorm(_headDim);
        key_ln = new RmsNorm(_headDim);
        per_dim_scale = new PerDimScale(_headDim);
        if (useRope) _rope = new RotaryEmbedding(_headDim);
        RegisterComponents();
    }

    /// <param name="x">(b, n, d)</param>
    /// <param name="position">(b, n) positions for RoPE</param>
    /// <param name="patchMask">(b, n) true where the patch is masked, or null</param>
    public override Tensor forward(Tensor x, Tensor position, Tensor? patchMask)
    {
        using var scope = NewDisposeScope();
        var (b, n) = (x.shape[0], x.shape[1]);

        var q = query_proj.forward(x).view(b, n, _heads, _headDim);
        var k = key_proj.forward(x).view(b, n, _heads, _headDim);
        var v = value_proj.forward(x).view(b, n, _heads, _headDim);

        if (_rope is not null)
        {
            q = _rope.forward(q, position);
            k = _rope.forward(k, position);
        }
        q = query_ln.forward(q);
        k = key_ln.forward(k);
        q = per_dim_scale.forward(q);

        // (b, n, h, hd) -> (b, h, n, hd)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2);

        // Causality and the patch mask COMBINE; they are not alternatives. Passing the
        // patch mask through SDPA's attn_mask while relying on is_casual for ordering
        // silently drops causality, because is_casual is ignored once a mask is given.
        Tensor? mask = null;
        if (_causal || patchMask is not null)
        {
            var m = zeros([b, 1, n, n], ScalarType.Float32, x.device);
            if (_causal)
            {
                var future = ones([n, n], ScalarType.Bool, x.device).triu(diagonal: 1);
                m = m.masked_fill(future, float.NegativeInfinity);
            }
            if (patchMask is not null)
                m = m.masked_fill(patchMask.unsqueeze(1).unsqueeze(1), float.NegativeInfinity);
            mask = m;
        }

        // The reference leaves rescale_logits false, so its logits are QK^T * sqrt(d) —
        // NOT the conventional QK^T / sqrt(d). TorchSharp's SDPA always divides by
        // sqrt(d) and offers no override, so scaling the query by d yields sqrt(d)
        // overall. PerDimScale has already replaced the usual query scaling.
        q = q * _headDim;
        var o = functional.scaled_dot_product_attention(q, k, v, attn_mask: mask);

        o = o.transpose(1, 2).contiguous().view(b, n, _heads * _headDim);
        return out_proj.forward(o).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// One mixing layer: sequence attention over patches, then variate attention across
/// channels, then a feed-forward block.
///
/// <para>Every sub-block is sandwich normed — <c>post_ln(f(pre_ln(x))) + x</c>. A
/// conventional pre-norm residual reads as equivalent and is not.</para>
/// </summary>
public sealed class MixingTransformerLayer : Module<Tensor, Tensor, Tensor>
{
    private readonly bool _useVariateAttention;

    public RmsNorm pre_seq_attn_ln, post_seq_attn_ln, pre_var_attn_ln, post_var_attn_ln,
                   pre_ff_ln, post_ff_ln;
    public MultiHeadAttention seq_attn, var_attn;
    public Linear ff0, ff1;

    public MixingTransformerLayer(long modelDims, long heads, bool useVariateAttention)
        : base(nameof(MixingTransformerLayer))
    {
        _useVariateAttention = useVariateAttention;
        pre_seq_attn_ln = new RmsNorm(modelDims); post_seq_attn_ln = new RmsNorm(modelDims);
        pre_var_attn_ln = new RmsNorm(modelDims); post_var_attn_ln = new RmsNorm(modelDims);
        pre_ff_ln = new RmsNorm(modelDims); post_ff_ln = new RmsNorm(modelDims);
        // Sequence attention is causal over patches; variate attention is not ordered,
        // and config sets use_rope_var false.
        seq_attn = new MultiHeadAttention(modelDims, heads, useRope: true, causal: true);
        var_attn = new MultiHeadAttention(modelDims, heads, useRope: false, causal: false);
        ff0 = Linear(modelDims, modelDims, hasBias: false);
        ff1 = Linear(modelDims, modelDims, hasBias: false);
        RegisterComponents();
    }

    /// <param name="x">(b, v, n, d)</param>
    /// <param name="patchMask">(b, v, n) true where masked</param>
    public override Tensor forward(Tensor x, Tensor patchMask)
    {
        using var scope = NewDisposeScope();
        var (b, v, n, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);

        // Sequence attention: fold variates into the batch so each channel attends
        // over its own patch history.
        var sIn = pre_seq_attn_ln.forward(x).reshape(b * v, n, d);
        var sMask = patchMask.reshape(b * v, n);
        var pos = arange(n, ScalarType.Int32, x.device).unsqueeze(0).expand(b * v, n);
        var sOut = seq_attn.forward(sIn, pos, sMask).view(b, v, n, d);
        var h1 = post_seq_attn_ln.forward(sOut) + x;

        var h2 = h1;
        if (_useVariateAttention)
        {
            // Variate attention: fold patches into the batch so channels attend to each
            // other at the same instant.
            var vIn = pre_var_attn_ln.forward(h1).permute(0, 2, 1, 3).reshape(b * n, v, d);
            var vMask = patchMask.permute(0, 2, 1).reshape(b * n, v);
            var vPos = zeros([b * n, v], ScalarType.Int32, x.device);
            var vOut = var_attn.forward(vIn, vPos, vMask).view(b, n, v, d).permute(0, 2, 1, 3);
            h2 = post_var_attn_ln.forward(vOut) + h1;
        }

        var ff = ff1.forward(functional.relu(ff0.forward(pre_ff_ln.forward(h2))));
        return (post_ff_ln.forward(ff) + h2).MoveToOuterDisposeScope();
    }
}

/// <summary>Two-layer block with a projected residual:
/// <c>output_layer(relu(hidden_layer(x))) + residual_layer(x)</c>.</summary>
public sealed class ResidualBlock : Module<Tensor, Tensor>
{
    public Linear hidden_layer, output_layer, residual_layer;

    public ResidualBlock(long inputDims, long hiddenDims, long outputDims)
        : base(nameof(ResidualBlock))
    {
        hidden_layer = Linear(inputDims, hiddenDims, hasBias: false);
        output_layer = Linear(hiddenDims, outputDims, hasBias: false);
        residual_layer = Linear(inputDims, outputDims, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var hidden = functional.relu(hidden_layer.forward(x));
        return (output_layer.forward(hidden) + residual_layer.forward(x)).MoveToOuterDisposeScope();
    }
}

/// <summary>Holds the layer stack under the name the checkpoint uses
/// (<c>transformer_stack.layers.N</c>), so weights bind without a rename table.</summary>
public sealed class StackedMixingTransformer : Module<Tensor, Tensor, Tensor>
{
    public ModuleList<MixingTransformerLayer> layers;

    public StackedMixingTransformer(long modelDims, long heads, bool useVariateAttention, int count)
        : base(nameof(StackedMixingTransformer))
    {
        layers = new ModuleList<MixingTransformerLayer>(
            [.. Enumerable.Range(0, count).Select(_ =>
                new MixingTransformerLayer(modelDims, heads, useVariateAttention))]);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor patchMask)
    {
        var h = x;
        foreach (var layer in layers) h = layer.forward(h, patchMask);
        return h;
    }
}
