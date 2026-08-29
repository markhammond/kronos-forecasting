using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace Kronos.Forecasting;

/// <summary>Root-mean-square norm, weight-only. Compute in float32 and cast back, as the
/// reference does; the width matters for parity when activations are narrower.</summary>
public sealed class RmsNorm : nn.Module<Tensor, Tensor>
{
    private readonly double _eps;
    public readonly Parameter weight;

    public RmsNorm(long dim, double eps = 1e-5) : base(nameof(RmsNorm))
    {
        _eps = eps;
        weight = nn.Parameter(ones(dim));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var f = x.to(ScalarType.Float32);
        var normed = f * rsqrt(f.pow(2).mean([-1L], keepdim: true) + _eps);
        return (normed.type_as(x) * weight).MoveToOuterDisposeScope();
    }
}

/// <summary>SwiGLU feed-forward: <c>w2(silu(w1 x) * w3 x)</c>, all three unbiased.</summary>
public sealed class SwiGluFeedForward : nn.Module<Tensor, Tensor>
{
    private readonly Linear w1, w2, w3;

    public SwiGluFeedForward(long dModel, long ffDim) : base(nameof(SwiGluFeedForward))
    {
        w1 = nn.Linear(dModel, ffDim, hasBias: false);
        w3 = nn.Linear(dModel, ffDim, hasBias: false);
        w2 = nn.Linear(ffDim, dModel, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return w2.forward(F.silu(w1.forward(x)) * w3.forward(x)).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        w1.weight!.set_(st.Get($"{p}.w1.weight"));
        w2.weight!.set_(st.Get($"{p}.w2.weight"));
        w3.weight!.set_(st.Get($"{p}.w3.weight"));
    }
}

/// <summary>Rotary position embedding. Tables depend only on sequence length, so cache
/// per length — and rebuild on the tensor's own device: a module moved after first use
/// otherwise reads a stale cross-device buffer.</summary>
public sealed class RotaryEmbedding
{
    private readonly long _dim;
    private Tensor? _cos, _sin;
    private long _cachedLen = -1;
    private Device? _cachedDevice;

    public RotaryEmbedding(long headDim) => _dim = headDim;

    private (Tensor cos, Tensor sin) Cache(long seqLen, Device device)
    {
        if (_cachedLen == seqLen && _cachedDevice is not null && _cachedDevice.type == device.type)
            return (_cos!, _sin!);

        using var scope = NewDisposeScope();
        var invFreq = 1.0 / pow(10000.0, arange(0, _dim, 2, ScalarType.Float32, device) / _dim);
        var t = arange(seqLen, ScalarType.Float32, device);
        var freqs = outer(t, invFreq);                       // [S, dim/2]
        var emb = cat([freqs, freqs], dim: -1);              // [S, dim]
        _cos?.Dispose(); _sin?.Dispose();
        // Detach, not move-to-outer: the caller's scope ends with its forward pass, so
        // a cache handed to it is disposed on the way out and the next call reads freed
        // memory. The cache must outlive every scope and is owned by this object.
        _cos = emb.cos().unsqueeze(0).unsqueeze(0).DetachFromDisposeScope();   // [1,1,S,dim]
        _sin = emb.sin().unsqueeze(0).unsqueeze(0).DetachFromDisposeScope();
        _cachedLen = seqLen; _cachedDevice = device;
        return (_cos, _sin);
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var half = x.shape[^1] / 2;
        var x1 = x.narrow(-1, 0, half);
        var x2 = x.narrow(-1, half, half);
        return cat([-x2, x1], dim: -1);
    }

    public (Tensor q, Tensor k) Apply(Tensor q, Tensor k)
    {
        var (cos, sin) = Cache(q.shape[^2], q.device);
        return (q * cos + RotateHalf(q) * sin, k * cos + RotateHalf(k) * sin);
    }
}

/// <summary>Causal multi-head self-attention with rotary positions. Projections carry
/// bias.</summary>
public sealed class SelfAttentionWithRope : nn.Module<Tensor, Tensor>
{
    private readonly Linear q_proj, k_proj, v_proj, out_proj;
    private readonly RotaryEmbedding _rotary;
    private readonly long _heads, _headDim, _dModel;

    public SelfAttentionWithRope(long dModel, long nHeads) : base(nameof(SelfAttentionWithRope))
    {
        _dModel = dModel; _heads = nHeads; _headDim = dModel / nHeads;
        q_proj = nn.Linear(dModel, dModel);
        k_proj = nn.Linear(dModel, dModel);
        v_proj = nn.Linear(dModel, dModel);
        out_proj = nn.Linear(dModel, dModel);
        _rotary = new RotaryEmbedding(_headDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var (b, s) = (x.shape[0], x.shape[1]);
        Tensor Split(Tensor t) => t.view(b, s, _heads, _headDim).transpose(1, 2);

        var q = Split(q_proj.forward(x));
        var k = Split(k_proj.forward(x));
        var v = Split(v_proj.forward(x));
        (q, k) = _rotary.Apply(q, k);

        var attn = F.scaled_dot_product_attention(q, k, v, is_casual: true);
        var merged = attn.transpose(1, 2).contiguous().view(b, s, _dModel);
        return out_proj.forward(merged).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        foreach (var (proj, name) in new[] { (q_proj, "q_proj"), (k_proj, "k_proj"), (v_proj, "v_proj"), (out_proj, "out_proj") })
        {
            proj.weight!.set_(st.Get($"{p}.{name}.weight"));
            proj.bias!.set_(st.Get($"{p}.{name}.bias"));
        }
    }
}

/// <summary>Pre-norm block: <c>x + attn(norm1 x)</c>, then <c>x + ffn(norm2 x)</c>.</summary>
public sealed class KronosTransformerBlock : nn.Module<Tensor, Tensor>
{
    private readonly RmsNorm norm1, norm2;
    private readonly SelfAttentionWithRope self_attn;
    private readonly SwiGluFeedForward ffn;

    public KronosTransformerBlock(long dModel, long nHeads, long ffDim) : base(nameof(KronosTransformerBlock))
    {
        norm1 = new RmsNorm(dModel);
        self_attn = new SelfAttentionWithRope(dModel, nHeads);
        norm2 = new RmsNorm(dModel);
        ffn = new SwiGluFeedForward(dModel, ffDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = x + self_attn.forward(norm1.forward(x));
        return (h + ffn.forward(norm2.forward(h))).MoveToOuterDisposeScope();
    }

    /// <summary>Intermediates under the reference's export names, for bisection.</summary>
    public (Tensor norm1Out, Tensor attnOut, Tensor blockOut) Probe(Tensor x)
    {
        var h = norm1.forward(x);
        var a = self_attn.forward(h);
        var z1 = x + a;
        return (h, a, z1 + ffn.forward(norm2.forward(z1)));
    }

    public void Load(Safetensors st, string p)
    {
        norm1.weight.set_(st.Get($"{p}.norm1.weight"));
        norm2.weight.set_(st.Get($"{p}.norm2.weight"));
        self_attn.Load(st, $"{p}.self_attn");
        ffn.Load(st, $"{p}.ffn");
    }
}

/// <summary>Cross-attention with rotary positions. Two reference behaviours are
/// load-bearing for parity: <c>is_causal = self.training</c>, so inference is
/// <b>non-causal</b>; and the rotary cache is sized from the <i>query</i> length, which is
/// 1 during decode, making the rotation a no-op. Reproduce both.</summary>
public sealed class CrossAttentionWithRope : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Linear q_proj, k_proj, v_proj, out_proj;
    private readonly RotaryEmbedding _rotary;
    private readonly long _heads, _headDim, _dModel;

    public CrossAttentionWithRope(long dModel, long nHeads) : base(nameof(CrossAttentionWithRope))
    {
        _dModel = dModel; _heads = nHeads; _headDim = dModel / nHeads;
        q_proj = nn.Linear(dModel, dModel);
        k_proj = nn.Linear(dModel, dModel);
        v_proj = nn.Linear(dModel, dModel);
        out_proj = nn.Linear(dModel, dModel);
        _rotary = new RotaryEmbedding(_headDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor query, Tensor keyValue)
    {
        using var scope = NewDisposeScope();
        var (b, qLen) = (query.shape[0], query.shape[1]);
        var kLen = keyValue.shape[1];

        var q = q_proj.forward(query).view(b, qLen, _heads, _headDim).transpose(1, 2);
        var k = k_proj.forward(keyValue).view(b, kLen, _heads, _headDim).transpose(1, 2);
        var v = v_proj.forward(keyValue).view(b, kLen, _heads, _headDim).transpose(1, 2);
        (q, k) = _rotary.Apply(q, k);

        var attn = F.scaled_dot_product_attention(q, k, v, is_casual: false);
        var merged = attn.transpose(1, 2).contiguous().view(b, qLen, _dModel);
        return out_proj.forward(merged).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        foreach (var (proj, name) in new[] { (q_proj, "q_proj"), (k_proj, "k_proj"), (v_proj, "v_proj"), (out_proj, "out_proj") })
        {
            proj.weight!.set_(st.Get($"{p}.{name}.weight"));
            proj.bias!.set_(st.Get($"{p}.{name}.bias"));
        }
    }
}

/// <summary>Condition the sequence on a sibling subtoken: sibling is the query, hidden
/// states are key and value. A single-position query broadcasts across the sequence.</summary>
public sealed class DependencyAwareLayer : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly CrossAttentionWithRope cross_attn;
    private readonly RmsNorm norm;

    public DependencyAwareLayer(long dModel, long nHeads) : base(nameof(DependencyAwareLayer))
    {
        cross_attn = new CrossAttentionWithRope(dModel, nHeads);
        norm = new RmsNorm(dModel);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hiddenStates, Tensor siblingEmbed)
    {
        using var scope = NewDisposeScope();
        var attn = cross_attn.forward(siblingEmbed, hiddenStates);
        return norm.forward(hiddenStates + attn).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        cross_attn.Load(st, $"{p}.cross_attn");
        norm.weight.set_(st.Get($"{p}.norm.weight"));
    }
}

/// <summary>Fuse the two subtoken streams. Scale each embedding by <c>sqrt(d_model)</c>
/// before concatenation; omitting it is wrong by a constant factor and looks plausible.</summary>
public sealed class HierarchicalEmbedding : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Embedding emb_s1, emb_s2;
    private readonly Linear fusion_proj;
    private readonly double _scale;

    public HierarchicalEmbedding(long s1Bits, long s2Bits, long dModel) : base(nameof(HierarchicalEmbedding))
    {
        emb_s1 = nn.Embedding(1L << (int)s1Bits, dModel);
        emb_s2 = nn.Embedding(1L << (int)s2Bits, dModel);
        fusion_proj = nn.Linear(dModel * 2, dModel);
        _scale = Math.Sqrt(dModel);
        RegisterComponents();
    }

    /// <summary>Raw s1 lookup, unscaled and unfused: the dependency layer's sibling
    /// query. Not <see cref="forward"/>.</summary>
    public Tensor EmbedS1Raw(Tensor s1Ids) => emb_s1.forward(s1Ids);

    public override Tensor forward(Tensor s1Ids, Tensor s2Ids)
    {
        using var scope = NewDisposeScope();
        var a = emb_s1.forward(s1Ids) * _scale;
        var b = emb_s2.forward(s2Ids) * _scale;
        return fusion_proj.forward(cat([a, b], dim: -1)).MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        emb_s1.weight!.set_(st.Get($"{p}.emb_s1.weight"));
        emb_s2.weight!.set_(st.Get($"{p}.emb_s2.weight"));
        fusion_proj.weight!.set_(st.Get($"{p}.fusion_proj.weight"));
        fusion_proj.bias!.set_(st.Get($"{p}.fusion_proj.bias"));
    }
}

/// <summary>Calendar embedding: minute, hour, weekday, day, month, summed. Learned —
/// <c>learn_te</c> is true in the shipped config, so the sinusoidal path is never taken.</summary>
public sealed class TemporalEmbedding : nn.Module<Tensor, Tensor>
{
    private readonly Embedding minute_embed, hour_embed, weekday_embed, day_embed, month_embed;

    public TemporalEmbedding(long dModel) : base(nameof(TemporalEmbedding))
    {
        minute_embed = nn.Embedding(60, dModel);
        hour_embed = nn.Embedding(24, dModel);
        weekday_embed = nn.Embedding(7, dModel);
        day_embed = nn.Embedding(32, dModel);
        month_embed = nn.Embedding(13, dModel);
        RegisterComponents();
    }

    public override Tensor forward(Tensor stamp)
    {
        using var scope = NewDisposeScope();
        var s = stamp.to(ScalarType.Int64);
        Tensor Col(int i) => s.select(-1, i);
        var sum = hour_embed.forward(Col(1))
                + weekday_embed.forward(Col(2))
                + day_embed.forward(Col(3))
                + month_embed.forward(Col(4))
                + minute_embed.forward(Col(0));
        return sum.MoveToOuterDisposeScope();
    }

    public void Load(Safetensors st, string p)
    {
        minute_embed.weight!.set_(st.Get($"{p}.minute_embed.weight"));
        hour_embed.weight!.set_(st.Get($"{p}.hour_embed.weight"));
        weekday_embed.weight!.set_(st.Get($"{p}.weekday_embed.weight"));
        day_embed.weight!.set_(st.Get($"{p}.day_embed.weight"));
        month_embed.weight!.set_(st.Get($"{p}.month_embed.weight"));
    }
}

/// <summary>Two vocabulary projections: <c>proj_s1</c> over the sequence, <c>proj_s2</c>
/// over it once conditioned on the sampled coarse token.</summary>
public sealed class DualHead : nn.Module<Tensor, Tensor>
{
    private readonly Linear proj_s1, proj_s2;

    public DualHead(long s1Bits, long s2Bits, long dModel) : base(nameof(DualHead))
    {
        proj_s1 = nn.Linear(dModel, 1L << (int)s1Bits);
        proj_s2 = nn.Linear(dModel, 1L << (int)s2Bits);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => proj_s1.forward(x);
    public Tensor CondForward(Tensor x2) => proj_s2.forward(x2);

    public void Load(Safetensors st, string p)
    {
        proj_s1.weight!.set_(st.Get($"{p}.proj_s1.weight"));
        proj_s1.bias!.set_(st.Get($"{p}.proj_s1.bias"));
        proj_s2.weight!.set_(st.Get($"{p}.proj_s2.weight"));
        proj_s2.bias!.set_(st.Get($"{p}.proj_s2.bias"));
    }
}
