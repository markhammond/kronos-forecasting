using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Tsfm.Forecasting.TimesFm;

/// <summary>Checkpoint hyper-parameters, read from the published config.json.</summary>
public sealed record TimesFmConfig(
    int InputPatchLen, int OutputPatchLen, double[] Quantiles,
    int NumLayers, long ModelDims, long NumHeads,
    bool UseVariateAttention, int MaxVariates, double ValueClip)
{
    public int Rolls => OutputPatchLen / InputPatchLen;
    public int NumQuantiles => Quantiles.Length;

    public static TimesFmConfig Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var tc = r.GetProperty("transformer_config");
        var t = tc.GetProperty("transformer");
        return new TimesFmConfig(
            r.GetProperty("input_patch_len").GetInt32(),
            r.GetProperty("output_patch_len").GetInt32(),
            [.. r.GetProperty("quantiles").EnumerateArray().Select(q => q.GetDouble())],
            tc.GetProperty("num_layers").GetInt32(),
            t.GetProperty("model_dims").GetInt64(),
            t.GetProperty("num_heads").GetInt64(),
            r.GetProperty("use_variate_attention").GetBoolean(),
            t.GetProperty("max_variates").GetInt32(),
            r.GetProperty("value_clip").GetDouble());
    }
}

/// <summary>
/// TimesFM 3.0: patch embedding, a stack of mixing layers, and a quantile head.
///
/// <para>One forward pass yields <c>OutputPatchLen</c> steps at every quantile, so a
/// predictive interval costs no more than a point forecast — unlike a sampled model,
/// where the interval is paid for in rollouts.</para>
/// </summary>
public sealed class TimesFmModel : Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly TimesFmConfig _cfg;

    public ResidualBlock pre_transformer_resblock;
    public StackedMixingTransformer transformer_stack;
    public Linear output_head;

    public TimesFmModel(TimesFmConfig cfg) : base(nameof(TimesFmModel))
    {
        _cfg = cfg;
        pre_transformer_resblock = new ResidualBlock(
            2 * (cfg.InputPatchLen + cfg.OutputPatchLen), cfg.ModelDims, cfg.ModelDims);
        transformer_stack = new StackedMixingTransformer(
            cfg.ModelDims, cfg.NumHeads, cfg.UseVariateAttention, cfg.NumLayers);
        output_head = Linear(cfg.ModelDims, cfg.OutputPatchLen * cfg.NumQuantiles, hasBias: true);
        RegisterComponents();
    }

    /// <summary>Patch embedding input and its projection — exposed so parity can be
    /// bisected rather than judged on the final logits alone.</summary>
    public (Tensor ResblockInput, Tensor ResblockOutput, Tensor PatchMask) Preprocess(
        Tensor values, Tensor masks, Tensor patchIsTarget)
    {
        using var scope = NewDisposeScope();
        values = nan_to_num(values, 0.0).clamp(-_cfg.ValueClip, _cfg.ValueClip);

        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(values, masks);
        var normalised = TimesFmPreprocess.Revin(values, mu, sigma);
        normalised = where(masks, zeros_like(normalised), normalised);

        var (fcov, wrap) = TimesFmPreprocess.OutputPatchViaRoll(values, _cfg.Rolls);
        fcov = TimesFmPreprocess.Revin(fcov, mu, sigma);
        var (fcovMasksRaw, _) = TimesFmPreprocess.OutputPatchViaRoll(masks, _cfg.Rolls);
        // A future slot is unusable if it is masked, belongs to the target variate, or
        // wrapped past the end of the sequence.
        var fcovMasks = fcovMasksRaw | patchIsTarget.unsqueeze(-1) | wrap;
        fcov = where(fcovMasks, zeros_like(fcov), fcov);

        var valuesCat = cat([normalised, fcov], -1);
        var masksCat = cat([masks, fcovMasks], -1);
        var rbIn = cat([valuesCat, masksCat.to(ScalarType.Float32)], -1);
        var rbOut = pre_transformer_resblock.forward(rbIn);
        var patchMask = masksCat.all(dim: 3);

        return (rbIn.MoveToOuterDisposeScope(),
                rbOut.MoveToOuterDisposeScope(),
                patchMask.MoveToOuterDisposeScope());
    }

    /// <returns>Logits (b, v, n, OutputPatchLen * NumQuantiles).</returns>
    public override Tensor forward(Tensor values, Tensor masks, Tensor patchIsTarget)
    {
        using var scope = NewDisposeScope();
        var (_, rbOut, patchMask) = Preprocess(values, masks, patchIsTarget);

        // Mask only LEADING fully-masked patches. Horizon patches are also fully masked
        // but follow real context, and must stay visible to attention — a plain
        // all-masked test would silently blind the model to its own horizon.
        var effective = patchMask.to(ScalarType.Int32).cumprod(2).to(ScalarType.Bool);

        var h = transformer_stack.forward(rbOut, effective);
        return output_head.forward(h).MoveToOuterDisposeScope();
    }

    /// <summary>Bind published weights by their PyTorch names. Every parameter must be
    /// matched: a silently unbound tensor leaves random init in place and still
    /// produces plausible forecasts.</summary>
    public static TimesFmModel Load(string checkpointDir, Device device)
    {
        var cfg = TimesFmConfig.Load(Path.Combine(checkpointDir, "config.json"));
        var net = new TimesFmModel(cfg);

        using var st = Safetensors.Load(Path.Combine(checkpointDir, "model.safetensors"));
        var named = net.named_parameters().ToDictionary(p => p.name, p => p.parameter);

        var bound = 0;
        var missing = new List<string>();
        using (no_grad())
        {
            foreach (var (name, param) in named)
            {
                if (!st.Has(name)) { missing.Add(name); continue; }
                using var t = st.Get(name);
                if (!t.shape.SequenceEqual(param.shape))
                    throw new InvalidDataException(
                        $"{name}: checkpoint {string.Join('x', t.shape)} vs model {string.Join('x', param.shape)}");
                param.copy_(t);
                bound++;
            }
        }
        if (missing.Count > 0)
            throw new InvalidDataException(
                $"{missing.Count} parameters absent from the checkpoint, e.g. {string.Join(", ", missing.Take(5))}");

        Console.Error.WriteLine($"bound {bound} parameters from {checkpointDir}");
        net.to(device);
        net.eval();
        return net;
    }
}
