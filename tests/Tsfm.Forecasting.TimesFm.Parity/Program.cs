// Bisecting parity harness. Compares the .NET port against reference tensors dumped by
// reference/dump_parity.py, one stage at a time.
//
// Final-logit agreement alone cannot distinguish a wrong RoPE ordering from a misplaced
// norm — both merely shift the output. Reporting each stage boundary localises the FIRST
// divergence, which is the only one worth debugging.

using Tsfm.Forecasting;
using Tsfm.Forecasting.TimesFm;

using static TorchSharp.torch;

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
var ckpt = Path.Combine(root, "checkpoints", "timesfm-3.0-pytorch");
var fixturePath = Path.Combine(root, "reference", "parity.safetensors");

if (!File.Exists(fixturePath))
{
    Console.Error.WriteLine($"missing {fixturePath}; run .venv/bin/python reference/dump_parity.py");
    return 1;
}

Device device;
try { device = new Device("mps"); using (ones(1).to(device)) { } }
catch { device = new Device("cpu"); }
Console.WriteLine($"device={device.type}\n");

using var _ = no_grad();
using var fx = Safetensors.Load(fixturePath);
Tensor E(string n) => fx.Get(n).to(device);

var model = TimesFmModel.Load(ckpt, device);

var values = E("in_values");
var masks = E("in_masks").to(ScalarType.Bool);
var isTarget = E("in_patch_is_target").to(ScalarType.Bool);

var ok = true;
Console.WriteLine($"{"stage",-24} {"max|Δ|",12} {"rel",12}  note");

void Check(string stage, Tensor got, string refName, double tol, string note = "")
{
    var want = E(refName);
    var absMax = (got - want).abs().max().item<float>();
    var scale = want.abs().max().item<float>();
    var rel = absMax / Math.Max(scale, 1e-12);
    var pass = rel <= tol || absMax <= 1e-6;
    ok &= pass;
    Console.WriteLine($"{stage,-24} {absMax,12:E3} {rel,12:E3}  {(pass ? "ok" : "FAIL")} {note}");
}

// 1. Running statistics — the causal recurrence that sets the model's scale.
var (rn, rmu, rsigma) = TimesFmPreprocess.RunningStats(values, masks);
Check("running_n", rn, "running_n", 1e-6);
Check("running_mu", rmu, "running_mu", 1e-5);
Check("running_sigma", rsigma, "running_sigma", 1e-5);

// 2. Patch embedding: catches the roll, the wrap mask and the 192-wide concatenation.
var (rbIn, rbOut, patchMask) = model.Preprocess(values, masks, isTarget);
Check("resblock_input", rbIn, "resblock_input", 1e-5);
Check("resblock_output", rbOut, "resblock_output", 1e-4);
Check("patch_mask", patchMask.to(ScalarType.Float32), "patch_mask", 1e-6);

// 3. Layers: the first divergence localises the fault to one sub-block.
var effective = patchMask.to(ScalarType.Int32).cumprod(2).to(ScalarType.Bool);
var h = rbOut;
for (var i = 0; i < model.transformer_stack.layers.Count; i++)
{
    h = model.transformer_stack.layers[i].forward(h, effective);
    if (i is 0 or 1 or 19) Check($"layer{i}_output", h, $"layer{i}_output", 2e-4);
}
Check("transformer_output", h, "transformer_output", 3e-4);

// 4. Quantile head.
var logits = model.output_head.forward(h);
Check("raw_logits", logits, "raw_logits", 3e-4, "(64 steps x 9 quantiles)");

Console.WriteLine();
Console.WriteLine(ok ? "PARITY: PASS" : "PARITY: FAIL — first failing stage above localises the fault");
return ok ? 0 : 1;
