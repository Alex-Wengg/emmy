# Common-kernel study: latest results

This report separates the standalone operator measurements from the retained whole-layer and dynamic-FP8 studies.
The standalone corpus defines four explicit computations on five GPUs; it does not recover the unidentified
boundaries behind the paper's old isolated-kernel bars. The single [results archive](results.tar.gz) contains the
corresponding records, complete command artifacts, correctness checks and source provenance. The recipe and ten
common-corpus goldens remain as reproduction inputs.

## Standalone operators

The `study=isolated` recipe defines twenty targets: four computations on V100 SXM2 16GB, A100 SXM4 40GB,
H100 80GB HBM3, RTX 4090 and RTX 5090. These are synthetic inputs with Qwen3-0.6B's hidden width, not pretrained
layer outputs. The RMSNorm formula and dimensions match the layer. Choosing the mean and cast as separate timed
computations is a new protocol decision; `Cast-P` does not assert identity with the old figure's `Copy-P` bar.

| Target | Inputs | Timed computation and output |
| --- | --- | --- |
| Norm-P | FP16 `x[1,512,1024]`, FP16 `weight[1024]` filled with ones | RMSNorm, FP16 `[1,512,1024]` |
| Norm-D | FP16 `x[1,1,1024]`, same weight | RMSNorm, FP16 `[1,1,1024]` |
| Mean-D | FP32 `z[1,1,1024]`, prepared by squaring FP16 random values in FP32 | Last-axis mean with dimensions retained, FP32 `[1,1,1]` |
| Cast-P | FP32 `v[1,512,1024]` | Materialized conversion to FP16 `[1,512,1024]` |

RMSNorm computes `u = float32(x)`, `v = u * rsqrt(mean(u*u, -1, keepdim=True) + 1e-6)`, then
`float16(v) * weight`. Every process constructs the inputs on CPU with `torch.manual_seed(0)` and `torch.randn`.
Input construction, transfers, compilation and reference checks are outside the measured computation. `P` and `D`
denote the two sequence lengths; these standalone operators do not perform attention or use a KV cache.

### Measurement protocol

The compiler, recipe and both priors are frozen at `8f391542c9ec78a06baa2ffad030369048c1c350`. Each target runs in
five fresh processes, each with a new tuning database, empty repository-golden scope and no evidence writes.
Ambient Emmy settings are cleared. All four precision controls (`FAST_MATH`, `FAST_EXP`, `F16_MMA_F32_ACC`,
`FP8_MMA`) are disabled, and compilation uses deployable O3. This measures the untuned prior-selected schedule;
there is no online search, LLM proposal, MCTS or manual schedule selection in this experiment.

Each invocation compares the same explicit frontend computation and inputs through eager PyTorch, full-graph
PyTorch 2.13 Inductor and Emmy. The CLI interleaves CUDA-graph-captured backends, with ten requested warmups and
100 measurement iterations in the hot-cache regime. Calibration can extend warmup; both sides exclude the same
warmup iterations. Each timing window submits one graph containing a complete batch of forwards, with matching
batch size for these single-launch cases. It reports the minimum per-call latency for each backend.
The table summarizes five invocation latencies by median and computes speedup as the median of five **paired**
Inductor/Emmy ratios. The bracketed speedup range is descriptive, not a confidence interval. No timing from an
unrelated invocation or stored-program replay contributes to a paired ratio.

The strict Emmy check uses eager outputs with `rtol=atol=0.001`. Inductor uses the CLI's existing dtype-scaled
eager check and must compile the complete graph. Raw receipts distinguish direct eager passes from the FP64
allowance. Each invocation is capped at 115 seconds, followed by at most five seconds for termination; every
repeat is attempted even if another fails. This protocol preserves failures instead of selecting passing runs.

The timing change is [PR #1169](https://github.com/cloudrift-ai/emmy/pull/1169). Previously, PyTorch replayed a
one-forward graph repeatedly from Python inside an event window, while Emmy captured the full batch. Host submission
gaps could inflate PyTorch latency. The first standalone run exposed this mismatch; its timings are excluded.
The corrected runs use unchanged generated kernels and schedules. Older whole-layer measurements below predate
the timing fix and require new paired measurements before supporting performance claims.

This is a hot-cache throughput-derived latency experiment with fixed inputs, not a request-latency or cold-cache
study. Fresh processes do not imply empty on-disk compiler caches. GPU clocks are not locked, and the local 5090
also serves desktop graphics. Interleaving and five repeats reduce some variation; they do not remove every bias.

### Corrected measurements

All twenty targets completed, with all 100 strict Emmy checks and all 100 full-graph Inductor checks passing.
Every backend was captured. Each Emmy computation uses one launch; its source and complete schedule are identical
across repeats and match the preliminary run. Every fresh database contains zero performance and routing rows.
Norm-P's maximum absolute error is 0.0009765625; the other three computations match eager exactly. None uses the
FP64 allowance.

Latency columns are medians of five process-level minima, in microseconds. Speedup is the median paired
Inductor/Emmy ratio; values above one favor Emmy. Ranges include all five ratios.

| GPU | Target | Emmy, µs | Inductor, µs | Paired speedup [range] |
| --- | --- | ---: | ---: | ---: |
| V100 | Norm-P | 3.618 | 3.318 | 0.917 [0.910–0.928] |
| V100 | Norm-D | 2.036 | 1.737 | 0.848 [0.844–0.855] |
| V100 | Mean-D | 2.086 | 1.909 | 0.915 [0.914–0.919] |
| V100 | Cast-P | 2.594 | 2.553 | 0.984 [0.982–0.989] |
| A100 | Norm-P | 2.884 | 2.602 | 0.905 [0.857–0.909] |
| A100 | Norm-D | 2.029 | 1.461 | 0.722 [0.718–0.726] |
| A100 | Mean-D | 2.184 | 1.973 | 0.904 [0.901–0.911] |
| A100 | Cast-P | 2.669 | 2.837 | 1.062 [1.056–1.063] |
| H100 | Norm-P | 2.490 | 1.872 | 0.750 [0.703–0.804] |
| H100 | Norm-D | 1.757 | 1.361 | 0.777 [0.767–0.779] |
| H100 | Mean-D | 1.536 | 1.359 | 0.885 [0.882–0.895] |
| H100 | Cast-P | 1.552 | 1.512 | 0.973 [0.972–0.978] |
| RTX 4090 | Norm-P | 1.889 | 1.721 | 0.912 [0.908–0.916] |
| RTX 4090 | Norm-D | 1.266 | 0.961 | 0.763 [0.759–0.763] |
| RTX 4090 | Mean-D | 1.130 | 1.003 | 0.887 [0.884–0.889] |
| RTX 4090 | Cast-P | 1.470 | 1.476 | 1.004 [0.996–1.007] |
| RTX 5090 | Norm-P | 1.787 | 1.656 | 0.928 [0.922–0.938] |
| RTX 5090 | Norm-D | 1.143 | 0.818 | 0.725 [0.701–0.734] |
| RTX 5090 | Mean-D | 0.968 | 0.806 | 0.834 [0.833–0.842] |
| RTX 5090 | Cast-P | 1.013 | 0.986 | 0.974 [0.972–0.983] |

| GPU | Geometric mean of four paired median speedups | Medians above one |
| --- | ---: | ---: |
| V100 | 0.915 | 0/4 |
| A100 | 0.890 | 1/4 |
| H100 | 0.842 | 0/4 |
| RTX 4090 | 0.887 | 1/4 |
| RTX 5090 | 0.860 | 0/4 |

This corpus shows no broad win. Two of twenty medians exceed one, but RTX 4090 casting crosses parity across
repeats and is effectively tied. Only A100 casting favors Emmy in every repeat, by about 6.2% in speedup. The
geometric mean across all twenty paired medians is 0.878. H100 Norm-P has the widest ratio range; none of its
repeats reverses the loss. These descriptive summaries cover only four untuned standalone operators. They do
not establish results for GEMM, attention, a tuned kernel corpus, FP8 projections or whole-layer execution.

The next optimization target is Norm-D: its gap appears on every GPU, with a median paired Emmy/Inductor latency
ratio of about 1.38x on A100 and RTX 5090. Compare the selected reduction schedule and memory operations with Inductor,
then measure candidates under this corrected protocol. Whole-layer claims separately require a corrected rerun.

### Standalone source and systems

All five runs use source `8f391542c9ec78a06baa2ffad030369048c1c350`, Python 3.12.3, PyTorch 2.13.0 and
Triton 3.7.1 on Ubuntu 24.04 x86-64. Source attestations cover all 432 staged tracked files. The exact same
private native binary is used on every card, SHA-256
`0fcb53578875cc28b56acbe9303d05f20114fcabc1e6778a5b8ae21c29ee44b9`, built from
`79ccb2cec069b7c2d106ad26577c1c4604935b31`. Its Rust sources match the frozen experiment checkout; the archive
includes the binary, locked build-input hashes and compiler-version receipts. Package freezes describe installed
packages; the source attestations and explicit import paths identify the code actually measured.

| GPU | Run ID | CUDA wheel / NVCC | NVIDIA driver |
| --- | --- | --- | --- |
| V100 SXM2 16GB | `20261014T044955Z` | 12.6 / 12.9.86 | 580.178.04 |
| A100 SXM4 40GB | `20261014T044923Z` | 13.0 / 13.0.88 | 580.178.04 |
| H100 80GB HBM3 | `20261014T044926Z` | 13.0 / 13.0.88 | 580.178.04 |
| RTX 4090 24GB | `20261014T044956Z` | 13.0 / 12.9.86 | 580.159.03 |
| RTX 5090 32GB | `20261014T044931Z` | 13.0 / 13.0.88 | 580.178.04 |

To reproduce, save this results archive, then check out the frozen source and prepare the recorded Python/CUDA
environment on the exact card.
The isolated lane deliberately reuses that environment. Extract `isolated/runtime/` from the archive and copy
its native library into `~/.cache/emmy/cgo-2027-runtime/79ccb2cec069b7c2d106ad26577c1c4604935b31/` on the execution
host. The recipe verifies its hash before use. The supplied binary targets these Ubuntu x86-64/Python 3.12 hosts;
building for another platform creates a new runtime provenance and requires an explicit recipe update.

Run `emmy bench experiments/cgo-2027/kernels --filter study=isolated` with `--local` or the existing host's `--ssh`
and an exact `--filter 'deploy.gpu=GPU_NAME'`, with `PYTHONPATH` set to the checkout. The twenty matrix rows define
the full denominator; do not substitute a hardware alias or select only favorable operators. Compilation and
accuracy checks are included in each bounded process, but excluded from reported latency.

## Common FP16 corpus: historical measurements

The retained whole-layer results below use the earlier timing harness. Their correctness receipts remain useful;
their performance ratios are unqualified until both backends are remeasured with complete captured batches.

The workload is pretrained `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero,
FP16, batch one and sequence lengths 1 and 512. These are captured whole-forward latencies, with deployable O3,
hot-cache timing, ten warmups and 100 iterations. Emmy and full-graph `torch.compile` share the module, weights and
inputs within each actual-model invocation. Each table row is one invocation, not a repeated estimate of its margin.

All common rows use standard math except H100 sequence 512, which enables `FAST_EXP`. `FAST_MATH`,
`F16_MMA_F32_ACC` and `FP8_MMA` remain disabled. H100's accepted prefill uses the measured TMA gate/up schedule.
Later H100 profiling and candidate trials did not replace this accepted run.

| GPU | Sequence | Emmy, µs | PyTorch, µs | PyTorch / Emmy | Launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 SXM2 16GB | 1 | 54.218 | 65.521 | 1.208x | 8 |
| V100 SXM2 16GB | 512 | 494.080 | 606.804 | 1.228x | 21 |
| A100 SXM4 40GB | 1 | 50.595 | 55.501 | 1.097x | 14 |
| A100 SXM4 40GB | 512 | 178.005 | 188.738 | 1.060x | 11 |
| H100 80GB HBM3 | 1 | 27.100 | 29.400 | 1.085x | 14 |
| H100 80GB HBM3 | 512 | 81.999 | 77.805 | 0.949x | 11 |
| RTX 4090 24GB | 1 | 23.715 | 31.736 | 1.338x | 14 |
| RTX 4090 24GB | 512 | 165.365 | 160.172 | 0.969x | 11 |
| RTX 5090 32GB | 1 | 36.856 | 36.699 | 0.996x | 13 |
| RTX 5090 32GB | 512 | 129.860 | 136.043 | 1.048x | 11 |

The earlier table reported seven of ten wins, but the timing mismatch prevents using that count as performance
evidence. Sequence one attends to one key; it does not measure decode with a populated KV cache.

### Correctness and repeatability

All ten actual-model comparisons pass the existing dtype-scaled eager check for both backends. Five fresh-process
strict replays per common shape use generated inputs and their own eager program, with `rtol=atol=0.001`, fresh
tuning databases and strict measured evidence. Their inputs differ from the actual-model comparison, so their
latencies must not be divided into the PyTorch times above to construct speedups.

| GPU | Sequence | Strict replay median [range], µs |
| --- | ---: | ---: |
| V100 | 1 | 53.736 [53.638–54.028] |
| V100 | 512 | 494.080 [492.032–496.128] |
| A100 | 1 | 50.269 [50.252–50.570] |
| A100 | 512 | 177.493 [177.323–177.835] |
| H100 | 1 | 23.612 [23.289–24.080] |
| H100, fast exponential | 512 | 82.594 [82.213–83.045] |
| RTX 4090 | 1 | 23.719 [23.671–23.746] |
| RTX 4090 | 512 | 164.523 [164.011–165.717] |
| RTX 5090 | 1 | 36.839 [36.826–36.858] |
| RTX 5090 | 512 | 130.967 [129.400–131.076] |

The 50 principal strict receipts all pass. Ten additional A100/H100 decode receipts cover alternate stored pin
spellings, and five additional H100 prefill receipts cover the standard-math TMA regime; these also pass. They are
not additional workload shapes. The standard H100 prefill median is 83.239 µs [83.116–83.510].

A separate H100 strict live-model diagnostic at source `27cecb6f4` failed on 5 of 524,288 values: maximum absolute
error 0.00391, mean absolute error 4.96e-5 and maximum relative error 0.488. Its command, log and source attestation
are retained as a qualification limitation. This is not a strict live-model verdict from the final `911ca5739`
run, and there is no matched standard-math observation establishing the cause. The accepted actual-model results
use the scaled check; passing stored-program replays do not establish strict live-model equivalence.

## Dynamic-FP8 checkpoint results: historical failures

These runs also predate the timing fix. They retain failure and coverage evidence, not qualified speed comparisons.

The final RTX runs also measure layer zero of `RedHatAI/Qwen3-0.6B-FP8-dynamic` revision
`068a9040b238d65f5c1064f10232bb15b96c0ff0`, sequences 1 and 512, standard math with FP8 MMA, O3, ten warmups,
100 iterations and up to twelve tuning trials per measured kernel. This is a checkpoint-derived layer study.
The live eager and PyTorch references load decoded weights; they are not native FP8-library kernels.

All four final generated-program recordings pass the accepted scaled check. Every final FP8 task nevertheless
has failed terminal status: all twenty strict replays fail, and both live prefill comparisons exceed the existing
60-second GPU-stage limit. Those failures and their logs remain in the archive.

| GPU | Sequence | Final generated-program Emmy, µs | Paired eager, µs | Paired PyTorch, µs | Strict replays |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 4090 | 1 | 1449.984 | 485.945 | Accuracy failure | 5/5 fail |
| RTX 4090 | 512 | 271806.458 | 763.480 | Accuracy failure | 5/5 fail |
| RTX 5090 | 1 | 1309.248 | 388.719 | 28.655 | 5/5 fail |
| RTX 5090 | 512 | 238831.009 | 612.200 | Accuracy failure | 5/5 fail |

| GPU | Sequence | Live-model Emmy, µs | Live-model PyTorch, µs | Outcome |
| --- | ---: | ---: | ---: | --- |
| RTX 4090 | 1 | 1171.456 | 30.692 | Both pass scaled accuracy |
| RTX 4090 | 512 | — | — | GPU-stage timeout; no accuracy or timing result |
| RTX 5090 | 1 | 1048.128 | 36.808 | Both pass scaled accuracy |
| RTX 5090 | 512 | — | — | GPU-stage timeout; no accuracy or timing result |

The generated and live programs can expose different outputs and select different kernels; their timings and
accuracy verdicts are separate comparisons. Strict failures include FP8 carrier differences and FP64 comparison
failures. RTX 5090 prefill's final output passes the FP64 allowance, but an intermediate scale peak fails; the
overall strict verdict remains failure. These runs support coverage analysis, not competitive FP8 performance or
strict numerical equivalence. No tolerance changed and no failed backend supplies a valid comparison latency.

## Historical source and system provenance

Each GPU uses one physical card. The whole-layer runs retain their original source revisions; rebasing the
paper branch does not remeasure them at the new compiler head. All runs use PyTorch 2.13.0 and Transformers 5.14.1.
RTX 5090 shares local desktop graphics use; the other four cards are remote.

| GPU | Run ID | Source revision | CUDA wheel / NVCC | NVIDIA driver |
| --- | --- | --- | --- | --- |
| V100 | `20261013T061146Z` | `dd7f4b766` | 12.6 / 12.9.86 | 580.178.04 |
| A100 | `20261013T175052Z` | `97949ef00` | 13.0 / 13.0.88 | 580.178.04 |
| H100 | `20261013T202915Z` | `911ca5739` | 13.0 / 13.0.88 | 580.178.04 |
| RTX 4090 | `20261013T104811Z` | `134c5bd4a` | 13.0 / 12.9.86 | 580.159.03 |
| RTX 5090 | `20261013T111703Z` | `8af48e53c` | 13.0 / 13.0.88 | 580.178.04 |

The recipe's H200/B200 rows, larger layers, convergence and budget-ablation studies were not measured in this
reproduction. These whole-layer results do not reproduce the paper's isolated-kernel figure or geometric means,
and do not demonstrate an automatic LLM-plus-MCTS search within 30 minutes. The paper must match its claims and
protocol to the retained evidence. The [Neptune comparison](../compiler_neptune_emmy_pytorch_a100/RESULTS.md) and
[quantized-kernel study](../quantized_kernels_rtx5090/RESULTS.md) remain separate.

## Archive

`results.tar.gz` is the only results archive for this study. It holds 34 terminal experiment records and their
34 complete command artifact bundles: twenty successful corrected standalone rows, ten historical common rows
and four failed historical FP8 rows. All 75 payloads from the previous consolidated archive remain byte-identical.
The 144 direct files include logs, source attestations, the private native runtime and a manifest with sizes and
SHA-256 hashes. The command bundles contain 13,580 regular payload files in total.

| GPU | Corrected standalone directory | Historical whole-layer directory |
| --- | --- | --- |
| V100 | `isolated/v100/2026-10-14_04-49-55/` | `v100/2026-10-13_06-11-46/` |
| A100 | `isolated/a100/2026-10-14_04-49-23/` | `a100/2026-10-13_17-50-52/` |
| H100 | `isolated/h100/2026-10-14_04-49-26/` | `h100/2026-10-13_20-29-15/` |
| RTX 4090 | `isolated/rtx4090/2026-10-14_04-49-56/` | `rtx4090/2026-10-13_10-48-11/` |
| RTX 5090 | `isolated/rtx5090/2026-10-14_04-49-31/` | `rtx5090/2026-10-13_11-17-03/` |

Each experiment record has a matching `_artifacts.tar.gz` beside it. For standalone runs,
`verification/repeat-0.json` through `repeat-4.json` each contain all three backends and the paired correctness
verdicts. `program.py`, input hashes, generated-source dumps, status files, package freezes, fresh databases,
device reports and cubins retain the execution context. The frozen recipe is under `isolated/protocol/`, source
and remote-byte attestations under `isolated/provenance/`, and runtime build/import receipts under
`isolated/runtime/`.

Inside historical command bundles, `torch-compile/model.json` holds the actual-model comparison;
`verification/repeat-0` through `repeat-4` contain strict replays, sometimes as extensionless JSON files and
sometimes as directories of target JSON files. Working goldens and tuning databases remain with those runs.

The separate H100 strict live diagnostic is under `h100/2026-10-13_20-29-15/follow-up-diagnostics/raw-fixed/`.
Its source is distinct from the final recipe run, as described above. The preliminary standalone run with biased
timing is excluded. No superseded platform snapshots, previous results archives or exploratory optimization
rounds are nested in this archive. Older committed snapshots remain available in Git history.

Archive size: 11,474,358 bytes. SHA-256:
`a8581d5edf481ea5d1fbedd381c4c0041a3c2fa3491a1bc4b3f6da1952956581`.
