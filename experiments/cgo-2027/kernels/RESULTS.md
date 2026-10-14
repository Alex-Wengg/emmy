# Common-kernel study: latest results

This report separates the corrected standalone and whole-layer measurements from retained dynamic-FP8 failures.
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
The corrected standalone runs use unchanged generated kernels and schedules. The whole-layer comparison below
also uses the corrected timing protocol and replaces its earlier measurements.

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
then measure candidates under this corrected protocol. Whole-layer results measure a different, tuned workload.

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

## Common FP16 corpus: corrected whole-layer measurements

The workload is pretrained `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero,
FP16, batch one and sequence lengths 1 and 512. Compiler and recipe source is frozen at
`6399aac0b5fbf82ff761a75fc6c710d00a016ee3`. Its compiler implementation matches the standalone source above;
the later revision changes the common recipe and documentation. The private native runtime is identical.

Each shape runs five fresh actual-model processes. Emmy and full-graph Inductor share the module, pretrained
weights and inputs in each process. The ten committed per-card goldens supply measured schedules under
`--strict-evidence`; there is no tuning or evidence recording. Each process has a fresh database containing only
the scoped golden's imported measurements. Golden inputs and the ordered generated-source/schedule vectors stay
unchanged across repeats.

These are whole-forward latencies, not sums of separately timed kernels. Both backends capture complete batches
before timing and submit one graph per event window. Each backend normalizes by its own calibrated batch size.
The protocol uses deployable O3, hot caches, ten requested warmups and 100 measured iterations, with matched
exclusion of extended warmup. All common rows use standard math except H100 sequence 512, which enables
`FAST_EXP`; `FAST_MATH`, `F16_MMA_F32_ACC` and `FP8_MMA` remain disabled. H100 prefill retains its measured TMA
gate/up schedule. Clocks are not locked, and RTX 5090 also serves desktop graphics.

The table gives median backend latencies and the median of paired Inductor/Emmy ratios, with the observed ratio
range. Each invocation is capped at 115 seconds plus five seconds for termination. Every repeat is attempted.

| GPU | Sequence | Emmy, µs | Inductor, µs | Paired speedup [range] | Launches | Clean repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V100 SXM2 16GB | 1 | 58.907 | 69.947 | 1.187 [1.173–1.197] | 8 | 5/5 |
| V100 SXM2 16GB | 512 | 496.128 | 618.261 | 1.246 [1.244–1.247] | 21 | **4/5; task failed** |
| A100 SXM4 40GB | 1 | 49.775 | 54.462 | 1.094 [1.088–1.111] | 14 | 5/5 |
| A100 SXM4 40GB | 512 | 176.640 | 186.984 | 1.059 [1.057–1.064] | 11 | 5/5 |
| H100 80GB HBM3 | 1 | 28.163 | 33.435 | 1.192 [1.094–1.206] | 14 | 5/5 |
| H100 80GB HBM3 | 512 | 80.716 | 77.340 | 0.958 [0.954–0.962] | 11 | 5/5 |
| RTX 4090 24GB | 1 | 23.102 | 48.111 | 2.080 [2.076–2.095] | 14 | 5/5 |
| RTX 4090 24GB | 512 | 164.352 | 159.544 | 0.972 [0.970–0.973] | 11 | 5/5 |
| RTX 5090 32GB | 1 | 35.757 | 34.300 | 0.955 [0.954–0.977] | 13 | 5/5 |
| RTX 5090 32GB | 512 | 130.871 | 134.167 | 1.023 [1.018–1.027] | 11 | 5/5 |

Six of nine complete rows favor Emmy in every repeat. H100 prefill, RTX 4090 prefill and RTX 5090 sequence one
remain slower. V100 prefill favors Emmy in its four clean observations, but its incomplete row is excluded from
that count. These are tuned whole-layer results for one model and two shapes; they do not contradict the losses
in the four untuned standalone operators. Sequence one attends to one key, not a populated decode KV cache.

### Accuracy, failures and variation

All fifty raw comparisons contain captured timings for all three backends and passing dtype-scaled eager checks
for Emmy and full-graph Inductor. These are scaled checks, not strict `1e-3` live-model verdicts. Forty-nine
processes exit successfully. V100 prefill repeat zero exits 124 after writing its complete JSON receipt; that
observation is retained but excluded from the table's qualified medians. The task remains failed. No repeat was
rerun or replaced, and the timeout database's WAL is retained with the raw bundle.

H100 sequence one has the widest Emmy range: 27.715–30.584 µs, or 10.19% of its median. Its speedup stays above
one in every repeat. The ranges are descriptive, not confidence intervals. Source and schedule invariance does
not eliminate clock, cache or host variation.

A retained H100 strict live-model diagnostic at older source `27cecb6f4` failed on 5 of 524,288 values: maximum
absolute error 0.00391 and mean absolute error 4.96e-5. Its command, log and source attestation remain as a
qualification limitation. It is not a strict verdict on the corrected source, and no matched standard-math
observation establishes the cause. Earlier stored-program strict replays do not prove live-model equivalence.

### Source and reproduction

All five runs use the frozen source above, Python 3.12.3, PyTorch 2.13.0, Transformers 5.14.1 and Triton 3.7.1.
The operating systems, CUDA wheels, NVCC versions and drivers are the same as in the standalone systems table.
Each run uses one physical card; all 432 staged source files and each downloaded command bundle were verified.

| GPU | Common run ID |
| --- | --- |
| V100 | `20261014T051107Z` |
| A100 | `20261014T051701Z` |
| H100 | `20261014T051706Z` |
| RTX 4090 | `20261014T051116Z` |
| RTX 5090 | `20261014T051124Z` |

Save the archive, check out `6399aac0b5fbf82ff761a75fc6c710d00a016ee3`, and prepare the recorded runtime as in the
standalone reproduction instructions. Run `emmy bench experiments/cgo-2027/kernels --filter study=common` with
`--local` or the retained host's `--ssh`, an exact `--filter 'deploy.gpu=GPU_NAME'`, and `PYTHONPATH` set to the
checkout. The recipe copies the committed golden for each shape without modifying it.

## Dynamic-FP8 checkpoint results: historical failures

These runs also predate the timing fix. They retain failure and coverage evidence, not qualified speed comparisons.

The retained RTX runs measure layer zero of `RedHatAI/Qwen3-0.6B-FP8-dynamic` revision
`068a9040b238d65f5c1064f10232bb15b96c0ff0`, sequences 1 and 512, standard math with FP8 MMA, O3, ten warmups,
100 iterations and up to twelve tuning trials per measured kernel. This is a checkpoint-derived layer study.
The live eager and PyTorch references load decoded weights; they are not native FP8-library kernels.

All four final generated-program recordings pass the accepted scaled check. Every final FP8 task nevertheless
has failed terminal status: all twenty strict replays fail, and both live prefill comparisons exceed the existing
60-second GPU-stage limit. Those failures and their logs remain in the archive.

| GPU | Sequence | Strict generated-program replays | Live-model outcome |
| --- | ---: | --- | --- |
| RTX 4090 | 1 | 5/5 fail | Both backends pass scaled accuracy |
| RTX 4090 | 512 | 5/5 fail | GPU-stage timeout |
| RTX 5090 | 1 | 5/5 fail | Both backends pass scaled accuracy |
| RTX 5090 | 512 | 5/5 fail | GPU-stage timeout |

The generated and live programs can expose different outputs and select different kernels; their timings and
accuracy verdicts are separate comparisons. Strict failures include FP8 carrier differences and FP64 comparison
failures. RTX 5090 prefill's final output passes the FP64 allowance, but an intermediate scale peak fails; the
overall strict verdict remains failure. These runs support coverage analysis, not competitive FP8 performance or
strict numerical equivalence. No tolerance changed and no failed backend supplies a valid comparison latency.

These four failed tasks are the latest FP8 observations, not reruns with the corrected compiler. RTX 4090 uses
run `20261013T104811Z`, source `134c5bd4a`; RTX 5090 uses run `20261013T111703Z`, source `8af48e53c`.
Both use PyTorch 2.13.0 and Transformers 5.14.1, with the CUDA and driver versions in the systems table above.
Their raw timings remain in the evidence archive but are not reported as qualified speed comparisons.

The recipe's H200/B200 rows, larger layers, convergence and budget-ablation studies were not measured in this
reproduction. These whole-layer results do not reproduce the paper's isolated-kernel figure or geometric means,
and do not demonstrate an automatic LLM-plus-MCTS search within 30 minutes. The paper must match its claims and
protocol to the retained evidence. The [Neptune comparison](../compiler_neptune_emmy_pytorch_a100/RESULTS.md) and
[quantized-kernel study](../quantized_kernels_rtx5090/RESULTS.md) remain separate.

## Archive

`results.tar.gz` is the only results archive for this study. It holds 34 terminal experiment records and their
34 complete command artifact bundles: twenty successful standalone rows, nine successful common rows, one failed
common row and four failed historical FP8 rows. The corrected common runs replace the superseded whole-layer
records and bundles. Standalone evidence, the four FP8 failures and the separate H100 strict diagnostic are
retained byte-for-byte. Source attestations, the private native runtime and file hashes accompany the raw results.

| GPU | Corrected standalone directory | Corrected whole-layer directory |
| --- | --- | --- |
| V100 | `isolated/v100/2026-10-14_04-49-55/` | `common/v100/2026-10-14_05-11-07/` |
| A100 | `isolated/a100/2026-10-14_04-49-23/` | `common/a100/2026-10-14_05-17-01/` |
| H100 | `isolated/h100/2026-10-14_04-49-26/` | `common/h100/2026-10-14_05-17-06/` |
| RTX 4090 | `isolated/rtx4090/2026-10-14_04-49-56/` | `common/rtx4090/2026-10-14_05-11-16/` |
| RTX 5090 | `isolated/rtx5090/2026-10-14_04-49-31/` | `common/rtx5090/2026-10-14_05-11-24/` |

Each experiment record has a matching `_artifacts.tar.gz` beside it. For both corrected studies,
`verification/repeat-0.json` through `repeat-4.json` each contain all three backends and the paired correctness
verdicts. Standalone programs, common working goldens, input hashes, generated-source dumps, status files,
package freezes, fresh databases, device reports and cubins retain the execution context. The frozen recipes
are under `isolated/protocol/` and `common/protocol/`, source and remote-byte attestations under each study's
`provenance/`, and runtime build/import
receipts under `isolated/runtime/`.

The retained FP8 tasks remain under `rtx4090/2026-10-13_10-48-11/` and `rtx5090/2026-10-13_11-17-03/`.
Inside those historical command bundles, `torch-compile/model.json` holds the actual-model comparison;
`verification/repeat-0` through `repeat-4` contain strict replays, sometimes as extensionless JSON files and
sometimes as directories of target JSON files. Working goldens and tuning databases remain with those runs.

The separate H100 strict live diagnostic is under `h100/2026-10-13_20-29-15/follow-up-diagnostics/raw-fixed/`.
Its source is distinct from the final recipe run, as described above. The preliminary standalone run with biased
timing is excluded. No superseded platform snapshots, previous results archives or exploratory optimization
rounds are nested in this archive. Older committed snapshots remain available in Git history.

Archive size: 40,234,075 bytes. SHA-256:
`efe8722b4eb87727abd329ea23d5e30ae8b95037618cbf4e48f79d0e2244acb1`.
