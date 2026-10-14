# Common-kernel study

This study contains four explicit standalone computations and a Qwen3-0.6B layer on five GPUs. The historical
standalone measurements are complete. The current run after scalar binding and prior refitting has four complete
Norm-D rows and nine complete whole-layer rows; two V100 rows are incomplete.
Four historical dynamic-FP8 tasks remain failed; they supply coverage and failure evidence, not qualified
performance comparisons.

The standalone boundaries are defined below. They do not recover the unidentified boundaries behind the paper's
old isolated-kernel bars. The [results archive](results.tar.gz) retains the raw records, commands, accuracy checks,
source provenance and migration evidence. The recipe and ten common-corpus goldens are reproduction inputs.

## Protocol and scope

Each invocation compares eager PyTorch, full-graph PyTorch 2.13 Inductor and Emmy on the same frontend computation
and inputs. The recipe runs five fresh processes per target, with deployable O3, hot caches, ten requested warmups
and 100 measurement iterations. Calibration may extend warmup; every backend excludes those iterations.
Backends are interleaved. Each timing window submits one captured graph containing a complete batch of forwards,
then normalizes by that backend's batch size. Batch sizes match for the single-launch standalone cases.

Complete rows report the median of five process-level minimum latencies and the median of five paired
Inductor/Emmy ratios. Incomplete rows report only their stated number of clean repeats and remain excluded from
complete-row summaries. Values above one favor Emmy. The ratio range includes every qualified repeat and is
descriptive, not a confidence interval. No unrelated invocation or stored-program replay contributes to a paired
ratio. Each process is capped at 115 seconds plus five seconds for termination; every repeat is attempted.
Nonzero exits remain failures even when a process has already written a timing receipt.

This complete-batch timing protocol was introduced in [PR #1169](https://github.com/cloudrift-ai/emmy/pull/1169).
Previously, PyTorch replayed a one-forward graph repeatedly from Python within the event window while Emmy
captured the complete batch. Host submission gaps could inflate PyTorch latency. The preliminary standalone
measurements exposed that mismatch and are excluded. All standalone numbers below use the correction.

These are fixed-input, hot-cache throughput-derived latencies, not request latencies or cold-cache measurements.
Fresh processes do not imply empty compiler caches. Clocks are not locked, and the local RTX 5090 also serves
desktop graphics. Five repeats and interleaving do not remove every source of variation.

## Historical standalone operators

The twenty targets combine four computations with V100 SXM2 16GB, A100 SXM4 40GB, H100 80GB HBM3, RTX 4090 24GB
and RTX 5090 32GB. Inputs are synthetic and use Qwen3-0.6B's hidden width. They are not pretrained layer outputs.
The RMSNorm formula and dimensions match the layer; timing the mean and cast separately is a prospective protocol
choice. `Cast-P` does not assert identity with the old figure's `Copy-P` bar.

| Target | Inputs | Timed computation and output |
| --- | --- | --- |
| Norm-P | FP16 `x[1,512,1024]`, FP16 unit `weight[1024]` | RMSNorm, FP16 `[1,512,1024]` |
| Norm-D | FP16 `x[1,1,1024]`, same weight | RMSNorm, FP16 `[1,1,1024]` |
| Mean-D | FP32 `z[1,1,1024]`, prepared by squaring FP16 random values in FP32 | Last-axis mean retaining dimensions, FP32 `[1,1,1]` |
| Cast-P | FP32 `v[1,512,1024]` | Materialized conversion to FP16 `[1,512,1024]` |

RMSNorm computes `u = float32(x)`, `v = u * rsqrt(mean(u*u, -1, keepdim=True) + 1e-6)`, then
`float16(v) * weight`. Every process constructs CPU inputs with `torch.manual_seed(0)` and `torch.randn`.
Construction, transfers, compilation and reference checks are outside the measured computation. `P` and `D`
denote sequence lengths 512 and one; neither computation performs attention or uses a KV cache.

Compiler, recipe and both priors are frozen at `8f391542c9ec78a06baa2ffad030369048c1c350`. Each process uses an
empty golden scope, a fresh database and no evidence writes. Ambient Emmy settings are cleared. All four
precision controls (`FAST_MATH`, `FAST_EXP`, `F16_MMA_F32_ACC`, `FP8_MMA`) are false. This measures the untuned
prior-selected schedule, without online search, LLM proposals, MCTS or manual schedule selection.

Emmy is checked against eager with `rtol=atol=0.001`. Inductor uses the CLI's existing dtype-scaled eager check
and must compile the complete graph. All 100 strict Emmy checks and all 100 Inductor checks passed, and every
backend was captured. Norm-P's maximum absolute error is 0.0009765625; the other three computations match eager
exactly. No invocation uses the FP64 allowance. Every Emmy computation has one launch, with source and complete
schedule unchanged across repeats. Every fresh database contains zero performance and routing rows.

Latency columns are medians in microseconds. Speedup is the median paired Inductor/Emmy ratio.

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

There is no broad win: the geometric mean across all twenty paired medians is 0.878. Two medians exceed one,
but RTX 4090 casting crosses parity across repeats. Only A100 casting favors Emmy in every repeat, with about
6.2% speedup. H100 Norm-P has the widest ratio range, but every repeat remains slower. These results cover four
untuned standalone operators, not GEMM, attention, FP8 projections or tuned whole-layer execution.

## Current Norm-D after scalar binding

The current compiler binds fixed FP32 denominators and epsilon before fusion. Values are rounded to FP32 storage
first, then represented as typed literals in the kernel body and exact identity. Runtime inputs and
checkpoint-backed constants remain device buffers. This removes two scalar loads from Norm-D without changing
its formula, precision controls or maximal fusion policy.

The final run repeats Norm-D on all five cards with the historical frontend program, synthetic inputs and timing
protocol. Both priors were refitted from all fourteen repository goldens at feature version 10, using five-fold
cross-validation. The unchanged reproduction gate passed all 250 cases, including the 51 NVFP4 cases restored
after removing the old pending tag. Each process uses an empty evidence scope and a fresh database, so it measures
the current prior-selected schedule. This is an execution comparison, not a held-out prior evaluation. The
historical and final runs occur at different times with unlocked clocks; only ratios within each invocation are
direct paired comparisons.

The final source is `0add27dbb415f5336d04e9d11e19e1c8799d43dd`. The archive records the recipe and both
prior hashes, complete fit metrics, dataset manifests and the 250-case reproduction receipt. Schedule evaluation
completed; serial placement evaluation reached its 20-minute cap. That timeout is retained alongside the
successful parallel reproduction checks. The approved placement-export retry completed in 37 minutes.

| GPU | Emmy, µs | Inductor, µs | Paired speedup [range] | Clean repeats |
| --- | ---: | ---: | ---: | --- |
| V100 SXM2 16GB | 1.848 | 1.735 | 0.939 [0.930–0.948] | 2 / 5, incomplete |
| A100 SXM4 40GB | 1.719 | 1.456 | 0.848 [0.847–0.852] | 5 / 5 |
| H100 80GB HBM3 | 1.542 | 1.356 | 0.879 [0.873–0.881] | 5 / 5 |
| RTX 4090 24GB | 1.126 | 1.009 | 0.900 [0.896–0.901] | 5 / 5, remote completion recovered |
| RTX 5090 32GB | 0.919 | 0.830 | 0.906 [0.895–0.907] | 5 / 5 |

A100, H100, RTX 4090 and RTX 5090 each complete five clean repeats. All backends are captured. Emmy matches eager
exactly under the strict `1e-3` check, and Inductor passes its scaled check. Each card selects one launch with
invariant source and schedule; every fresh database remains empty. All four cards remain below Inductor parity. Their
paired ratios are higher than the historical Norm-D ratios, but the separate dates and changed priors prevent
attributing that difference solely to scalar binding. A100's Emmy latency varies by 12% across repeats while its
paired ratio varies by less than 1%, which reinforces comparing the backends within the same invocation.

The local controller was lost during an out-of-memory incident at 22:13:09 UTC. RTX 4090's remote task completed
with exit zero, and all five process receipts and remote artifact hashes were recovered. V100 produced all five
strict-passing captured receipts, but the last three pipelines exited 141 after the output connection was lost.
They remain diagnostic evidence and do not enter the table. The V100 task failed; its two clean observations do
not establish a completed five-repeat comparison. No failed repeat was rerun. Both experiment records were
finalized from the recovered remote outcomes. Their completion timestamp is the recovery time, not the remote
measurement completion time. The original `running` records are preserved with the recovery proof.

The 25 attempted repeats produce 25 raw receipts and 22 clean exits. No five-card aggregate is reported, and the
historical twenty-case geometric mean is not updated with a mixture of current and historical measurements.

### Profiling evidence

Cold-cache Nsight Compute runs on A100 and H100 compare the kernels before and after scalar binding. Each captures
one launch using kernel replay, cache flushing and no clock control. Global-load sectors fall from 160 to 128 and
LSU warp instructions from 262 to 230 on both cards. Total warp instructions fall from 1,176 to 1,000 on A100 and
from 1,324 to 1,116 on H100. Registers remain at 17 and 18, respectively. Long-scoreboard stall percentages fall
from 54.69% to 35.09% on A100 and from 64.55% to 50.43% on H100 in these observations.

These counters establish less memory work for those profiled kernels. They do not measure the hot-cache protocol
or establish a speedup. The archive retains the exact profiling sources, successful and failed commands, full
counter output, CUDA, SASS and runtime attestations. Profiler durations do not enter any latency table. Both A100
and H100 select a 128-thread kernel in the final run, which differs from the profiled 512-thread kernel. The
counters characterize the archived profiling schedule and cannot be attributed to the final timed kernel.

## Current common corpus after migration

The workload is pretrained `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer zero,
FP16, batch one, sequences one and 512. Sequence one attends to one key, not a populated decode KV cache.
Each shape runs five fresh actual-model processes. Emmy and full-graph Inductor share the module, pretrained
weights and inputs within each process. These are whole-forward latencies, not sums of separately timed kernels.

The ten per-card goldens supply measured schedules under `--strict-evidence`, with no tuning or recording.
Each fresh database may contain only the scoped golden's imported evidence. All precision controls are false
except H100 sequence 512, which retains its measured `FAST_EXP` regime and TMA gate/up schedule. Both backends
must pass the CLI's existing dtype-scaled eager check. These are scaled live-model checks, not strict `1e-3`
verdicts. All 49 clean repeats across five cards pass both backend checks and capture all three backends. Ordered
source and schedule vectors are invariant within each row's clean repeats. Their fresh databases contain only
scoped golden evidence, with no new measurements.

The final source and priors use the freeze declared above. This table replaces the earlier common run
at `6399aac0b5fbf82ff761a75fc6c710d00a016ee3`. That historical run had nine complete rows and one failed V100
prefill task: repeat zero exited 124 after writing JSON and was excluded from its four-observation median.
No historical common latency is carried into the current table; its previous report and archive remain in Git.

| GPU | Sequence | Emmy, µs | Inductor, µs | Paired speedup [range] | Launches | Clean repeats |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V100 SXM2 16GB | 1 | 57.539 | 69.908 | 1.214 [1.214–1.226] | 8 | 5 / 5 |
| V100 SXM2 16GB | 512 | 488.448 | 618.218 | 1.266 [1.263–1.271] | 21 | 4 / 5, incomplete |
| A100 SXM4 40GB | 1 | 47.151 | 54.974 | 1.161 [1.140–1.170] | 14 | 5 / 5 |
| A100 SXM4 40GB | 512 | 170.496 | 187.455 | 1.101 [1.088–1.103] | 11 | 5 / 5 |
| H100 80GB HBM3 | 1 | 28.964 | 33.556 | 1.161 [1.130–1.252] | 14 | 5 / 5 |
| H100 80GB HBM3 | 512 | 78.512 | 77.423 | 0.990 [0.986–0.992] | 11 | 5 / 5 |
| RTX 4090 24GB | 1 | 22.378 | 48.040 | 2.147 [2.125–2.158] | 14 | 5 / 5 |
| RTX 4090 24GB | 512 | 163.669 | 159.895 | 0.977 [0.974–0.980] | 11 | 5 / 5 |
| RTX 5090 32GB | 1 | 36.800 | 34.589 | 0.919 [0.783–0.979] | 13 | 5 / 5 |
| RTX 5090 32GB | 512 | 130.120 | 141.345 | 1.093 [1.037–1.103] | 11 | 5 / 5 |

V100 prefill repeat zero reached the process cap with exit 124 after CUDA lowering, before any accuracy or timing
receipt. Its lowered sources match the next repeat. The other four repeats complete and supply the explicitly
incomplete row above; they do not convert the task to success. The failed attempt and its artifacts are retained.

Six of nine complete rows favor Emmy in every repeat. Sequence one favors Emmy on V100, A100, H100 and RTX 4090;
RTX 5090 favors Inductor. For sequence 512, A100 and RTX 5090 favor Emmy, while H100 and RTX 4090 favor Inductor.
The incomplete V100 prefill row remains outside that count. These are tuned whole-layer results, which do not
establish a broad standalone-kernel win.

RTX 5090 sequence one has the widest paired ratio range, 0.783–0.979. Inductor latency varies from 29.352 to
35.609 µs, while Emmy varies from 36.365 to 37.625 µs. The sources and schedules remain unchanged. All five repeats
remain below parity, but the range limits the precision of a single slowdown estimate. The local card also runs
desktop graphics, and clocks are unlocked. No repeat was replaced or discarded because of its performance.

### Migration evidence

The ten goldens retain ten programs, 215 kernels, 80 routing decisions and 184 authored rows. Of 173 measured
rows, 102 retain the same exact kernel identity and measurement. The other 71 slots receive fresh evidence from
68 successful guarded recordings; three equivalent V100 aliases share exact evidence keys. Eleven original
proposals remain unmeasured. All ten files pass fresh-lowering checks.

Every restored row joins its raw result, guarded writer and captured database measurement by exact kernel
identity, bindings, card, cache mode, effective precision and complete schedule. The isolated median stays beside
its own invocation's reference timing. These strict checks use same-input greedy execution; they do not establish
whole-layer eager accuracy or latency. Unpinned strict-evidence source selection passes on all cards, including
both H100 prefill precision regimes. Slower recorded alternatives remain in the goldens.

Working copies, authored cut correspondence, commands, databases, failed preparation/selection attempts and
source checks are retained. H100's first eight recordings overlapped a CPU prior-evaluation process; subsequent
records followed its pause, without extending that evaluation's original timeout. RTX 5090 recording occurred
between unrelated jobs without GPU measurement overlap. Final whole-layer runs are separate, after prior refitting.

A historical H100 strict live-model diagnostic at `27cecb6f4` failed on 5 of 524,288 values: maximum absolute
error 0.00391 and mean absolute error 4.96e-5. Its command, log and source attestation remain a limitation.
It is not a strict verdict on the final source, and no matched standard-math observation establishes its cause.
Stored-program strict passes do not establish live-model strict equivalence.

## Historical dynamic-FP8 failures

These observations predate the timing correction. They retain failure and coverage evidence, not qualified speed
comparisons. The checkpoint is `RedHatAI/Qwen3-0.6B-FP8-dynamic` revision
`068a9040b238d65f5c1064f10232bb15b96c0ff0`, layer zero, sequences one and 512. Runs use standard math with FP8 MMA,
O3, ten warmups, 100 iterations and up to twelve tuning trials per measured kernel. Live eager and Inductor
references use decoded checkpoint weights; they are not native FP8-library kernels.

All four final generated-program recordings pass the accepted scaled check, but every task terminates failed.
All twenty strict replays fail, and both live prefill comparisons exceed the existing 60-second GPU-stage limit.

| GPU | Sequence | Strict generated-program replays | Live-model outcome |
| --- | ---: | --- | --- |
| RTX 4090 | 1 | 5/5 fail | Both backends pass scaled accuracy |
| RTX 4090 | 512 | 5/5 fail | GPU-stage timeout |
| RTX 5090 | 1 | 5/5 fail | Both backends pass scaled accuracy |
| RTX 5090 | 512 | 5/5 fail | GPU-stage timeout |

Generated and live programs may expose different outputs and select different kernels; their accuracy and timing
verdicts are separate. Failures include FP8 carrier differences and FP64 comparison failures. RTX 5090 prefill's
final output passes the FP64 allowance, but an intermediate scale peak fails, so its overall strict verdict fails.
No tolerance changed, and failed backends supply no valid comparison latency.

RTX 4090 uses run `20261013T104811Z`, source `134c5bd4a`; RTX 5090 uses run `20261013T111703Z`, source
`8af48e53c`. These are the latest FP8 observations, not reruns of the final compiler. Both use PyTorch 2.13.0 and
Transformers 5.14.1. Their timings remain raw evidence only.

## Source, systems and reproduction

Historical standalone runs use Python 3.12.3, PyTorch 2.13.0 and Triton 3.7.1 on Ubuntu 24.04 x86-64. Their
attestations cover 432 staged tracked files. All use private native binary SHA-256
`0fcb53578875cc28b56acbe9303d05f20114fcabc1e6778a5b8ae21c29ee44b9`, built from
`79ccb2cec069b7c2d106ad26577c1c4604935b31`, whose Rust sources match the historical standalone checkout.
Build-input hashes and compiler receipts are archived. Explicit import paths identify the source actually used;
installed-package lists alone do not establish it.

| GPU | Historical standalone run ID | CUDA wheel / NVCC | Driver | Final Norm-D run ID | Final common run ID |
| --- | --- | --- | --- | --- | --- |
| V100 SXM2 16GB | `20261014T044955Z` | 12.6 / 12.9.86 | 580.178.04 | `20261014T221221Z` | `20261014T215755Z` |
| A100 SXM4 40GB | `20261014T044923Z` | 13.0 / 13.0.88 | 580.178.04 | `20261014T220945Z` | `20261014T215837Z` |
| H100 80GB HBM3 | `20261014T044926Z` | 13.0 / 13.0.88 | 580.178.04 | `20261014T220521Z` | `20261014T215842Z` |
| RTX 4090 24GB | `20261014T044956Z` | 13.0 / 12.9.86 | 580.159.03 | `20261014T221119Z` | `20261014T215857Z` |
| RTX 5090 32GB | `20261014T044931Z` | 13.0 / 13.0.88 | 580.178.04 | `20261014T222451Z` | `20261014T221733Z` |

All five cards verify 433 staged source, recipe and runtime-build files against the current source
freeze and use the same private native binary identified above. Their software inventories confirm PyTorch 2.13.0,
Triton 3.7.1 and Transformers 5.14.1; the CUDA compiler and driver versions in the table also hold for these runs.
Explicit imports, both prior hashes, input hashes, package inventories and post-run artifact byte checks are
retained. After the measured freeze, formatting, import ordering and an explicit synchronous callback binding
preserve compiler behavior, as checked by normalized ASTs and 130 generated-source comparisons. Later CI repairs
update four test fixtures and freshen an unrelated DeepSeek recipe golden; compiler code, common-corpus inputs
and both priors remain byte-identical across those repairs. The DeepSeek change removes one stale timing while
retaining its schedule, and its four prior-reproduction slices pass with the shipped weights. These later changes
are recorded separately from the frozen fourteen-file refit input. The measured source remains the exact
reproduction target.

The final RTX 5090 common and Norm-D tasks ran after the controller failure, in separate services with a 12 GiB
host-memory cap and swap disabled. Their observed memory peaks were 5.4 and 1.3 GiB, respectively. All fifteen
repeats completed without new out-of-memory or GPU faults. These were the first attempts for those current rows.

Save the archive and check out the source for the chosen study: `8f391542c9ec78a06baa2ffad030369048c1c350` for
the twenty historical standalone rows, or `0add27dbb415f5336d04e9d11e19e1c8799d43dd` for the current Norm-D and
common runs. Prepare the recorded environment on the exact card. The recipe reuses it without installing packages.
Extract the archived native runtime and place it at the private cache path required by that frozen recipe; its hash
is checked before use. The supplied historical binary targets Ubuntu x86-64/Python 3.12. Another platform requires
a new build and explicit provenance.

Run `emmy bench experiments/cgo-2027/kernels` with `PYTHONPATH` set to the checkout, `--local` or an existing
host's `--ssh`, and exact `--filter 'deploy.gpu=GPU_NAME'`. Use `--filter study=isolated` for all four standalone
computations, add `--filter operator=norm-d` for the five-card Norm-D follow-up, or use `--filter study=common`
for the two whole-layer shapes. The complete declared target set is the denominator; do not select only favorable
operators. Common runs copy committed goldens without modifying them.

## Evidence limits and archive

This reproduction does not measure H200/B200, larger-layer supplements, convergence or budget ablations. It does
not reproduce the paper's old isolated-kernel bars or establish automatic LLM-plus-MCTS search within 30 minutes.
The [Neptune comparison](../compiler_neptune_emmy_pytorch_a100/RESULTS.md) and
[quantized-kernel study](../quantized_kernels_rtx5090/RESULTS.md) remain separate. Paper claims must use the
workloads, boundaries, precision regimes and correctness qualifications actually represented here.

The single `results.tar.gz` retains all twenty historical standalone records/bundles, four failed FP8
records/bundles and the separate H100 strict diagnostic byte-for-byte. Final common runs replace the earlier common
records as a complete set. Final Norm-D records, profiling and common-migration evidence are added separately.
The preliminary biased timings and superseded full platform snapshots are not part of the publication archive;
older committed snapshots remain in Git history.

Historical standalone evidence is under `isolated/<card>/<run>/`, with its recipe, source attestations and native
build records under `isolated/protocol/`, `isolated/provenance/` and `isolated/runtime/`. Current Norm-D and common
records are under `norm-d/<card>/<timestamp>/` and `common/<card>/<timestamp>/`. New profiling is under
`norm-d/profiling/`; migration evidence is under `common/migration/`. Each new record retains its artifact bundle,
five repeat statuses and every available receipt, input and source hashes, schedules, environment, database and
capture/accuracy checks. Missing receipts are identified above. Final source and artifact checks are under
`provenance/`, including `gcp-final/`, `recovery/`, `rtx5090-final/` and `post-ci-source/`. Prior fit evidence is under
`priors/refit/`, with its exact input source snapshot under `priors/source/`.

FP8 failures remain under `rtx4090/2026-10-13_10-48-11/` and `rtx5090/2026-10-13_11-17-03/`. Their bundles contain
`torch-compile/model.json` and strict `verification/repeat-0` through `repeat-4` evidence, sometimes extensionless
JSON or per-target directories. The H100 strict diagnostic remains under
`h100/2026-10-13_20-29-15/follow-up-diagnostics/raw-fixed/`.

Closed migration databases match their final copied bases. H100's nonempty transfer-era WAL/SHM are preserved
separately under `common/migration/h100/transient-transfer-sidecars/`, not beside the final database. Omitted empty
locks and audit sidecars are documented. No failed measurement or preparation attempt is removed for its verdict.

The archive contains 39 canonical terminal experiment records and 39 complete artifact bundles: 33 tasks
succeeded and six failed. The fifteen current tasks cover 75 attempted repeats, 74 JSON receipts and 71 clean
exits. The V100 prefill timeout has no JSON receipt; the three V100 Norm-D exits of 141 retain diagnostic receipts.
The two original recovered `running` records remain provenance and are not additional experiment tasks.

All 11,075 direct regular files were verified byte-for-byte against the staged source, including each bundle;
the 39 bundles contain 39,238 regular members. The root manifest records every other direct file's size and SHA-256.
An independent audit verifies the final measurements and evidence joins. All 86 retained historical members are
byte-identical; 44 superseded common-run members are replaced. Credential-prefix checks scanned 10,914 direct
text files and 37,645 nested text files without a match. Binary files, including the source snapshot pack, were
excluded from text scanning; their byte hashes remain recorded.

Archive size: **72,553,419 bytes**. SHA-256:
`9eb5c9e5740394f41092d1095beb4c7f959e72df73ab165a3adae3ef90657ec6`.
