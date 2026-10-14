# Common-kernel study: latest results

This report retains the latest accepted Qwen3-0.6B common-corpus run on each of five GPUs, plus the final RTX
FP8 runs from the same reproduction. All runs are from 2026-10-13 UTC. Consolidation changes no measurement.
The single [results archive](results.tar.gz) contains the corresponding records, complete command artifacts,
correctness checks and source provenance. The recipe and ten common-corpus goldens remain as reproduction inputs.

## Common FP16 corpus

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

Emmy is faster in seven of ten observations. H100 prefill remains 5.39% slower and RTX 4090 prefill 3.24% slower.
RTX 5090 sequence one is within 0.5% of PyTorch. These measurements establish neither a broad kernel win nor stable
margins for near ties. Sequence one attends to one key; it does not measure decode with a populated KV cache.

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

## Dynamic-FP8 checkpoint results

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

## Source and system provenance

Each GPU uses one physical card. The latest accepted runs retain their original source revisions; rebasing the
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

`results.tar.gz` is the only results archive for this study. It holds the 14 original terminal experiment records
and their 14 complete, byte-identical command artifact bundles: ten successful common rows and four failed FP8
rows. It also retains the final benchmark logs and source attestations. Its 76 direct files include a manifest
recording original archive hashes, original member paths, sizes and SHA-256 hashes for every retained payload.

| GPU | Directory inside the archive |
| --- | --- |
| V100 | `v100/2026-10-13_06-11-46/` |
| A100 | `a100/2026-10-13_17-50-52/` |
| H100 | `h100/2026-10-13_20-29-15/` |
| RTX 4090 | `rtx4090/2026-10-13_10-48-11/` |
| RTX 5090 | `rtx5090/2026-10-13_11-17-03/` |

Each experiment record has a matching `_artifacts.tar.gz` beside it. Inside that command bundle,
`torch-compile/model.json` holds the actual-model comparison; `verification/repeat-0` through `repeat-4` contain
strict replays, sometimes as extensionless JSON files and sometimes as directories of target JSON files. Logs,
status files, package freezes, working goldens, tuning databases and cubins retain the execution context.

The separate H100 strict live diagnostic is under `h100/2026-10-13_20-29-15/follow-up-diagnostics/raw-fixed/`.
Its source is distinct from the final recipe run, as described above. No superseded platform snapshots, previous
results archives or exploratory optimization rounds are nested in this archive. They remain available in Git
history before this consolidation.

Archive size: 8,174,395 bytes. SHA-256:
`92a277e4c1f39f6da72d38ca7867a6c6f42784783d85fa97d6b07fc3adb72358`.
