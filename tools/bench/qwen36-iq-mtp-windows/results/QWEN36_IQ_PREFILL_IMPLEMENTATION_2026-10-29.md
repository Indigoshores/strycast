# Qwen3.6 IQ prefill implementation — 2026-10-29

**Later follow-up on October 29:** the serial/concurrent verification blocker
below has been resolved by fixing MoE scratch initialization stream ordering.
The full IQ4_XS model target, including an added exact snapshot replay check,
now passes. See [verification fix and current evidence](QWEN36_IQ_PREFILL_VERIFICATION_FIX_2026-10-29.md).
The original kernel implementation was committed and pushed as
[`d4bbdfa`](https://github.com/Indigoshores/strycast/commit/d4bbdfa2ba592de9232d11ce7b0c8b928795c42c).
The original measurements and failures below are retained as historical evidence;
their readiness and Git-state statements describe the earlier run.

Implemented and measured in Strycast as an **experimental, default-off** path. It improves this device's IQ4_XS prefill by about 12 times at 2K tokens. **Full-model verification parity fails, so this is not production-qualified.** Normal `gufo.exe` and `gufo-server.exe` were rebuilt with the option OFF. Strata source, settings, and champion binaries were not changed.

## Implementation and ownership

Repository: `C:\Users\Butch\Documents\02_Apps\strycast`.
Branch: `codex/qwen36-iq-prefill-20261029`.
Base: `03ba946d37bbfdbf80af56740aa10c27c5c0fd67`.
Changes remain uncommitted for review.

- Added IQ3_S and IQ4_XS support to the routed FP16 WMMA MoE kernels, including direct projection and paired gate/up paths. The requested IQ4_XS model also contains IQ3_S experts; implementing both is necessary.
- Decodes packed weights into tile-local signed codes and scales using existing IQ decoding primitives. No full-model FP16 weight copy is allocated.
- Added opt-in `GUFO_QWEN_IQ_PREFILL` in `src/models/qwen/CMakeLists.txt`, default OFF. Only the Qwen prefill dispatcher is opted in; GPU decode implementation remains unchanged.
- Fixed missing IQ3_S/IQ4_XS dispatch in CPU generic `Dequantize` and `Dot`. Canonical implementations already existed, but the generic dispatcher aborted when the full-model CPU reference encountered IQ tensors.
- Added independent CPU dequantization/FP64-dot oracle coverage, generic CPU dispatch checks, routed and paired IQ tests, ragged tiles, empty experts, and batch-invariance checks.
- Updated the Qwen3.6 model README, quality notes, benchmark notes, and experiment history.

Exact source changes are preserved in [iq-prefill-implementation.patch](iq-prefill-implementation.patch). Binary and patch digests are in [iq-prefill-artifact-digests.json](iq-prefill-artifact-digests.json).

## Model and environment

Model: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
File size: 17,730,509,792 bytes (17.73 decimal GB).
SHA256: `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`.
733 tensors, no MTP head. Expert tensor mix includes 80 IQ3_S gate/up tensors, 37 IQ4_XS down tensors, and 3 Q6_K down tensors.

Windows, Strix Halo gfx1151, native TheRock 10 SDK. Firmware reports 128 GiB installed; Windows exposes 31.65 GiB as CPU-visible physical memory. The final probe reports 22.39 GiB available CPU-visible RAM. Installed memory and CPU-visible RAM are distinct measurements; neither is a measurement of free GPU heap capacity.

The matched control and candidate use the same compiler, dependencies, RelWithDebInfo configuration, and `BUILD_TESTING=ON`. This is a controlled comparison, not certification of the ordinary published release build. Models ran sequentially without a concurrent benchmark or build.

## Final matched performance

Native command for each binary:

```text
bench --model <same IQ4_XS GGUF> --speculative off -p 512,2048 -n 128 -r 2
```

Automatic warmups precede measured repetitions. Load time is excluded. The tg128 row is the separate native zero-depth decode test.

- pp512: control **290.38 ± 2.85 tokens/s**; candidate **2,198.64 ± 28.93 tokens/s** (7.57 times).
- pp2048: control **266.05 ± 0.37 tokens/s**; candidate **3,192.83 ± 4.60 tokens/s** (12.00 times).
- tg128: control **56.69 ± 0.17 tokens/s**; candidate **57.84 ± 0.08 tokens/s**. No decode optimization is claimed from this small difference.

Raw logs: [control native](iq-prefill-control-native-final.log), [candidate native](iq-prefill-candidate-native-final.log).

Matched HTTP requests used one session, context 8192, prefill chunk 2048, speculative decoding OFF, identical prompt hashes, 1,950 input tokens, 128 output tokens, temperature zero, seed 1234, and no cached prompt tokens. One warmup was excluded; two responses per arm were measured. SDK/DLL paths were pinned by the launcher; other environment settings were inherited equally. The receipt's generic llama-specific environment metadata is not evidence that those options affect Strycast.

- HTTP prefill: **258.66 → 3,105.22 tokens/s** (12.01 times).
- HTTP decode: **56.54 → 56.87 tokens/s**, essentially unchanged.
- Whole-request mean wall time: **9,837.02 → 2,900.89 ms**, 70.5% lower.

Evidence: [comparison and protocol checks](iq-prefill-http-comparison.json), [control HTTP measurements](iq-prefill-control-http-string-bench.json), [candidate HTTP measurements](iq-prefill-candidate-http-string-bench.json). Matching launch JSON, server logs, and response-0 through response-2 JSON are retained alongside these files under the same arm prefixes.

## Correctness evidence and qualification blocker

**Passed:**

- [Generic CPU GEMM parity](iq-prefill-cpu-dispatch.log): existing supported formats pass.
- [MoE operator suite](iq-prefill-moe-ops-final.log): original formats and added IQ direct/paired cases pass without relaxing existing tolerances.
- Independent CPU oracle's worst normalized routed error: IQ3_S **0.000451561**, IQ4_XS **0.000438793**. These are approximately 0.0452% and 0.0439%; they are numerical errors, not task-quality scores.
- IQ paired tests remain below the existing 0.005 threshold; routed tests remain below 0.01. Tests cover 129 output rows, K=512, an empty expert, both activation layouts, multiple token counts, and tile widths 16/48/64/128.
- Existing decoder row-versus-batch invariance passes bit for bit across all six tested formats.
- Model load, native inference, and single-session HTTP inference succeed. The full-model target's two 12-position CPU/GPU greedy fixtures pass, as do prefill/decode greedy and long-prompt chunk greedy checks.
- Formatting and `git diff --check` pass.

**Failed; do not promote:**

- [Candidate full-model target](iq-prefill-model-target-final.log): serial eight-row verification differs from sequential decode in nearly all logits, with maximum differences approximately 0.09–0.29 per row. The existing exact assertion fails. Concurrent verification is not reached in this run.
- [OFF control full-model target](iq-prefill-model-target-control-final.log): the same serial verification passes exactly, but concurrent verification fails at session 0, row 0.
- The candidate's serial failure must not be described as reproduced by the control. The control has a separate concurrent failure. Changed prefill states may expose a downstream arithmetic/state dependency, but its exact cause is **not established**.
- Long-prompt one-chunk versus split-chunk tests agree on greedy output but are not logit-identical: candidate max absolute difference 2.9801, RMSE 0.471828; control 2.40615 and 0.433938.

An earlier full-model abort was diagnosed independently: [native fault trace](iq-prefill-control-target-debug.log) reaches `abort -> quant::Dot -> TensorGEMV -> MoeForward -> ForwardLayer -> ForwardModel -> CheckCpuGpuLogitParity`. The generic CPU IQ dispatcher fix removes that abort; it does not resolve the verification failures above. LLDB startup also failed on a missing interpreter DLL; the native Go/Windows debugger supplied the trace instead. No assertions or tolerances were weakened.

No long-context quality evaluation, speculative acceptance certification, multi-session certification, dual-model memory/residency test, Linux build, or task-quality benchmark was completed. This work does not establish production readiness.

## Reproduce the experimental build

The local Go launcher applies the existing VS/TheRock environment and logs builds:

```powershell
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\model-matched.exe' build -S . -B build/windows-release -DBUILD_TESTING=ON -DGUFO_QWEN_IQ_PREFILL=ON
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\model-matched.exe' build --build build/windows-release --target gufo gufo-server qwen35ba3b_moe_ops_test qwen35ba3b_target_test ggml_gemm_parity_test --parallel 1
```

Run these from the Strycast repository. Enabling this flag is an experimental build choice. Current normal build was restored to OFF after preserving the candidate binaries:

- `build/windows-release/gufo-iq-prefill-candidate.exe`
- `build/windows-release/gufo-server-iq-prefill-candidate.exe`
- Original pre-edit binaries are retained with `-iq-prefill-baseline.exe` names.
- Normal `gufo.exe` / `gufo-server.exe` contain the CPU reference fix but use the OFF prefill dispatcher.

The next required engineering step is to trace the first diverging verification operator from identical saved states, then repair the arithmetic/state contract and rerun the unchanged serial and concurrent checks before any default change. Passing isolated kernels is insufficient. Prior Strycast experiment history already reports that MoE HIP graph replay did not improve decode; another graph rewrite is not justified by this prefill result.

All owned benchmark servers were stopped. [Final probe](iq-prefill-implementation-final-probe.json) found no HTTP listener on ports 8080, 8081, or 8082. No model file was modified, no Strata server was launched, and nothing was committed or published.
