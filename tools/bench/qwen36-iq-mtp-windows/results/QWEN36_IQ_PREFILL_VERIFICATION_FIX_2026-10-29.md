# Qwen3.6 IQ prefill verification fix — 2026-10-29

**Later speculative-decode follow-up:** the IQ4_XS target can use the MTP head
from the existing Q4_K_XL file. A bounded screen measured 57.19 -> 73.18 tok/s
pooled decode with two drafts, and subsequent short/long/tool/concurrent checks
matched AR. Prompt-only pure JSON still needs explicit schema constraints in
the tested cases. See [speculative results and launcher](QWEN36_IQ4_XS_SPECULATIVE_RESULTS_2026-10-29.md).

The IQ4_XS serial and concurrent verification blocker is resolved on this Windows/TheRock device. The fast path remains an explicit build opt-in. This is a correctness fix to scratch initialization, with no change to IQ kernel arithmetic or test tolerances.

## Diagnosis and upstream search

Strycast upstream `main` matched our original base `03ba946d37bbfdbf80af56740aa10c27c5c0fd67` when checked. GitHub issue/PR and file-history searches found relevant prior work, but no ready-made IQ fix:

- [Strycast GDN verification arithmetic fix](https://github.com/slimsami/strycast/commit/196c9c2e77c418295833cd37a9673c83c92b7da3) aligns F32/BF16 control projections with scalar decode. Already present locally.
- [Strycast PR 7](https://github.com/slimsami/strycast/pull/7) qualifies K-quant MMQ decode and exact verification, not this IQ checkpoint.
- [Strycast PR 13](https://github.com/slimsami/strycast/pull/13) adds the fused router, its zero-initialized completion counter, and fused MoE residual normalization. Already present locally. The scratch initialization in that path was the relevant source area.
- [Gufo per-stream MMQ scratch fix](https://github.com/slimsami/strycast/commit/68ff41a86f1e76bfd61ca85e47722ae78c17f05e) separates MMQ pools by stream. Already present locally; replacing its pool or reverting it was unnecessary.
- [Gufo Windows first-use setup issue 745](https://github.com/gufo-org/gufo/issues/745) discusses a latency problem with unchanged output. It does not supply a fix for the data overwrite diagnosed here.

Native gograph navigation and targeted source reads traced the failing checks through `ForwardVerificationChunk`, `ForwardVerificationBatch`, `ExecuteDecodeStep`, `ExecuteMoeDecodeStep`, the MMQ down projection, and `QwenGpuArena` initialization. The graph's C++ call coverage is incomplete; graph output was used for navigation, with source and runtime traces supplying the decisive evidence.

## Root cause and controlled evidence

`QwenGpuArena` creates an inference stream using `hipStreamNonBlocking`, but its constructor initialized the entire MoE scratch allocation with `hipMemset` on the default stream. That clear was not ordered before work on the inference stream. A later MMQ pool allocation calls `hipDeviceSynchronize`, allowing the pending clear to erase scratch that already contains live results.

This was not established by guessing from logit differences:

1. A focused probe prefills once, saves a snapshot, decodes eight rows, restores the snapshot, and decodes the same rows again. The old code fails even this sequential replay, independent of batched verification.
2. Captured hidden outputs agree through layer 33 and first differ at layer 34 on the cold IQ route.
3. At layer 34, input hidden state, normalization, QKV/gate/control projections, convolution and recurrent input state, SSM output, and mixer output match before and after snapshot restore.
4. Router logits, selected expert ids, shared-expert output and expert activations are populated before the first Q6_K down projection. Tracing inside that MMQ adapter finds the router intact at entry and all zero immediately after pool acquisition, before quantization or the matrix kernel runs. The buffers do not overlap the pool allocation.
5. An experimental control that drains device initialization immediately after each executor is created makes snapshot replay, serial verification and concurrent verification all exact. That device-wide synchronization workaround was removed.
6. Ordering the clear on the arena stream produces the same passing results without the workaround or temporary host/kernel tracing.

Raw failure/diagnostic evidence is retained in [serial probe](iq-parity-serial-probe.log), [late-layer probe](iq-parity-serial-late-probe.log), [operator trace](iq-parity-serial-operator-trace.log), [MMQ stage trace](iq-parity-mmq-stage-trace.log), [serial initialization-sync control](iq-parity-serial-sync-control.log), and [concurrent initialization-sync control](iq-parity-concurrent-sync-control.log). Binary tensor captures are retained under `iq-parity-trace`, `iq-parity-trace-detail`, and `iq-parity-trace-mmq` beside these logs.

AMD documents the stream argument of [`hipMemsetAsync`](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html) as associating the operation with a selected stream. The local source/runtime evidence above establishes this particular race; no claim of a new driver defect is needed.

## Surgical change and regression protection

In `src/models/qwen/hip/arena.cpp`, replace:

```cpp
HIP_CHECK(hipMemset(d_moe_scratch, 0, moe_bytes));
```

with:

```cpp
HIP_CHECK(hipMemsetAsync(d_moe_scratch, 0, moe_bytes, stream));
```

The full scratch clear is preserved, including the fused router's zeroed counter. Same-stream ordering places it before all prefill/decode work. The fix applies whether `GUFO_QWEN_IQ_PREFILL` is ON or OFF.

The existing target test now also requires eight sequential decode rows to reproduce bit for bit after snapshot restore, before it runs its original exact serial and concurrent checks. No assertion or tolerance was weakened. A focused manual diagnostic executable, `qwen35ba3b_verification_probe MODEL serial|concurrent`, captures layer outputs and compares the same contracts without running the CPU-reference portion of the full target. Temporary production tracing and the experimental global-sync workaround were removed.

## Validation

Same requested model: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 17,730,509,792 bytes. Previously verified SHA256: `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`; the model file was not modified or downloaded again.

Windows 11, gfx1151 Strix Halo, TheRock 10.0.0, RelWithDebInfo, `BUILD_TESTING=ON`.

- [Fixed option-on full model target](iq-parity-fixed-full-target.log): **PASS**. Both CPU/GPU greedy fixtures, prefill/decode greedy check, 1100-token chunk greedy check, added exact snapshot replay, eight exact serial verification rows, and three concurrent sessions / 23 exact rows pass.
- [Fixed serial probe](iq-parity-fixed-serial-probe.log): **PASS**, zero changed replay or verification logits.
- [Fixed concurrent probe](iq-parity-fixed-concurrent-probe.log): **PASS**, zero changed logits across 23 rows.
- [Fixed MoE operator suite](iq-parity-fixed-moe-ops.log): **PASS**, including IQ oracle checks and all old formats. IQ oracle normalized errors remain 0.000451561 / 0.000438793; decode batch-invariance differences remain zero.
- [UD-Q4_K_XL concurrent model probe](iq-parity-fixed-q4-concurrent-probe.log): **PASS**, all 23 verification rows unchanged across the three sessions. This is an additional K-quant regression check, not a full Q4 target or quality evaluation.
- Formatting and `git diff --check`: **PASS**.

The [fixed option-off full target](iq-parity-fixed-control-full-target.log) also **PASSES**, including snapshot replay, all eight serial rows, and all 23 concurrent rows. Its long-chunk comparison reports max absolute difference 1.55858 and RMSE 0.278926 with greedy agreement. Thus both originally failing configurations pass after the same stream-ordering fix.

Long-prompt single-chunk versus split-chunk output is still not logit-identical. The fixed option-on run reports max absolute difference **1.2138**, RMSE **0.219402**, with the tested greedy prediction preserved. This prefill comparison has a different numerical contract from decode/verification: different FP16/BF16 chunk arithmetic is allowed by the existing test, whereas verification must be exact. It is not a newly waived assertion or proof of broad task quality.

## Retained performance

[Isolated native measurement](iq-parity-fixed-native-isolated.log): `bench --speculative off -p 512,2048 -n 128 -r 2`, same model, automatic shape warmups, load excluded, no competing tests/builds:

- pp512: **2,186.91 +/- 26.99 tokens/s**.
- pp2048: **3,218.73 +/- 12.37 tokens/s**.
- Separate zero-depth tg128: **59.01 +/- 0.09 tokens/s**.

The first follow-up native screen overlapped the tail of the operator suite and is retained only as discarded evidence in `iq-parity-fixed-native.log`. It is not used above. The original pre-kernel control measured 266.05 tokens/s at pp2048; this remains about a 12-times prefill improvement, not a newly matched control/candidate performance comparison for this correctness-only patch.

[Fixed HTTP measurements](iq-parity-fixed-http-string-bench.json): identical 1,950-token prompt, 128 generated tokens, no prompt cache reuse, one excluded warmup and two measured requests, one session, context 8192, prefill chunk 2048, speculation off:

- Measured prefill: 3,121.94 and 3,032.71 tokens/s; mean approximately **3,077.33 tokens/s**.
- Measured decode: 58.43 and 56.99 tokens/s; mean approximately **57.71 tokens/s**.

The HTTP warmup is excluded (2,344.98 prefill / 58.00 decode). Matching launch, response and server log artifacts share the `iq-parity-fixed-http` prefix. Successful HTTP requests are a serving smoke check, not an accuracy benchmark.

## Local deliverables and limits

Preserved fixed option-on binaries:

- `strycast\build\windows-release\gufo-iq-prefill-fixed.exe`
- `strycast\build\windows-release\gufo-server-iq-prefill-fixed.exe`
- `strycast\build\windows-release\qwen35ba3b_target_iq_fixed.exe`

Earlier baseline/candidate binaries remain preserved. Normal `gufo.exe` and `gufo-server.exe` are rebuilt with the stream-order fix and IQ prefill OFF; the fast option is not silently enabled. Strata source, launcher settings and champion binaries were not changed.

These checks resolve the reported exactness failures for this Windows fixture. They do not certify task accuracy, long-context quality, sustained multi-user service, MTP acceptance (this checkpoint has no MTP head), dual-model residency, or Linux. No Linux/CUDA/SYCL build was performed. No new upstream issue, PR, or comment was posted.

## Delivery and cleanup

Committed and pushed to the existing fork branch `codex/qwen36-iq-prefill-20261029` as
[`04e94a9`](https://github.com/Indigoshores/strycast/commit/04e94a9).
The exact staged changes are also preserved in [source patch](iq-prefill-verification-fix.patch).
[Artifact digests](iq-parity-fix-artifact-digests.json) bind the patch, fixed option-on and normal option-off binaries, and full-target receipts. The fixed fast CLI/server SHA256 is `72200a67f689b308b4aa120a8e843709cd47a50d90fb918e0e0b0fde9c85a3d1`; the normal option-off CLI/server SHA256 is `0b57df75bdf6a8fd87fbdada54bb48cb6750b22a0cad57c04f556d979b44f2e3`.

The owned HTTP benchmark server was stopped. [Final probe](iq-parity-fix-final-probe.json) found ports 8080, 8081, and 8082 idle. Generated `.gograph/` cache remains untracked; no model weights, SDK files, benchmark binaries, or local raw logs were committed.
