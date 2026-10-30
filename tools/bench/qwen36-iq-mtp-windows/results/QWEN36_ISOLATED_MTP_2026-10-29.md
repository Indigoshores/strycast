# Qwen3.6 IQ4_XS: standalone MTP and source removal

Date: October 29, 2026. Windows 11, 128 GB Strix Halo, gfx1151.

The server now uses a **1.62 GB standalone MTP package**. The original **22.85 GB** Q4 model was deleted after replacement verification, reclaiming **21.23 GB net**. The IQ4_XS target remains unchanged.

## Files and launcher

- Target: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
- Standalone MTP: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`.
- Package size: **1,620,490,976 bytes**, approximately 1.51 GiB.
- Package SHA256: `3fff99797b9c9d5afbd3ff368efac1d348080875d87c38148bd0122724a7f6d9`.
- Deleted source: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, **22,853,663,008 bytes**.
- Net saving: **21,233,172,032 bytes**, approximately 19.77 GiB.

Start with the updated launcher:

```powershell
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\start-qwen36-iq-mtp2.exe'
```

The [server settings](QWEN36_IQ4_XS_MTP2_SERVER_SETTINGS_2026-10-29.md) and [machine-readable launch profile](iq-spec-mtp2-launch-profile.json) now select the standalone file. MTP remains `--speculative mtp --draft-tokens 2 --min-draft-tokens 1`; context 32768, two sessions, prefill chunk 2048, thinking off, temperature 0, seed 42, RAM checkpoint cache off. Address: `http://127.0.0.1:8081/v1`, model `gufo`.

The preserved fast engine remains `gufo-iq-prefill-fixed.exe`, SHA256 `72200a67f689b308b4aa120a8e843709cd47a50d90fb918e0e0b0fde9c85a3d1`, with TheRock 10.0.0. This extraction required no engine modification or rebuild.

## Package contents and compatibility

The GGUF contains **22 original tensors**: 20 private MTP tensors under `blk.40.*`, plus `token_embd.weight` and `output.weight`. No trunk tensors from layers 0 through 39 are included. Metadata and 32-byte alignment are preserved; raw metadata's 41 blocks include one MTP block, leaving 40 target blocks under the existing config parser.

The private MTP block occupies **528,857,088 bytes**. Each shared Q8_0 embedding/output matrix occupies **540,344,320 bytes**. The existing `QwenMtpWeights::LoadFromGguf` loader requires valid shared matrices before HIP model creation ties its GPU references to the IQ target. These two source matrices have different hashes, so both were retained as real weights. This explains why the compatible standalone package is 1.62 GB rather than just the 0.53 GB private block.

Every retained tensor was copied without requantization and verified against its source payload hash. The [extraction receipt](isolated-mtp-extraction.json) records the package and tensor verification. No placeholders or fabricated tensors were used.

## Runtime proof before deletion

The full source was first renamed to a temporary holding path, making its original path unavailable. All following tests used the standalone path, then stopped their owned servers:

- **Four natural-workload completions:** two prose and two Go coding runs, 256 generated tokens each. Every input/output hash, output token count, draft count and accepted count matched the original full-source MTP2 receipts.
- **Pooled decode: 72.88 tokens/s**, compared with the original full-source screen's 73.18. This is a packaging recheck, without a new matched AR run; the small timing difference does not establish a speed change.
- **824 drafted / 604 accepted tokens: 73.30% acceptance**, identical to the original counts. MTP was active.
- **Structured JSON, automatic tool-call/result and two-session checks: 10/10 pass**, with exact agreement against AR and positive MTP draft counters. Schema constraints specify types and keys, not expected answer values.

Evidence: [natural-workload summary](isolated-mtp-runtime-summary.json), [measured rows](isolated-mtp-runtime-rows.json), [JSON/tool/concurrency summary](isolated-mtp-json-summary.json), [JSON/tool/concurrency rows](isolated-mtp-json-rows.json).

The original source was removed only after those gates passed and its exact path, size and Windows file identity were checked. The deletion was nonrecursive and affected only that original file at its holding path. [Deletion receipt](isolated-mtp-deletion.json). A subsequent targeted filesystem check confirmed that both original and holding paths are absent; the standalone file remains at its expected size. No full-model backup was retained.

## Local tooling changes

All paths below are under `C:\Users\Butch\Documents\02_Apps\strycast-bench-tools`, outside the Strycast engine repository:

- `isolate-mtp.go` / `isolate-mtp.exe`: bounded GGUF extraction, metadata and payload verification, reversible source holding, and gated single-file retirement.
- `start-qwen36-iq-mtp2.go` / `.exe`: changed the draft model path and file-size guard to the standalone package. The engine digest and runtime arguments remain the tested profile.
- `spec-tools.go`: added `specIsolatedMTPVerify` for standalone-versus-original receipt comparison with the original source path unavailable.
- `main.go`: dispatches `isolated-mtp-verify`.
- `results/iq-spec-mtp2-launch-profile.json`: updated MTP path and package note.

The old launcher source/executable and profile were preserved with `source22gb` names as small historical artifacts. They refer to the deleted model and are not the current launcher. [Current artifact digests](isolated-mtp-artifacts.json).

No Strata server or second Flash-Next instance was started. [Final cleanup probe](isolated-mtp-final-probe.json) found no HTTP listener on ports 8080, 8081 or 8082.

This turn did not rerun the earlier 32K prompt ladder or broad task-quality evaluation. The [earlier speculative report](QWEN36_IQ4_XS_SPECULATIVE_RESULTS_2026-10-29.md) preserves those measurements and the original prompt-only JSON formatting failures. Thinking-on, nonzero-temperature sampling, arbitrary tools, sustained agent workloads and concurrent Strata residency remain outside the established evidence.
