# Qwen3.6 IQ4_XS speculative decode results — October 29, 2026

**Follow-up:** the current launcher uses a **1.62 GB standalone MTP package**, extracted without changing tensor bytes. The original 22.85 GB Q4 file was deleted after standalone runtime verification, saving **21.23 GB net**. Rechecked natural-workload decode was **72.88 tokens/s**, with identical outputs and draft counts; structured JSON/tools/two-session checks passed **10/10**. The measurements below retain their original full-source provenance. [Standalone report](QWEN36_ISOLATED_MTP_2026-10-29.md).

**MTP with two draft tokens is the best tested setting: 73.18 tokens/s pooled decode versus 57.19 without speculation, a 28.0% gain.** This original screen used the MTP head in the full Q4_K_XL file with the fast IQ4_XS target. It did not load a second complete 35B target onto the GPU.

All 29 speculative completions across the performance and functional checks match their plain-decode references exactly. This establishes the tested execution equivalence, not universal model quality. Prompt-only JSON formatting failed three cases in both modes; explicit JSON-schema output fixes those cases in the follow-up checks. Original failures remain recorded.

## Files and compatibility

- Target: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 17,730,509,792 bytes. Its earlier verified SHA256 is `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`. This file has no embedded MTP head.
- Original measurement source, now deleted: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` in the same folder, 22,853,663,008 bytes. Strycast loaded its private MTP tensors and shared the IQ target's embedding/output weights. Compatibility was established by initialization and matching generated outputs, not merely inferred from the filename. Current replacement: `Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`.
- DFlash2: downloaded `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` into the same models folder. 570,468,544 bytes; [download receipt](iq-dflash2-download-receipt.json). Repository `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`, pinned revision `1e26e3499d865b9d2fc55e019d65b3f0e8ab6846`, verified LFS SHA256 `69bc81f5344dfeb67bcfc09277dd9617500fca1823f72e6e08d5df92e73a8b44`.
- Engine: `strycast\build\windows-release\gufo-iq-prefill-fixed.exe`, SHA256 `72200a67f689b308b4aa120a8e843709cd47a50d90fb918e0e0b0fde9c85a3d1`. This is the preserved IQ-prefill-ON build with the stream initialization fix. Normal `gufo.exe` remains an option-OFF build.

The external draft interfaces and published DFlash2 file are documented in [Strycast's model guide](https://github.com/slimsami/strycast/blob/main/docs/models/qwen3.6-35b-a3b/README.md). Local source `src/models/qwen/hip/mtp/model.cpp` loads private head tensors and reuses the target embedding/output; no engine patch was needed for external-head MTP.

## Matched natural-workload screen

Windows 11, gfx1151 Strix Halo, TheRock 10.0.0. Same engine and IQ4_XS file in all modes. Two sessions preallocated, one active request at a time, context 32768, prefill chunk 2048, thinking off, temperature 0, seed 42, presence penalty 0, repeat penalty 1, RAM checkpoint cache off, and no prompt reuse. Each mode starts a fresh server and completes one excluded warmup, then two repetitions of a prose task and a Go coding task with a 256-output-token cap. All measured responses produced 256 tokens.

Rates below pool generated tokens over native decode milliseconds. These are short screening runs, not confidence intervals or a universal ranking:

- **Plain decode:** 57.19 tokens/s pooled; prose 58.02; code 56.38.
- **MTP2:** 73.18 pooled (**+28.0%**); prose 69.18 (**+19.2%**); code 77.67 (**+37.8%**). Draft acceptance 73.3% pooled: 62.4% prose and 86.6% code.
- **DFlash2, fixed three drafts:** 61.80 pooled (**+8.1%**); prose 56.11 (**-3.3%**); code 68.78 (**+22.0%**). Acceptance 55.4% pooled.
- **DFlash2, fixed seven drafts:** 45.05 pooled (**-21.2%**); prose 38.40; code 54.50. Acceptance 31.6% pooled. Do not select this setting from this screen.

Evidence: [summary](iq-spec-natural-screen-summary.json), [all 16 measured rows](iq-spec-natural-screen-rows.json), [independent receipt audit](iq-spec-natural-audit.json). Matching raw responses, launch receipts, and server logs use the `iq-spec-natural-<mode>` prefix. Audit confirms identical input hashes across modes, zero cached tokens, and zero output mismatches. There was no competing build or benchmark during the screen.

MTP1, larger MTP chains, adaptive DFlash2, other sampling settings, and other workloads were not swept. MTP2 is the best among the four tested profiles.

## Functional gate: short, long, tools and two sessions

[Functional rows](iq-spec-validation-v2-rows.json), [summary](iq-spec-validation-v2-summary.json), and [saved known-answer fixtures](iq-spec-validation-v2-fixtures.json) cover 12 responses per mode, plain decode versus MTP2:

- Arithmetic plus strict JSON on a 43-token prompt: correct.
- Three-code retrieval with controls at beginning, middle and end: correct at **98, 2,051, 8,189, 16,373 and 31,997 actual input tokens**. The nominal 128-token fixture tokenizes to 98, and the nominal 32K fixture is kept below the 32768-token context/output limit.
- Multi-turn preference update: returns the correct latest value, green, but wraps the JSON in Markdown in both modes.
- Automatic `lookup_invoice` selection: correct function and schema-valid `invoice_id` in both modes. This uses `tool_choice: auto`, not a forced function.
- Tool-result continuation: returns the correct invoice id, total and status from a local mock, but wraps the JSON in Markdown in both modes. No external invoice service or other real tool was invoked.
- Tool-not-needed control: answers correctly without a tool call.
- Two simultaneous requests with distinct nonces and all integers 1 through 32: correct data, no cross-session contamination. One response wraps its JSON in Markdown in both modes.

**Exactness gate PASS: all 24 responses match the corresponding AR response. Prompt-only pure-JSON gate FAIL: 18/24 pass, with three formatting failures per mode.** The known answers inside the fenced JSON are correct, but fences still violate the requested pure-JSON interface. We did not strip fences in the checker or label that gate passed.

At the 31,997-token recall case, native decode reports 49.1 tokens/s AR versus 68.1 MTP2, with the same 43 output tokens. Prefill reports approximately 2481 versus 2434 tokens/s. These short-answer rates are workload-specific; use the 256-token screen above for the primary decode comparison.

## Structured JSON follow-up

[Structured rows](iq-spec-json-schema-rows.json), [summary](iq-spec-json-schema-summary.json): reran the affected patterns using `response_format` of type `json_schema`, with required keys, types, and no additional properties. Schema constraints specify **no expected answer values**.

**10/10 responses pass pure JSON, known-answer correctness and exact AR/MTP agreement.** This includes the full automatic tool-call/result workflow, the multi-turn preference update, and two simultaneous nonce/sequence responses. The tool call itself remains automatic; the final data response uses an explicit schema. Native draft counters confirm MTP remains active under the schema constraints.

Example request option for a simple JSON response:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "preference",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"color": {"type": "string"}},
        "required": ["color"],
        "additionalProperties": false
      }
    }
  }
}
```

This is a tested client configuration for machine-readable JSON. It does not turn the original prompt-only formatting failures into passes, or guarantee arbitrary tool/schema behavior.

## Concurrency and memory boundaries

The single-active-request 28% gain does not carry over unchanged to two active requests. In the unconstrained two-request test, per-request decode was 43.5 / 44.3 tokens/s AR and 44.3 / 46.2 MTP2. In the structured two-request test, it was 43.8 / 44.9 AR and 47.8 / 47.8 MTP2. These are short correctness exercises, not a sustained concurrent throughput benchmark.

Driver-reported GPU device use at server startup, with two sessions and context 32768:

- AR: 23,449 MiB, approximately 22.90 GiB.
- MTP2: 24,551 MiB, approximately 23.98 GiB (**+1,102 MiB / 1.08 GiB**).
- DFlash2: 25,492 MiB, approximately 24.89 GiB (**+2,043 MiB / 2.00 GiB**).

These are startup device-memory snapshots, not peak host RAM measurements. Strata was not launched or loaded alongside this test, so dual-model residency and bandwidth contention remain untested.

## Tested profile and launcher

The server profile is:

```text
--speculative mtp
--mtp-model <existing Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf>
--draft-tokens 2
--min-draft-tokens 1
--sessions 2
--context 32768
--prefill-chunk 2048
--cache-ram-bytes off
--think off
--temperature 0
```

HTTP serving requires the MTP minimum to remain **1**; an initial attempt with 2 was rejected by the server's configuration validation before inference. The fixed proposal chain is capped by `--draft-tokens 2`. An initial chat smoke also required an explicit served model alias; the working profile exposes `gufo`. These were setup corrections, not generated-output failures.

A native Go launcher is compiled at:

`C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\start-qwen36-iq-mtp2.exe`

Run it from PowerShell:

```powershell
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\start-qwen36-iq-mtp2.exe'
```

It checks the measured engine digest and known model sizes, preserves an existing listener on port 8081, pins the tested TheRock/DLL paths, and starts only this IQ target with its MTP head. Client URL: **`http://127.0.0.1:8081/v1`**, model **`gufo`**. Ctrl+C stops its owned child. `--print-config` displays the profile without starting a server. The launcher was compiled and its configuration inspected; the equivalent server profile was exercised by the benchmark/validation harness. It was not left running as a persistent deployment.

## Evidence limits and repository state

This task added local test/launch tooling and the DFlash2 download. It did not change engine code or global defaults. Strycast remains on pushed commit `04e94a978c0eeddb2e25fad7a5d1eaf82bae7b5f`; the earlier full-target and kernel qualifications remain separate evidence, not newly rerun claims.

Across this task: 16 measured screen responses, 24 general functional responses, and 10 structured follow-up responses. All **29 speculative** responses match their AR references. The six prompt-only formatting failures remain visible in the original failed gate; the explicit-schema follow-up passes. This is a bounded smoke/compatibility screen, not a broad accuracy benchmark.

Thinking-on workloads, nonzero-temperature sampling, arbitrary schemas/tools, sustained agent runs, real external services, long tool-filled contexts, a Q4-versus-IQ accuracy comparison, and simultaneous residency with Flash-Next remain unqualified. No Flash-Next process was launched and no existing model file was modified.

[Saved launcher profile](iq-spec-mtp2-launch-profile.json) records the executable,
arguments and SDK. [Artifact digests](iq-spec-artifact-digests.json) bind the local
tools, source and result summaries. The [final probe](iq-spec-final-probe.json)
confirms all owned benchmark servers stopped and ports 8080, 8081 and 8082 idle.
