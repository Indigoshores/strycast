# Qwen3.6 IQ4_XS + MTP2 server handoff

Date: October 29, 2026. Device: Windows 11, 128 GB Strix Halo, gfx1151.

**Standalone MTP update:** the launcher now uses a losslessly extracted **1.62 GB MTP package**. The original 22.85 GB Q4 file has been deleted, reclaiming **21.23 GB net**. With that source unavailable, four natural-workload completions and their draft/acceptance counts matched the original exactly; pooled decode was **72.88 tokens/s**. JSON, tool-result and two-session follow-up checks passed **10/10**. [Isolation report and receipts](QWEN36_ISOLATED_MTP_2026-10-29.md).

**Selected tested profile:** IQ4_XS with the routed IQ prefill build and two-token MTP drafting. Natural-workload decode increased from **57.19 to 73.18 tokens/s pooled**, with identical greedy output. Native pp2048 reached **3,218.73 tokens/s**. Those figures are from different workloads; they are not one paired prefill/decode measurement.

## 1. Model, engine and MTP used

- Target: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
- Target file size: **17,730,509,792 bytes / 17.73 decimal GB**. This describes disk size, not total server RAM use.
- Target SHA256: `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`.
- Standalone MTP: `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`.
- MTP package size: **1,620,490,976 bytes / 1.62 decimal GB**. Only this package is required alongside the IQ4_XS target; the full Q4 source is deleted.
- MTP package SHA256: `3fff99797b9c9d5afbd3ff368efac1d348080875d87c38148bd0122724a7f6d9`.
- Engine: `C:\Users\Butch\Documents\02_Apps\strycast\build\windows-release\gufo-iq-prefill-fixed.exe`.
- Engine SHA256: `72200a67f689b308b4aa120a8e843709cd47a50d90fb918e0e0b0fde9c85a3d1`.
- Runtime: `C:\Users\Butch\Documents\02_Apps\strycast-deps\therock-10.0.0`.

**MTP means multi-token prediction.** The IQ4_XS file has no embedded MTP head. The standalone GGUF contains the 20 original layer-40 MTP tensors plus the source embedding and output tensors required by the existing loader. Strycast then shares the IQ target's embedding and output weights on the GPU. The package contains no target trunk layers 0–39. No tensor was requantized and no engine change was needed.

The tested settings are `--speculative mtp --draft-tokens 2 --min-draft-tokens 1`. HTTP serving rejects a minimum of 2; leave the minimum at 1 and use the maximum to cap the chain. Overall measured MTP draft acceptance was **73.3%**, with 62.4% on prose and 86.6% on code.

DFlash2 was also tested using `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf`, **570,468,544 bytes**, from `aminya/Qwen3.6-35B-A3B-DFlash2-GGUF` at revision `1e26e3499d865b9d2fc55e019d65b3f0e8ab6846`. Verified SHA256: `69bc81f5344dfeb67bcfc09277dd9617500fca1823f72e6e08d5df92e73a8b44`. It remains in the models folder, but MTP2 is the selected profile.

## 2. Start the server

Use the native launcher, which pins the tested SDK, checks the engine digest and model sizes, and preserves an existing listener on port 8081:

```powershell
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\start-qwen36-iq-mtp2.exe'
```

- OpenAI-compatible base URL: **http://127.0.0.1:8081/v1**
- API model name: **gufo**
- Two sessions; 32,768-token context per session.
- Ctrl+C stops the launcher's owned server.
- Preview without launching: add `--print-config`.
- [Machine-readable launch profile](iq-spec-mtp2-launch-profile.json).

The launcher was compiled and its configuration checked. The equivalent server arguments were exercised by the benchmark and functional harness. The benchmark servers were stopped; the launcher was not left deployed.

### Equivalent direct runtime command

These are manual runtime commands, with the same environment and arguments used by the launcher:

```powershell
$engineRoot = 'C:\Users\Butch\Documents\02_Apps\strycast\build\windows-release'
$sdkRoot = 'C:\Users\Butch\Documents\02_Apps\strycast-deps\therock-10.0.0'
$target = 'C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
$mtp = 'C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf'

$env:PATH = "$engineRoot;$engineRoot\vcpkg_installed\x64-windows-release\bin;$sdkRoot\bin;$env:PATH"
$env:HIP_PATH = $sdkRoot
$env:ROCBLAS_TENSILE_LIBPATH = "$sdkRoot\bin\rocblas\library"
$env:HIPBLASLT_TENSILE_LIBPATH = "$sdkRoot\bin\hipblaslt\library"

$serverArgs = @(
    'serve', '--host', '127.0.0.1', '--port', '8081', 'llm',
    '--model', $target,
    '--served-model-name', 'gufo',
    '--sessions', '2', '--context', '32768',
    '--prefill-chunk', '2048', '--cache-ram-bytes', 'off',
    '--think', 'off', '--temperature', '0', '--seed', '42',
    '--presence-penalty', '0', '--repeat-penalty', '1',
    '--speculative', 'mtp', '--mtp-model', $mtp,
    '--draft-tokens', '2', '--min-draft-tokens', '1'
)
& "$engineRoot\gufo-iq-prefill-fixed.exe" @serverArgs
```

The ordinary `gufo.exe` / `gufo-server.exe` were restored to an IQ-prefill-OFF build. Use the preserved `gufo-iq-prefill-fixed.exe` for the measured fast path, or rebuild explicitly with the opt-in flag. No Strata launcher, binary or optimization setting was changed.

## 3. Prefill and decode speeds

### Native before/after, speculation OFF

Same IQ checkpoint and native recipe `bench -p 512,2048 -n 128 -r 2`, warmups excluded. The original control predates the stream-ordering fix; these are retained same-day historical measurements. The tg128 row is its own zero-depth decode workload, not decode measured after each listed prefill.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="min-width: 600px; width: 100%; border-collapse: collapse; background: #161b22; color: #c9d1d9;">
<thead style="background: #21262d;"><tr><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Measurement</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Original path, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Fixed IQ path, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Conditions</th></tr></thead>
<tbody>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Prefill: 512 tokens</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">290.38</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,186.91</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">No MTP</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Prefill: 2,048 tokens</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">266.05</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">3,218.73</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">No MTP; about 12.1 times faster</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Decode: tg128</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">56.69</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">59.01</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Separate zero-depth decode measurement</td></tr>
</tbody>
</table>
</div>

Sources: [original control](iq-prefill-control-native-final.log), [fixed IQ build](iq-parity-fixed-native-isolated.log). No GPU decode arithmetic was changed to obtain the small non-speculative decode difference.

A separate identical-recipe HTTP check, 1,950 input / 128 output tokens, one session and context 8192, measured **3,077.33 prefill tokens/s** and **57.71 decode tokens/s** after the stream fix. This is a two-request mean after one excluded warmup. [HTTP receipt](iq-parity-fixed-http-string-bench.json).

### Speculative decode screen: natural prose and Go coding

Two sessions preallocated, one active request, context 32768, prefill chunk 2048, thinking off, greedy, no cached input. One excluded warmup per mode; two tasks, two measured repetitions per task, 256 output tokens each. Rates pool total generated tokens over native decode time.

The comparison table below preserves the original full-source screen. The standalone package was subsequently rechecked on the same four MTP tasks at **72.88 pooled tokens/s**, with identical completion hashes and draft counts. No new matched AR screen or native-prefill benchmark was run for this packaging change.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="min-width: 600px; width: 100%; border-collapse: collapse; background: #161b22; color: #c9d1d9;">
<thead style="background: #21262d;"><tr><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Mode</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Prose decode, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Code decode, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Pooled decode, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Gain vs AR</th></tr></thead>
<tbody>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Plain decode / AR</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">58.02</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">56.38</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">57.19</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Baseline</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><span style="color: #e3b341;">MTP2 — selected</span></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">69.18</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">77.67</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><span style="color: #e3b341;">73.18</span></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><span style="color: #3fb950;">+28.0%</span></td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">DFlash2, fixed 3</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">56.11</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">68.78</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">61.80</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><span style="color: #3fb950;">+8.1%</span></td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">DFlash2, fixed 7</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">38.40</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">54.50</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">45.05</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><span style="color: #f85149;">-21.2%</span></td></tr>
</tbody>
</table>
</div>

MTP2 is best **among these four tested profiles**. This was not a sweep of MTP1, every draft length, adaptive DFlash2, or all agent workloads. [Summary](iq-spec-natural-screen-summary.json), [measured rows](iq-spec-natural-screen-rows.json), [receipt audit](iq-spec-natural-audit.json).

### Matched context ladder: plain decode versus MTP2

These are the same three-code recall prompts and **43-token answers** in both modes. They provide paired prefill/decode measurements at actual input lengths, rather than borrowing the pooled decode speed from the prose/code screen.

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="min-width: 600px; width: 100%; border-collapse: collapse; background: #161b22; color: #c9d1d9;">
<thead style="background: #21262d;"><tr><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Input tokens</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">AR prefill, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">MTP2 prefill, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">AR decode, tok/s</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">MTP2 decode, tok/s</th></tr></thead>
<tbody>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">98</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">595.8</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">578.7</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">59.1</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">88.9</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,051</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,978.6</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,948.3</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">59.1</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">89.6</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">8,189</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">3,217.6</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">3,118.4</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">54.9</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">79.5</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">16,373</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,832.8</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,777.8</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">52.6</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">73.9</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">31,997</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,480.5</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">2,434.0</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">49.1</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">68.1</td></tr>
</tbody>
</table>
</div>

Both modes retrieved all three control codes correctly at every listed length. All MTP answers matched the plain-decode answers exactly. These short-answer rates do not replace the 256-token natural-workload screen. [Functional receipts](iq-spec-validation-v2-rows.json), [saved fixtures](iq-spec-validation-v2-fixtures.json).

With two requests actually decoding at once, the gain was smaller: per-request rates were **43.5 / 44.3** without MTP and **44.3 / 46.2 tokens/s** with MTP2 in the unconstrained nonce tests. These short checks are not a sustained concurrency benchmark.

## 4. All selected settings and adjustments

<div style="overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 16px 0; border: 1px solid #30363d; border-radius: 8px;">
<table style="min-width: 600px; width: 100%; border-collapse: collapse; background: #161b22; color: #c9d1d9;">
<thead style="background: #21262d;"><tr><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Setting</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Tested value</th><th style="white-space: nowrap; padding: 8px 12px; text-align: left; color: #58a6ff; border-bottom: 1px solid #30363d;">Purpose / scope</th></tr></thead>
<tbody>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Build / target</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>GUFO_QWEN_IQ_PREFILL=ON</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Routed IQ3_S / IQ4_XS MoE prefill; default remains OFF</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Build / tests</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>BUILD_TESTING=ON</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Measured builds include the test configuration</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Build type</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>RelWithDebInfo</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Same configuration used for control/candidate comparisons</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">GPU architecture</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>gfx1151</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">AMD Strix Halo / Radeon 8060S</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">SDK</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>TheRock 10.0.0</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Pinned native Windows HIP runtime and libraries</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Host / port</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--host 127.0.0.1 --port 8081</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Local access; leaves port 8080 for Strata</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">API model id</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--served-model-name gufo</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Client sends model gufo</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Target</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--model &lt;IQ4_XS GGUF&gt;</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Exact target path below</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Sessions</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--sessions 2</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Two request sessions share one target model</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Context</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--context 32768</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Capacity per session, including generated output</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Prefill scheduling</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--prefill-chunk 2048</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Measured chunk size</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">RAM checkpoint cache</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--cache-ram-bytes off</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">No retained RAM checkpoints in this profile</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Speculation</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--speculative mtp</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Use the external MTP head</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">MTP source</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--mtp-model &lt;standalone MTP GGUF&gt;</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Private MTP tensors; shared IQ embedding/output</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Draft maximum</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--draft-tokens 2</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Selected tested chain cap</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Draft minimum</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--min-draft-tokens 1</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Required by this HTTP server; 2 was rejected</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Thinking</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--think off</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">All reported speculative measurements use thinking off</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Temperature</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--temperature 0</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Greedy sampling</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Seed</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--seed 42</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Fixed test seed</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Presence penalty</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--presence-penalty 0</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Explicit test setting</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Repeat penalty</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>--repeat-penalty 1</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Explicit test setting</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Request cache</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>&quot;cache_prompt&quot;: false</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Every measured request had zero cached tokens</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Request thinking</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>&quot;chat_template_kwargs&quot;: {&quot;enable_thinking&quot;: false}</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Used by the OpenAI-compatible chat checks</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Request output cap</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>256 screen; 16–256 functional</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Per-request caps, not a global server maximum</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Tool selection</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>&quot;tool_choice&quot;: &quot;auto&quot;</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Tested function choice was not forced</td></tr>
<tr><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Strict JSON</td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;"><code>&quot;response_format&quot;: {&quot;type&quot;: &quot;json_schema&quot;, ...}</code></td><td style="white-space: nowrap; padding: 8px 12px; border-bottom: 1px solid #30363d;">Required when the client needs pure JSON</td></tr>
</tbody>
</table>
</div>

Other effective sampling defaults observed in the server log were `top_k=0`, `top_p=1`, `min_p=0`, `min_keep=0`, `frequency_penalty=0`, and `repeat_last_n=64`. These were logged defaults, not additional custom optimizations.

The launcher explicitly pins `PATH`, `HIP_PATH`, `ROCBLAS_TENSILE_LIBPATH` and `HIPBLASLT_TENSILE_LIBPATH` as shown above. Remaining environment variables are inherited. No additional llama.cpp or Strata tuning environment variables were selected for this profile.

### Build the opt-in engine

From `C:\Users\Butch\Documents\02_Apps\strycast`, using the existing SDK/dependency configuration:

```powershell
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\model-matched.exe' build -S . -B build/windows-release -DBUILD_TESTING=ON -DGUFO_QWEN_IQ_PREFILL=ON
& 'C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\model-matched.exe' build --build build/windows-release --target gufo gufo-server qwen35ba3b_moe_ops_test qwen35ba3b_target_test --parallel 1
```

The flag defaults to OFF in source. Rebuilding produces a new binary; review and update the launcher's pinned engine digest before substituting a rebuilt executable.

## 5. Engine files and functions modified

Paths in this section are relative to `C:\Users\Butch\Documents\02_Apps\strycast`. Final changes are in [IQ prefill commit d4bbdfa](https://github.com/Indigoshores/strycast/commit/d4bbdfa2ba592de9232d11ce7b0c8b928795c42c) and [stream fix commit 04e94a9](https://github.com/Indigoshores/strycast/commit/04e94a978c0eeddb2e25fad7a5d1eaf82bae7b5f), branch `codex/qwen36-iq-prefill-20261029`.

### `src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`

Modified symbol: `WeightType`. Added `kIQ3_S = 21` and `kIQ4_XS = 23` to identify the packed IQ formats in routed kernels.

### `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`

Modified functions/templates: `RoutedF16RowBytes`, `RoutedF16GEMMKernel`, `LaunchRoutedF16`, `RoutedF16Gemm`, `LaunchRoutedGatedF16`, `RoutedGatedF16Supported` and `RoutedF16WideTilesSupported`.

Added packed IQ3_S/IQ4_XS decoding inside routed FP16 WMMA tiles, signed code-byte staging and per-K16 scales, direct projections, and paired gate/up support. Direct tiles support 16/48/64/128 rows; paired gate/up supports 64/128 rows. The implementation reuses the existing `DecodeQuantSub16` primitive; that primitive was not rewritten. It does not allocate a full-model FP16 weight copy.

### `src/models/qwen/hip/prefill_chunk.cpp`

Modified function: `RoutedWeightType`. Maps IQ3_S and IQ4_XS to the routed fast path under `GUFO_QWEN_IQ_PREFILL`, retaining the existing K-dimension divisibility guard. Existing fallback remains selected when the build flag is OFF.

### `src/models/qwen/CMakeLists.txt`

Modified build configuration: new `GUFO_QWEN_IQ_PREFILL` option, default OFF. When enabled, applies `GUFO_QWEN_IQ_PREFILL=1` to the `gufo_core` prefill source.

### `src/core/quant/ggml_gemm.cpp`

Modified functions: `Dequantize` and `Dot`. Dispatch IQ3_S/IQ4_XS to their already implemented canonical CPU routines. This fixed the CPU-reference abort and enabled model validation; it is not a GPU decode optimization.

### `src/models/qwen/hip/arena.cpp`

Modified function: `QwenGpuArena::QwenGpuArena`. Changes the MoE scratch initialization from default-stream `hipMemset` to `hipMemsetAsync(..., stream)` on the arena's nonblocking inference stream.

The old default-stream clear could execute late at an MMQ pool synchronization and erase live router/expert data. Same-stream ordering fixed snapshot replay, serial verification and concurrent verification. The full initialization clear was retained; no numerical assertion was weakened.

## 6. Test and diagnostic files modified

- `tests/models/qwen35ba3b/moe_ops_test.cpp`: added `RandomIq` and `CheckIqCpuOracle`; modified `RandomExperts`, `CheckRoutedF16` and `main`; expanded `kRoutedCases` from four to six formats. Existing `CheckRoutedPair` and `CheckDecodeBatchInvariance` automatically exercise the new IQ cases through that shared array; their bodies were not rewritten.
- `tests/models/qwen35ba3b/target_test.cpp`: modified `CheckVerificationMatchesDecode` to require exact eight-row snapshot replay before the existing verification comparison. The concurrent assertion was retained unchanged.
- `tests/models/qwen35ba3b/verification_probe.cpp`: added the manual diagnostic, with `Compare`, `Session` and `main`. Compares replay, captured layer outputs and serial/concurrent logits.
- `tests/models/qwen35ba3b/CMakeLists.txt`: adds the `qwen35ba3b_verification_probe` executable linked to `gufo_core`. It is a manual diagnostic, not an automatically registered CTest.

The unchanged CPU GEMM parity suite was also run. Kernel checks, exact replay, eight serial rows and 23 concurrent rows pass for this IQ file with prefill ON and OFF. Numerical tolerances remain unchanged.

## 7. Existing MTP code used, not modified

- `src/models/qwen/hip/mtp/model.cpp`: `QwenMtpGpuModel::Create` and `PackPrivateWeights` load the Q4 head's private tensors and share the IQ target's token embedding/output.
- `src/models/qwen/hip/mtp/backend.cpp`: `QwenMtpGpuDraftBackend::CreateFromGguf` and `PrimeTargetContext` initialize and prime the existing draft backend.
- The existing server and speculative verifier select the backend through the runtime flags.

No MTP/DFlash engine rewrite or new GPU decode kernel was required for the measured MTP improvement. Temporary tracing in `decode_step.cpp` and the MMQ adapter was removed before the pushed fix.

## 8. Local launcher and measurement tooling

These files are in `C:\Users\Butch\Documents\02_Apps\strycast-bench-tools`, outside the Strycast source commits:

- `start-qwen36-iq-mtp2.go`: new `main` and `launchIQ` implement the pinned Windows server launcher, digest/size checks, port preservation, SDK environment and owned-child shutdown.
- `isolate-mtp.go`: losslessly extracts and verifies the standalone GGUF; its `inspect`, `extract`, `hold`, `restore` and `retire` commands check metadata, tensor hashes and exact source identity before the single-file deletion. The launcher now selects this package and checks its size.
- `spec-tools.go`: added `specIsolatedMTPVerify` to compare the standalone package's completions and draft counts against the original full-source receipts while the original path is unavailable. `main.go` dispatches `isolated-mtp-verify`.
- `spec-tools.go`: `specArms` defines AR/MTP2/DFlash3/DFlash7; `specStart` and `specServer.Stop` manage isolated servers; `specRequest` and `specPOST` capture native HTTP timings; `specScreen` runs the bounded comparison; `specAudit` checks input/output hashes, cache counts and measured summaries. Supporting types/functions serialize receipts and calculate hashes.
- `spec-validation.go`: `specChat` issues chat requests; `specCount`, `specArchive` and `specFixtureFor` construct the prompt ladder; `specTool` defines the mock invoice function; `specValidate` runs short/long/multi-turn/tool/concurrent checks; `specFormat`, `specJSONMatches` and `specJSONCheck` enforce and verify structured JSON.
- `main.go`: dispatches `spec-screen`, `spec-validate`, `spec-json-check` and `spec-audit`. Its existing `configureGufo`, `runlog`, `servebench`, `benchString` and `build` provide the SDK/runtime, logging, native-build and earlier HTTP measurement plumbing.
- Supporting local utilities include `artifact-digests.go`, `audit-gguf.go` and `compare-http.go` for receipts/model/performance checks, and `native-fault.go` / `tail-budget.go` for the earlier fault/log investigation. They are diagnostic tooling, not engine changes.

Repository documentation updated across the two commits: `docs/models/qwen3.6-35b-a3b/README.md`, `BENCHMARKS.md`, `QUALITY.md` and `EXPERIMENTS.md`.

## 9. JSON and tool-call configuration

The unconstrained functional gate retained **six pure-JSON formatting failures**, three per mode: correct JSON data was wrapped in Markdown fences. No checker stripped those fences or relabeled the original gate passed.

Use explicit `response_format: json_schema` for machine-readable output:

```json
{
  "model": "gufo",
  "messages": [
    {"role": "user", "content": "Return only JSON with my preferred color in key color. My preferred color is green."}
  ],
  "temperature": 0,
  "max_tokens": 64,
  "cache_prompt": false,
  "chat_template_kwargs": {"enable_thinking": false},
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

Automatic tool selection was tested with `tool_choice: auto`. After the mock tool result, the final JSON response used a schema. **10/10 structured follow-up responses passed**, including the tool-result continuation and two concurrent requests. Constraints specify only keys/types, not expected answer values.

## 10. Memory, validation scope and evidence

At startup, the driver reported approximately **22.90 GiB GPU device use for AR** and **23.98 GiB for MTP2**, an increase of **1.08 GiB** with two sessions and a 32K context. These are startup snapshots, not peak host-RAM measurements. DFlash2 reported approximately 24.89 GiB.

Across the speculative task, **all 29 speculative completions** matched AR exactly. Checks reached 31,997 input tokens, automatic tool calls, multi-turn transcripts and two simultaneous requests. The separate full-model and operator tests also passed after the stream fix.

This is not a broad accuracy certification. Thinking-on, nonzero-temperature sampling, sustained agent workloads, arbitrary tool/schema combinations, long tool-filled contexts and running alongside Strata remain untested. Strata was not launched or modified by these runs. IQ prefill remains an explicit build opt-in; the MTP launcher is a separately selected runtime profile.

Evidence and handoff links:

- [Speculative results, raw receipt links and limits](QWEN36_IQ4_XS_SPECULATIVE_RESULTS_2026-10-29.md)
- [Stream-ordering root cause and exactness tests](QWEN36_IQ_PREFILL_VERIFICATION_FIX_2026-10-29.md)
- [Original IQ prefill implementation](QWEN36_IQ_PREFILL_IMPLEMENTATION_2026-10-29.md)
- [Saved server profile](iq-spec-mtp2-launch-profile.json)
- [Standalone MTP extraction, runtime checks and deletion](QWEN36_ISOLATED_MTP_2026-10-29.md)
- [Current standalone package and launcher digests](isolated-mtp-artifacts.json)
- [Speculative input/output audit](iq-spec-natural-audit.json)
- [Functional validation, including original formatting failures](iq-spec-validation-v2-summary.json)
- [Structured JSON follow-up](iq-spec-json-schema-summary.json)
- [Historical full-source tool/launcher artifact digests](iq-spec-artifact-digests.json)
- [Fixed engine and source patch digests](iq-parity-fix-artifact-digests.json)
- [Final benchmark cleanup probe](iq-spec-final-probe.json)
