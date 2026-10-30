# Qwen3.6 IQ4_XS standalone MTP: Windows experiment snapshot

This bundle preserves the October 29, 2026 local tooling and evidence for running
Qwen3.6-35B-A3B UD-IQ4_XS with a losslessly extracted standalone MTP package on a
128 GB Strix Halo. It changes no engine defaults or runtime code.

The 1,620,490,976-byte MTP GGUF replaced the 22,853,663,008-byte full Q4 source.
The original was deleted after byte-level verification and runtime checks,
reclaiming 21,233,172,032 bytes net. Four natural-workload completions and draft
counts match the original source exactly, at 72.88 pooled decode tokens/s.
Structured JSON, automatic tool/result and two-session follow-up checks pass 10/10.

- [Current server settings and complete optimization history](results/QWEN36_IQ4_XS_MTP2_SERVER_SETTINGS_2026-10-29.md)
- [Standalone extraction, verification and deletion report](results/QWEN36_ISOLATED_MTP_2026-10-29.md)
- [Original speculative comparison and quality boundaries](results/QWEN36_IQ4_XS_SPECULATIVE_RESULTS_2026-10-29.md)
- [Snapshot manifest](snapshot-manifest.json): exact copied file sizes and SHA256 values.

## Sources and build

These are standalone Go 1.26 programs, not a single Go package. On Windows, build
explicit file sets from this directory, selecting output paths outside the source
tree if desired:

```powershell
go build -o isolate-mtp.exe isolate-mtp.go
go build -o start-qwen36-iq-mtp2.exe start-qwen36-iq-mtp2.go
go build -o spec-tools.exe main.go spec-tools.go spec-validation.go
go build -o artifact-digests.exe artifact-digests.go
```

- `isolate-mtp.go`: GGUF subset extraction and tensor verification. Source holding,
  rollback and retirement are tightly bound to the recorded local source identity
  and passed runtime receipts. The recorded source deletion is already complete.
- `start-qwen36-iq-mtp2.go`: tested local launcher, SDK environment, engine digest,
  model sizes, listener preservation and owned-child shutdown.
- `main.go`, `spec-tools.go`, `spec-validation.go`: bounded benchmark, exact-output
  comparison, structured JSON, mock tool/result and concurrent-session harness.
- `artifact-digests.go`: hashes explicitly supplied artifacts.

These are exact local snapshots with absolute Windows paths, pinned engine hashes,
and hardware-specific settings. The active installation remains in
`C:\Users\Butch\Documents\02_Apps\strycast-bench-tools`; copying this directory
does not install, launch or configure a server. Receipts preserve historical local
paths and binary digests; a rebuild is compile verification, not a reproduction of
those historical binaries or performance measurements.

No GGUF weights, executables, SDK files, graph caches or unrelated workspace edits
are included. Text reports retain their linked receipts and logs. Earlier engine
changes are committed separately as `d4bbdfa` and `04e94a9`.
