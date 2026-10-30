// Native Windows launcher for the measured IQ4_XS + external MTP2 profile.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	if e := launchIQ(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func launchIQ() error {
	const root = `C:\Users\Butch\Documents\02_Apps\strycast\build\windows-release`
	const exe = root + `\gufo-iq-prefill-fixed.exe`
	const sdk = `C:\Users\Butch\Documents\02_Apps\strycast-deps\therock-10.0.0`
	const target = `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
	const draft = `C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`
	const expected = `72200a67f689b308b4aa120a8e843709cd47a50d90fb918e0e0b0fde9c85a3d1`
	args := []string{"serve", "--host", "127.0.0.1", "--port", "8081", "llm", "--model", target, "--sessions", "2", "--context", "32768", "--prefill-chunk", "2048", "--cache-ram-bytes", "off", "--think", "off", "--temperature", "0", "--seed", "42", "--presence-penalty", "0", "--repeat-penalty", "1", "--served-model-name", "gufo", "--speculative", "mtp", "--mtp-model", draft, "--draft-tokens", "2", "--min-draft-tokens", "1"}
	if len(os.Args) > 1 {
		if len(os.Args) != 2 || os.Args[1] != "--print-config" {
			return fmt.Errorf("usage: start-qwen36-iq-mtp2.exe [--print-config]")
		}
		b, e := json.MarshalIndent(map[string]any{"exe": exe, "args": args, "expected_exe_sha256": expected, "sdk": sdk, "base_url": "http://127.0.0.1:8081/v1", "api_model": "gufo", "reasoning": "off", "note": "Measured experimental IQ prefill build; standalone MTP sidecar extracted losslessly from Q4; launches no Strata process."}, "", "  ")
		if e != nil {
			return e
		}
		fmt.Println(string(b))
		return nil
	}
	if c, e := net.DialTimeout("tcp", "127.0.0.1:8081", time.Second); e == nil {
		c.Close()
		return fmt.Errorf("port 8081 is occupied; existing server preserved")
	}
	for name, size := range map[string]int64{target: 17730509792, draft: 1620490976} {
		s, e := os.Stat(name)
		if e != nil {
			return e
		}
		if s.Size() != size {
			return fmt.Errorf("model size changed: %s; verify the launch profile before use", name)
		}
	}
	f, e := os.Open(exe)
	if e != nil {
		return e
	}
	h := sha256.New()
	_, e = io.Copy(h, f)
	f.Close()
	if e != nil {
		return e
	}
	if hex.EncodeToString(h.Sum(nil)) != expected {
		return fmt.Errorf("engine differs from the measured binary; review the profile before launching")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	c := exec.CommandContext(ctx, exe, args...)
	c.Dir = root
	c.Env = append(os.Environ(), "PATH="+root+`;`+filepath.Join(root, "vcpkg_installed", "x64-windows-release", "bin")+`;`+filepath.Join(sdk, "bin")+`;`+os.Getenv("PATH"), "HIP_PATH="+sdk, "ROCBLAS_TENSILE_LIBPATH="+filepath.Join(sdk, "bin", "rocblas", "library"), "HIPBLASLT_TENSILE_LIBPATH="+filepath.Join(sdk, "bin", "hipblaslt", "library"))
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	c.Stdin = os.Stdin
	fmt.Println("Starting Qwen3.6 IQ4_XS + MTP2 at http://127.0.0.1:8081/v1 ; model gufo. Press Ctrl+C to stop this server.")
	e = c.Run()
	if ctx.Err() != nil {
		return nil
	}
	return e
}
