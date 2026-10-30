package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const outDir = `C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\results`
const modelName = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
const modelDir = `C:\Users\Butch\Documents\10-Models\models`

func init() { _ = os.Setenv("VCPKG_MAX_CONCURRENCY", "4") }

func fail(e error) {
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func save(name string, v any) {
	fail(os.MkdirAll(outDir, 0755))
	b, e := json.MarshalIndent(v, "", "  ")
	fail(e)
	fail(os.WriteFile(filepath.Join(outDir, name), b, 0644))
}
func get(url string, timeout time.Duration) (int, []byte, error) {
	c := http.Client{Timeout: timeout}
	r, e := c.Get(url)
	if e != nil {
		return 0, nil, e
	}
	defer r.Body.Close()
	b, e := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	return r.StatusCode, b, e
}

type memStatus struct {
	Length, Load                                                                          uint32
	TotalPhys, AvailPhys, TotalPage, AvailPage, TotalVirtual, AvailVirtual, AvailExtended uint64
}

func memory() map[string]any {
	m := memStatus{}
	m.Length = uint32(unsafe.Sizeof(m))
	lib := syscall.NewLazyDLL("kernel32.dll")
	p := lib.NewProc("GlobalMemoryStatusEx")
	p.Call(uintptr(unsafe.Pointer(&m)))
	var installedKB uint64
	lib.NewProc("GetPhysicallyInstalledSystemMemory").Call(uintptr(unsafe.Pointer(&installedKB)))
	return map[string]any{"installed_gib": float64(installedKB) / (1 << 20), "os_visible_gib": float64(m.TotalPhys) / (1 << 30), "available_gib": float64(m.AvailPhys) / (1 << 30), "commit_available_gib": float64(m.AvailPage) / (1 << 30)}
}
func probe() {
	result := map[string]any{"time": time.Now().Format(time.RFC3339), "memory": memory()}
	for _, port := range []int{8080, 8081, 8082} {
		base := fmt.Sprintf("http://127.0.0.1:%d", port)
		v := map[string]any{}
		for _, route := range []string{"/health", "/v1/models", "/props"} {
			s, b, e := get(base+route, 3*time.Second)
			if e != nil {
				v[route] = e.Error()
				continue
			}
			var j any
			if json.Unmarshal(b, &j) != nil {
				j = string(b)
			}
			if route == "/props" {
				if o, ok := j.(map[string]any); ok {
					j = map[string]any{"total_slots": o["total_slots"], "model_path": o["model_path"], "default_generation_settings": o["default_generation_settings"]}
				}
			}
			v[route] = map[string]any{"status": s, "body": j}
		}
		result[strconv.Itoa(port)] = v
	}
	label := "initial-probe.json"
	if len(os.Args) > 2 { label = os.Args[2] }
	save(label, result)
	b, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(b))
}
func info(paths []string) {
	for _, p := range paths {
		s, e := os.Stat(p)
		if e != nil {
			fmt.Printf("%s: %v\n", p, e)
			continue
		}
		fmt.Printf("%s size=%d directory=%v\n", p, s.Size(), s.IsDir())
	}
	for _, name := range []string{"cmake", "ninja", "git", "go", "clang-cl", "cl"} {
		p, e := exec.LookPath(name)
		fmt.Printf("tool %s path=%s error=%v\n", name, p, e)
	}
	for _, key := range []string{"HIP_PATH", "VCPKG_ROOT"} {
		fmt.Printf("env %s=%s\n", key, os.Getenv(key))
	}
	var free, total, all uint64
	p := syscall.NewLazyDLL("kernel32.dll").NewProc("GetDiskFreeSpaceExW")
	root, _ := syscall.UTF16PtrFromString(`C:\`)
	p.Call(uintptr(unsafe.Pointer(root)), uintptr(unsafe.Pointer(&free)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&all)))
	fmt.Printf("C free_gib=%.2f memory=%v\n", float64(free)/(1<<30), memory())
}
func release() {
	s, b, e := get("https://api.github.com/repos/slimsami/strycast/releases?per_page=5", 30*time.Second)
	fail(e)
	if s != 200 {
		fail(fmt.Errorf("release HTTP %d: %s", s, b))
	}
	var rows []struct {
		Tag    string `json:"tag_name"`
		Assets []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
			Size int64  `json:"size"`
		} `json:"assets"`
	}
	fail(json.Unmarshal(b, &rows))
	save("strycast-releases.json", rows)
	o, _ := json.MarshalIndent(rows, "", "  ")
	fmt.Println(string(o))
}

type fileMeta struct {
	Type, Path, Oid string
	Size            int64
	LFS             struct {
		Oid  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

func metadata(repoID, revision, filename string) (string, fileMeta) {
	repo := "https://huggingface.co/api/models/" + repoID
	s, b, e := get(repo+"/revision/"+revision, 30*time.Second)
	fail(e)
	if s != 200 {
		fail(fmt.Errorf("metadata HTTP %d", s))
	}
	var m struct {
		Sha string `json:"sha"`
	}
	fail(json.Unmarshal(b, &m))
	if len(m.Sha) != 40 {
		fail(fmt.Errorf("invalid revision %q", m.Sha))
	}
	s, b, e = get(repo+"/tree/"+m.Sha+"?recursive=false&expand=true&limit=100", 30*time.Second)
	fail(e)
	if s != 200 {
		fail(fmt.Errorf("tree HTTP %d: %s", s, b))
	}
	var fs []fileMeta
	fail(json.Unmarshal(b, &fs))
	for _, f := range fs {
		if f.Path == filename {
			return m.Sha, f
		}
	}
	fail(fmt.Errorf("model absent from remote tree"))
	return "", fileMeta{}
}
func digest(path string) string {
	f, e := os.Open(path)
	fail(e)
	defer f.Close()
	h := sha256.New()
	_, e = io.Copy(h, f)
	fail(e)
	return hex.EncodeToString(h.Sum(nil))
}
func download() {
	repoID, revision, filename, receiptName := "unsloth/Qwen3.6-35B-A3B-GGUF", "main", modelName, "download-receipt.json"
	if len(os.Args) > 2 {
		if len(os.Args) != 6 { fail(fmt.Errorf("download repo revision filename receipt-name")) }
		repoID, revision, filename, receiptName = os.Args[2], os.Args[3], os.Args[4], os.Args[5]
	}
	rev, m := metadata(repoID, revision, filename)
	expected := strings.TrimPrefix(m.LFS.Oid, "sha256:")
	if len(expected) != 64 {
		fail(fmt.Errorf("missing official sha256: %+v", m))
	}
	dest := filepath.Join(modelDir, filename)
	part := dest + ".partial"
	fmt.Printf("revision=%s bytes=%d expected_sha256=%s\n", rev, m.Size, expected)
	if s, e := os.Stat(dest); e == nil {
		if s.Size() != m.Size {
			fail(fmt.Errorf("existing final file size differs; preserving it"))
		}
		got := digest(dest)
		if got != expected {
			fail(fmt.Errorf("existing final file digest differs; preserving it"))
		}
		fmt.Println("existing download verified")
		return
	}
	var offset int64
	if s, e := os.Stat(part); e == nil {
		offset = s.Size()
	}
	if offset > m.Size {
		fail(fmt.Errorf("partial file exceeds expected size; preserving it"))
	}
	if offset < m.Size {
		req, e := http.NewRequest("GET", "https://huggingface.co/"+repoID+"/resolve/"+rev+"/"+filename+"?download=true", nil)
		fail(e)
		if offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		c := http.Client{Timeout: 90 * time.Minute}
		r, e := c.Do(req)
		fail(e)
		defer r.Body.Close()
		if offset > 0 && r.StatusCode != 206 {
			fail(fmt.Errorf("resume requires 206, got %d", r.StatusCode))
		}
		if offset == 0 && r.StatusCode != 200 {
			b, _ := io.ReadAll(io.LimitReader(r.Body, 2048))
			fail(fmt.Errorf("download HTTP %d: %s", r.StatusCode, b))
		}
		f, e := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		fail(e)
		start := time.Now()
		last := start
		total := offset
		buf := make([]byte, 4<<20)
		for {
			n, er := r.Body.Read(buf)
			if n > 0 {
				_, e = f.Write(buf[:n])
				fail(e)
				total += int64(n)
			}
			if time.Since(last) >= 20*time.Second {
				fmt.Printf("download %.2f/%.2f GiB %.1f MiB/s\n", float64(total)/(1<<30), float64(m.Size)/(1<<30), float64(total-offset)/(1<<20)/time.Since(start).Seconds())
				last = time.Now()
			}
			if er == io.EOF {
				break
			}
			fail(er)
		}
		fail(f.Close())
		if total != m.Size {
			fail(fmt.Errorf("size mismatch %d vs %d", total, m.Size))
		}
	}
	fmt.Println("verifying downloaded SHA256")
	got := digest(part)
	if got != expected {
		fail(fmt.Errorf("SHA256 mismatch %s", got))
	}
	fail(os.Rename(part, dest))
	receipt := map[string]any{"path": dest, "repository": repoID, "revision": rev, "bytes": m.Size, "sha256": got, "verified": true, "completed": time.Now().Format(time.RFC3339)}
	save(receiptName, receipt)
	o, _ := json.Marshal(receipt)
	fmt.Println(string(o))
}
func read(path string, start, end int) {
	f, e := os.Open(path)
	fail(e)
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 4<<20)
	n := 0
	for s.Scan() {
		n++
		if n >= start && n <= end {
			fmt.Printf("%d: %s\n", n, s.Text())
		}
		if n >= end {
			break
		}
	}
	fail(s.Err())
}
func run(timeout int, argv []string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, argv[0], argv[1:]...)
	configureGufo(c)
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	b, e := c.CombinedOutput()
	if len(b) > 16000 {
		b = b[len(b)-16000:]
	}
	fmt.Print(string(b))
	fail(e)
}

func runlog(timeout int, label string, argv []string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, argv[0], argv[1:]...)
	configureGufo(c)
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	fail(os.MkdirAll(outDir, 0755))
	f, e := os.Create(filepath.Join(outDir, label+".log"))
	fail(e)
	fmt.Fprintf(f, "Command: %s %q\nStarted: %s\n", argv[0], argv[1:], time.Now().Format(time.RFC3339))
	c.Stdout = io.MultiWriter(os.Stdout, f)
	c.Stderr = io.MultiWriter(os.Stderr, f)
	e = c.Run()
	fmt.Fprintf(f, "\nCompleted: %s; error: %v\n", time.Now().Format(time.RFC3339), e)
	fail(f.Close())
	fail(e)
}

func configureGufo(c *exec.Cmd) {
	if !strings.Contains(strings.ToLower(c.Path), `\strycast\build\windows-release\`) { return }
	const sdkRoot = `C:\Users\Butch\Documents\02_Apps\strycast-deps\therock-10.0.0`
	dir := filepath.Dir(c.Path)
	c.Dir = dir
	c.Env = append(os.Environ(),
		"PATH="+dir+`;`+filepath.Join(dir, "vcpkg_installed", "x64-windows-release", "bin")+`;`+filepath.Join(sdkRoot,"bin")+`;`+os.Getenv("PATH"),
		"HIP_PATH="+sdkRoot,
		"ROCBLAS_TENSILE_LIBPATH="+filepath.Join(sdkRoot,"bin","rocblas","library"),
		"HIPBLASLT_TENSILE_LIBPATH="+filepath.Join(sdkRoot,"bin","hipblaslt","library"))
}

type result struct {
	Label           string  `json:"label"`
	Endpoint        string  `json:"endpoint"`
	PromptTokens    int     `json:"prompt_tokens"`
	GeneratedTokens int     `json:"generated_tokens"`
	PromptMS        float64 `json:"prompt_ms"`
	PrefillTPS      float64 `json:"prefill_tps"`
	DecodeTPS       float64 `json:"decode_tps"`
	WallMS          float64 `json:"wall_ms"`
	Output          string  `json:"output"`
	Timings         any     `json:"timings"`
	Error           string  `json:"error,omitempty"`
	Memory          any     `json:"memory"`
}

func bench(base, label string) {
	c := http.Client{Timeout: 3 * time.Minute}
	var rows []result
	for _, size := range []int{512, 2048} {
		for rep := 0; rep < 3; rep++ {
			prompt := fmt.Sprintf("Benchmark independent request case-%d-%d. Read the following reference facts and then write a precise numbered summary.\n", size, rep)
			for n := 0; n < size/22; n++ {
				prompt += fmt.Sprintf("Record %d: The scheduling service validates task dependencies, stores durable execution records, reports worker progress, and retries failed operations with bounded exponential delays.\n", n)
			}
			prompt += "\nWrite a numbered technical summary with at least ten concrete points.\n"
			body := map[string]any{"prompt": prompt, "n_predict": 128, "temperature": 0, "seed": 1234, "cache_prompt": false, "stream": false, "ignore_eos": true}
			tokBody, _ := json.Marshal(map[string]any{"content": prompt, "add_special": true})
			tokResp, tokErr := c.Post(base+"/tokenize", "application/json", bytes.NewReader(tokBody))
			if tokErr != nil { fail(tokErr) }
			var tokenized struct { Tokens []int `json:"tokens"` }
			tokErr = json.NewDecoder(tokResp.Body).Decode(&tokenized)
			tokResp.Body.Close()
			fail(tokErr)
			if len(tokenized.Tokens) < size { fail(fmt.Errorf("insufficient benchmark input tokens: %d", len(tokenized.Tokens))) }
			body["prompt"] = tokenized.Tokens[:size]
			b, _ := json.Marshal(body)
			start := time.Now()
			r, e := c.Post(base+"/completion", "application/json", bytes.NewReader(b))
			row := result{Label: fmt.Sprintf("%s-size%d-rep%d", label, size, rep), Endpoint: base, Memory: memory()}
			if e != nil {
				row.Error = e.Error()
				rows = append(rows, row)
				save(label+"-bench.json", rows)
				fmt.Printf("%s ERROR %v\n", row.Label, e)
				return
			}
			raw, _ := io.ReadAll(io.LimitReader(r.Body, 4<<20))
			r.Body.Close()
			row.WallMS = float64(time.Since(start).Microseconds()) / 1000
			var j map[string]any
			if r.StatusCode != 200 || json.Unmarshal(raw, &j) != nil {
				row.Error = fmt.Sprintf("HTTP %d %s", r.StatusCode, raw)
				rows = append(rows, row)
				save(label+"-bench.json", rows)
				fmt.Println(row.Error)
				return
			}
			row.Output, _ = j["content"].(string)
			row.Timings = j["timings"]
			if t, ok := j["timings"].(map[string]any); ok {
				num := func(k string) float64 { v, _ := t[k].(float64); return v }
				row.PromptTokens = int(num("prompt_n"))
				row.GeneratedTokens = int(num("predicted_n"))
				row.PromptMS = num("prompt_ms")
				row.PrefillTPS = num("prompt_per_second")
				row.DecodeTPS = num("predicted_per_second")
			}
			rows = append(rows, row)
			save(label+"-bench.json", rows)
			kind := "timed"
			if rep == 0 {
				kind = "warmup"
			}
			fmt.Printf("%s %s prompt=%d generated=%d prefill=%.2f decode=%.2f wall_ms=%.1f\n", row.Label, kind, row.PromptTokens, row.GeneratedTokens, row.PrefillTPS, row.DecodeTPS, row.WallMS)
		}
	}
}
func gguf(path string) {
	f, e := os.Open(path)
	fail(e)
	defer f.Close()
	var magic, version uint32
	var tensors, kvs uint64
	fail(binary.Read(f, binary.LittleEndian, &magic))
	fail(binary.Read(f, binary.LittleEndian, &version))
	fail(binary.Read(f, binary.LittleEndian, &tensors))
	fail(binary.Read(f, binary.LittleEndian, &kvs))
	fmt.Printf("GGUF magic=%x version=%d tensors=%d metadata=%d\n", magic, version, tensors, kvs)
	if magic != 0x46554747 {
		fail(fmt.Errorf("invalid GGUF magic"))
	}
}

func benchString(base, label string) error {
	client := http.Client{Timeout: 2*time.Minute}
	var rows []map[string]any
	for rep:=0; rep<3; rep++ {
		prompt := fmt.Sprintf("<|im_start|>user\nIndependent benchmark request %d. Read these reference facts.\n", rep)
		for n:=0; n<62; n++ { prompt += fmt.Sprintf("Record %d: The scheduling service validates task dependencies, stores durable execution records, reports worker progress, and retries failed operations with bounded exponential delays.\n", n) }
		prompt += "Write a detailed numbered summary with at least twenty points and explanations.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
		body := map[string]any{"prompt":prompt,"n_predict":128,"temperature":0,"seed":1234,"cache_prompt":false,"stream":false}
		encoded, err := json.Marshal(body); if err!=nil{return err}
		sum:=sha256.Sum256([]byte(prompt))
		start:=time.Now()
		resp, err:=client.Post(base+"/completion","application/json",bytes.NewReader(encoded));if err!=nil{return err}
		raw,err:=io.ReadAll(io.LimitReader(resp.Body,4<<20));resp.Body.Close();if err!=nil{return err}
		var reply map[string]any
		if resp.StatusCode!=200 || json.Unmarshal(raw,&reply)!=nil{return fmt.Errorf("HTTP %d: %s",resp.StatusCode,raw)}
		save(fmt.Sprintf("%s-response-%d.json",label,rep),reply)
		timing,ok:=reply["timings"].(map[string]any);if !ok{return fmt.Errorf("response has no native timings")}
		num:=func(k string)float64{v,_:=timing[k].(float64);return v}
		kind:="measured";if rep==0{kind="warmup"}
		row:=map[string]any{"label":label,"repetition":rep,"kind":kind,"prompt_sha256":hex.EncodeToString(sum[:]),"prompt_tokens":num("prompt_n"),"generated_tokens":num("predicted_n"),"prefill_tps":num("prompt_per_second"),"decode_tps":num("predicted_per_second"),"timings":timing,"wall_ms":float64(time.Since(start).Microseconds())/1000}
		rows=append(rows,row);save(label+"-string-bench.json",rows)
		fmt.Printf("%s %s prompt=%.0f generated=%.0f cached=%.0f prefill=%.2f decode=%.2f\n",label,kind,num("prompt_n"),num("predicted_n"),num("cache_n"),num("prompt_per_second"),num("predicted_per_second"))
		if num("cache_n")!=0{return fmt.Errorf("unexpected cached input")}
		if num("predicted_n")!=128{return fmt.Errorf("generation stopped at %.0f tokens; not a tg128 comparison",num("predicted_n"))}
	}
	return nil
}
func servebench(argv []string) {
	if len(argv) < 5 {
		fail(fmt.Errorf("servebench exe model port label backend [extra args]"))
	}
	exe, model, port, label, backend := argv[0], argv[1], argv[2], argv[3], argv[4]
	conn, e := net.DialTimeout("tcp", "127.0.0.1:"+port, time.Second)
	if e == nil {
		conn.Close()
		fail(fmt.Errorf("port %s already occupied; preserving listener", port))
	}
	args := []string{}
	if strings.HasPrefix(backend, "llama") {
		args = []string{"-m", model, "--host", "127.0.0.1", "--port", port, "-ngl", "999", "-c", "8192", "-np", "1", "-b", "2048", "-ub", "512", "-t", "8", "-tb", "16", "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--cache-ram", "0", "--jinja"}
		if backend == "llama-tuned" { args = append(args, "-c", "262144", "-t", "4", "-tb", "8", "--poll", "100", "--prio", "2", "--prio-batch", "2", "-ctxcp", "32", "-cms", "64", "--cache-prompt", "--load-mode", "mmap", "--fit", "off", "--reasoning-preserve") }
	} else {
		args = []string{"serve", "--host", "127.0.0.1", "--port", port, "llm", "--model", model, "--sessions", "1", "--context", "8192"}
	}
	args = append(args, argv[5:]...)
	fail(os.MkdirAll(outDir, 0755))
	log, e := os.Create(filepath.Join(outDir, label+"-server.log"))
	fail(e)
	defer log.Close()
	c := exec.Command(exe, args...)
	c.Dir = filepath.Dir(exe)
	configureGufo(c)
	if backend == "llama-tuned" || backend == "llama-http" {
		c.Env = append(os.Environ(),
			`PATH=C:\Users\Butch\Documents\10-Models\Strata\engine\rocm\bin;C:\Program Files\AMD\ROCm\7.2\bin;`+os.Getenv("PATH"),
			"ROCBLAS_USE_HIPBLASLT=1", "HSA_ENABLE_SDMA=0", "GPU_MAX_HW_QUEUES=8", "LLAMA_QSA_GATHER=1", "GGML_HIP_GDN_CHUNK=1")
	}
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	c.Stdout = log
	c.Stderr = log
	fmt.Printf("launch %s %q memory=%v\n", exe, args, memory())
	fail(c.Start())
	done := make(chan error, 1)
	go func() { done <- c.Wait() }()
	defer func() {
		if c.Process != nil {
			_ = c.Process.Kill()
		}
	}()
	save(label+"-launch.json", map[string]any{"pid": c.Process.Pid, "exe": exe, "args": args, "backend_recipe": backend, "tuned_environment": map[string]string{"ROCBLAS_USE_HIPBLASLT":"1", "HSA_ENABLE_SDMA":"0", "GPU_MAX_HW_QUEUES":"8", "LLAMA_QSA_GATHER":"1", "GGML_HIP_GDN_CHUNK":"1"}, "memory": memory()})
	base := "http://127.0.0.1:" + port
	deadline := time.Now().Add(4 * time.Minute)
	ready := false
	for time.Now().Before(deadline) {
		select {
		case err := <-done:
			fmt.Printf("server exited: %v\n", err)
			save(label+"-smoke.json", map[string]any{"status":"server_exited", "error":fmt.Sprint(err)})
			read(filepath.Join(outDir, label+"-server.log"), 1, 160)
			return
		default:
		}
		s, _, er := get(base+"/health", time.Second)
		if er == nil && s == 200 {
			ready = true
			break
		}
		s, _, er = get(base+"/ready", time.Second)
		if er == nil && s == 200 {
			ready = true
			break
		}
		time.Sleep(2 * time.Second)
	}
	if !ready {
		fmt.Println("server readiness timeout")
		return
	}
	fmt.Println("server ready")
	if strings.HasSuffix(backend, "-http") {
		if err:=benchString(base,label);err!=nil { fmt.Println("benchmark error:",err);save(label+"-error.json",map[string]any{"error":err.Error()});return }
	} else if strings.HasPrefix(backend, "llama") {
		bench(base, label)
	} else {
		modelID := filepath.Base(model)
		if status, raw, err := get(base+"/v1/models", 5*time.Second); err == nil && status == 200 {
			var listing struct { Data []struct { ID string `json:"id"` } `json:"data"` }
			if json.Unmarshal(raw, &listing) == nil && len(listing.Data)>0 { modelID=listing.Data[0].ID }
		}
		body := map[string]any{"model": modelID, "messages": []map[string]string{{"role": "user", "content": "Reply with exactly STRYCAST_READY"}}, "temperature": 0, "max_tokens": 512, "stream": false}
		b, _ := json.Marshal(body)
		client := http.Client{Timeout: 90 * time.Second}
		r, e := client.Post(base+"/v1/chat/completions", "application/json", bytes.NewReader(b))
		if e != nil {
			fmt.Println(e)
			return
		}
		defer r.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		fmt.Printf("Strycast smoke HTTP %d: %s\n", r.StatusCode, raw)
		save(label+"-smoke.json", map[string]any{"status": r.StatusCode, "body": string(raw), "memory": memory()})
	}
	_ = c.Process.Kill()
	<-done
	c.Process = nil
}
func sdk() {
	const dir = `C:\Users\Butch\Documents\02_Apps\strycast-deps`
	const expected = "1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38"
	fail(os.MkdirAll(dir, 0755))
	dest := filepath.Join(dir, "therock-dist-windows-gfx1151-10.0.0.tar.gz")
	if _, e := os.Stat(dest); e != nil {
		req, e := http.NewRequest("GET", "https://github.com/endyjasmi/gufo/releases/download/toolchain/therock-10.0.0/therock-dist-windows-gfx1151-10.0.0.tar.gz", nil)
		fail(e)
		c := http.Client{Timeout: 30 * time.Minute}
		r, e := c.Do(req)
		fail(e)
		defer r.Body.Close()
		if r.StatusCode != 200 {
			fail(fmt.Errorf("SDK HTTP %d", r.StatusCode))
		}
		f, e := os.OpenFile(dest+".partial", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		fail(e)
		buf := make([]byte, 4<<20)
		var total int64
		last := time.Now()
		for {
			n, er := r.Body.Read(buf)
			if n > 0 {
				_, e = f.Write(buf[:n])
				fail(e)
				total += int64(n)
			}
			if time.Since(last) > 20*time.Second {
				fmt.Printf("SDK downloaded %.2f GiB\n", float64(total)/(1<<30))
				last = time.Now()
			}
			if er == io.EOF {
				break
			}
			fail(er)
		}
		fail(f.Close())
		if digest(dest+".partial") != expected {
			fail(fmt.Errorf("SDK sha mismatch"))
		}
		fail(os.Rename(dest+".partial", dest))
	}
	if digest(dest) != expected {
		fail(fmt.Errorf("SDK digest mismatch"))
	}
	fmt.Println("SDK archive SHA256 verified")
	root := filepath.Join(dir, "therock-10.0.0")
	if _, e := os.Stat(filepath.Join(root, ".extracted")); e == nil {
		fmt.Println("SDK already extracted")
		return
	}
	f, e := os.Open(dest)
	fail(e)
	defer f.Close()
	gz, e := gzip.NewReader(f)
	fail(e)
	defer gz.Close()
	tr := tar.NewReader(gz)
	count := 0
	for {
		h, er := tr.Next()
		if er == io.EOF {
			break
		}
		fail(er)
		name := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(name) || name == ".." || strings.HasPrefix(name, ".."+string(os.PathSeparator)) {
			fail(fmt.Errorf("unsafe archive path %q", h.Name))
		}
		target := filepath.Join(root, name)
		rel, e := filepath.Rel(root, target)
		fail(e)
		if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			fail(fmt.Errorf("archive traversal"))
		}
		switch h.Typeflag {
		case tar.TypeDir:
			fail(os.MkdirAll(target, 0755))
		case tar.TypeReg, tar.TypeRegA:
			fail(os.MkdirAll(filepath.Dir(target), 0755))
			out, e := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
			fail(e)
			_, e = io.Copy(out, tr)
			out.Close()
			fail(e)
			count++
		default:
			fail(fmt.Errorf("unexpected archive entry type %d %s", h.Typeflag, h.Name))
		}
		if count > 0 && count%2000 == 0 {
			fmt.Printf("SDK extracted %d files\n", count)
		}
	}
	fail(os.WriteFile(filepath.Join(root, ".extracted"), []byte(expected), 0644))
	fmt.Printf("SDK extracted %d files to %s\n", count, root)
}
func patchHeaders(root string) {
	base := filepath.Join(root, "lib", "llvm", "lib", "clang")
	versions, e := os.ReadDir(base)
	fail(e)
	funcs := []string{"isgreater", "isgreaterequal", "isless", "islessequal", "islessgreater", "isunordered"}
	open := "#if !defined(_MSC_VER)  // GUFO_MSVC_CMATH_PATCH\n"
	for _, v := range versions {
		for _, sub := range []string{"", "include"} {
			for _, header := range []string{"__clang_cuda_math_forward_declares.h", "__clang_hip_cmath.h"} {
				path := filepath.Join(base, v.Name(), sub, header)
				raw, e := os.ReadFile(path)
				if os.IsNotExist(e) {
					continue
				}
				fail(e)
				old := strings.ReplaceAll(string(raw), "\r\n", "\n")
				text := old
				for _, fn := range funcs {
					snippets := []string{"using ::" + fn + ";\n"}
					for _, typ := range []string{"float", "double"} {
						if header == "__clang_cuda_math_forward_declares.h" {
							snippets = append(snippets, fmt.Sprintf("__DEVICE__ bool %s(%s, %s);\n", fn, typ, typ))
						} else {
							snippets = append(snippets, fmt.Sprintf("__DEVICE__ __CONSTEXPR__ bool %s(%s __x, %s __y) {\n  return __builtin_%s(__x, __y);\n}\n", fn, typ, typ, fn))
						}
					}
					for _, snippet := range snippets {
						if !strings.Contains(text, open+snippet) && strings.Contains(text, snippet) {
							text = strings.Replace(text, snippet, open+snippet+"#endif\n", 1)
						}
					}
				}
				if text != old {
					if _, e := os.Stat(path + ".before-strycast"); os.IsNotExist(e) {
						fail(os.WriteFile(path+".before-strycast", raw, 0644))
					}
					fail(os.WriteFile(path, []byte(text), 0644))
					fmt.Println("patched", path)
				}
			}
		}
	}
}
func build(argv []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	var c *exec.Cmd
	if len(argv) > 0 && argv[0] == "bootstrap" {
		c = exec.CommandContext(ctx, "cmd.exe", "/d", "/c", `C:\Users\Butch\Documents\02_Apps\strycast-deps\vcpkg\bootstrap-vcpkg.bat`, "-disableMetrics")
	} else {
		args := append([]string{"/d", "/c", `tools\windows\vs-env.cmd`, "cmake"}, argv...)
		c = exec.CommandContext(ctx, "cmd.exe", args...)
		c.Dir = `C:\Users\Butch\Documents\02_Apps\strycast`
	}
	c.Env = append(os.Environ(), `VCPKG_ROOT=C:\Users\Butch\Documents\02_Apps\strycast-deps\vcpkg`, `HIP_PATH=C:\Users\Butch\Documents\02_Apps\strycast-deps\therock-10.0.0`)
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	fail(os.MkdirAll(outDir, 0755))
	log, e := os.Create(filepath.Join(outDir, "build-"+time.Now().Format("150405")+".log"))
	fail(e)
	defer log.Close()
	c.Stdout = io.MultiWriter(os.Stdout, log)
	c.Stderr = io.MultiWriter(os.Stderr, log)
	e = c.Run()
	if e != nil {
		fail(e)
	}
}
func main() {
	if len(os.Args) < 2 {
		fail(fmt.Errorf("subcommand required"))
	}
	switch os.Args[1] {
	case "probe":
		probe()
	case "info":
		info(os.Args[2:])
	case "releases":
		release()
	case "download":
		download()
	case "sdk":
		sdk()
	case "patch-sdk":
		patchHeaders(os.Args[2])
	case "build":
		build(os.Args[2:])
	case "read":
		a, e := strconv.Atoi(os.Args[3])
		fail(e)
		b, e := strconv.Atoi(os.Args[4])
		fail(e)
		read(os.Args[2], a, b)
	case "run":
		n, e := strconv.Atoi(os.Args[2])
		fail(e)
		run(n, os.Args[3:])
	case "runlog":
		n,e:=strconv.Atoi(os.Args[2]);fail(e)
		runlog(n,os.Args[3],os.Args[4:])
	case "bench":
		bench(os.Args[2], os.Args[3])
	case "gguf":
		gguf(os.Args[2])
	case "servebench":
		servebench(os.Args[2:])
	case "spec-screen":
		fail(specScreen(os.Args[2:]))
	case "spec-validate":
		fail(specValidate(os.Args[2:]))
	case "spec-json-check":
		fail(specJSONCheck(os.Args[2:]))
	case "spec-audit":
		fail(specAudit(os.Args[2:]))
	case "isolated-mtp-verify":
		fail(specIsolatedMTPVerify(os.Args[2:]))
	default:
		fail(fmt.Errorf("unknown subcommand"))
	}
}
