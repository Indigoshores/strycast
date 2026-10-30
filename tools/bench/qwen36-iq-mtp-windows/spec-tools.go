package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type specArm struct {
	Name string
	Args []string
}
type specServer struct {
	cmd  *exec.Cmd
	done chan error
	log  *os.File
	Base string
}
type specRow struct {
	Arm          string         `json:"arm"`
	Case         string         `json:"case"`
	Rep          int            `json:"repetition"`
	Status       int            `json:"http_status"`
	InputHash    string         `json:"input_sha256"`
	OutputHash   string         `json:"output_sha256"`
	Content      string         `json:"content"`
	PromptTokens float64        `json:"prompt_tokens"`
	Generated    float64        `json:"generated_tokens"`
	PrefillTPS   float64        `json:"prefill_tps"`
	DecodeTPS    float64        `json:"decode_tps"`
	WallMS       float64        `json:"wall_ms"`
	Timings      map[string]any `json:"timings"`
	MatchesAR    bool           `json:"matches_ar"`
	Correct      bool           `json:"correct"`
	Error        string         `json:"error,omitempty"`
}

func specHash(b []byte) string { v := sha256.Sum256(b); return hex.EncodeToString(v[:]) }
func specSave(name string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(filepath.Join(outDir, name), append(b, '\n'), 0644)
}
func specArms(mtp, dflash string) []specArm {
	return []specArm{
		{"ar", []string{"--speculative", "off"}},
		{"mtp2", []string{"--speculative", "mtp", "--mtp-model", mtp, "--draft-tokens", "2", "--min-draft-tokens", "1"}},
		{"dflash3", []string{"--speculative", "dflash2", "--dflash-model", dflash, "--draft-tokens", "3", "--draft-policy", "fixed"}},
		{"dflash7", []string{"--speculative", "dflash2", "--dflash-model", dflash, "--draft-tokens", "7", "--draft-policy", "fixed"}},
	}
}
func specStart(exe, model, label string, arm specArm) (*specServer, error) {
	if c, e := net.DialTimeout("tcp", "127.0.0.1:8081", time.Second); e == nil {
		c.Close()
		return nil, fmt.Errorf("8081 occupied; preserving listener")
	}
	args := []string{"serve", "--host", "127.0.0.1", "--port", "8081", "llm", "--model", model, "--sessions", "2", "--context", "32768", "--prefill-chunk", "2048", "--cache-ram-bytes", "off", "--think", "off", "--temperature", "0", "--seed", "42", "--presence-penalty", "0", "--repeat-penalty", "1"}
	args = append(args, "--served-model-name", "gufo")
	args = append(args, arm.Args...)
	f, e := os.Create(filepath.Join(outDir, label+"-"+arm.Name+"-server.log"))
	if e != nil {
		return nil, e
	}
	c := exec.Command(exe, args...)
	configureGufo(c)
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000}
	c.Stdout = f
	c.Stderr = f
	if e = c.Start(); e != nil {
		f.Close()
		return nil, e
	}
	s := &specServer{cmd: c, done: make(chan error, 1), log: f, Base: "http://127.0.0.1:8081"}
	go func() { s.done <- c.Wait() }()
	if e = specSave(label+"-"+arm.Name+"-launch.json", map[string]any{"exe": exe, "args": args, "pid": c.Process.Pid, "started": time.Now().Format(time.RFC3339), "memory": memory(), "sdk": "TheRock 10.0.0; configureGufo SDK/DLL paths; remaining environment inherited"}); e != nil {
		s.Stop()
		return nil, e
	}
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		select {
		case e := <-s.done:
			f.Close()
			return nil, fmt.Errorf("%s server exited: %v; see server log", arm.Name, e)
		default:
		}
		code, _, err := get(s.Base+"/health", time.Second)
		if err == nil && code == 200 {
			return s, nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	s.Stop()
	return nil, fmt.Errorf("%s startup timeout", arm.Name)
}
func (s *specServer) Stop() {
	if s == nil {
		return
	}
	_ = s.cmd.Process.Kill()
	select {
	case <-s.done:
	case <-time.After(10 * time.Second):
	}
	_ = s.log.Close()
}
func specPOST(base, path string, body any) (map[string]any, []byte, int, error) {
	b, e := json.Marshal(body)
	if e != nil {
		return nil, nil, 0, e
	}
	c := http.Client{Timeout: 3 * time.Minute}
	r, e := c.Post(base+path, "application/json", bytes.NewReader(b))
	if e != nil {
		return nil, nil, 0, e
	}
	defer r.Body.Close()
	raw, e := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if e != nil {
		return nil, raw, r.StatusCode, e
	}
	var v map[string]any
	if e = json.Unmarshal(raw, &v); e != nil {
		return nil, raw, r.StatusCode, e
	}
	return v, raw, r.StatusCode, nil
}
func specNum(m map[string]any, k string) float64 { v, _ := m[k].(float64); return v }
func specPrompt(text string) string {
	return "<|im_start|>user\n" + text + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}
func specRequest(s *specServer, label string, arm specArm, name string, rep int, prompt string, n int) (specRow, error) {
	body := map[string]any{"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 42, "presence_penalty": 0, "repeat_penalty": 1, "cache_prompt": false, "stream": false}
	start := time.Now()
	v, raw, status, e := specPOST(s.Base, "/completion", body)
	row := specRow{Arm: arm.Name, Case: name, Rep: rep, Status: status, InputHash: specHash([]byte(prompt)), WallMS: float64(time.Since(start).Microseconds()) / 1000}
	if e != nil {
		return row, e
	}
	if status != 200 {
		return row, fmt.Errorf("HTTP %d: %s", status, raw)
	}
	if e = specSave(fmt.Sprintf("%s-%s-%s-r%d-response.json", label, arm.Name, name, rep), v); e != nil {
		return row, e
	}
	row.Content, _ = v["content"].(string)
	row.OutputHash = specHash([]byte(row.Content))
	row.Timings, _ = v["timings"].(map[string]any)
	if row.Timings == nil {
		return row, fmt.Errorf("missing native timing fields")
	}
	row.PromptTokens = specNum(row.Timings, "prompt_n")
	row.Generated = specNum(row.Timings, "predicted_n")
	row.PrefillTPS = specNum(row.Timings, "prompt_per_second")
	row.DecodeTPS = specNum(row.Timings, "predicted_per_second")
	if row.Generated == 0 || row.DecodeTPS <= 0 {
		return row, fmt.Errorf("no measured output tokens")
	}
	if specNum(row.Timings, "cache_n") != 0 {
		return row, fmt.Errorf("unexpected prompt reuse")
	}
	return row, nil
}

type specScore struct {
	Arm        string  `json:"arm"`
	TPS        float64 `json:"pooled_decode_tps"`
	Gain       float64 `json:"gain_vs_ar"`
	Matches    bool    `json:"all_outputs_match_ar"`
	Generated  float64 `json:"generated_tokens"`
	Drafts     float64 `json:"draft_tokens"`
	Accepted   float64 `json:"accepted_tokens"`
	Acceptance float64 `json:"acceptance"`
}

func specScreen(args []string) error {
	if len(args) != 5 {
		return fmt.Errorf("spec-screen EXE TARGET MTP_GGUF DFLASH_GGUF LABEL")
	}
	exe, model, mtp, dflash, label := args[0], args[1], args[2], args[3], args[4]
	if e := os.MkdirAll(outDir, 0755); e != nil {
		return e
	}
	tasks := []struct{ Name, Text string }{
		{"prose", "Explain how virtual memory, page faults, working sets, and memory-mapped files interact. Give practical examples and explain common performance mistakes. Write a thorough answer of at least 700 words."},
		{"code", "Write a complete Go implementation of a bounded worker pool with context cancellation, safe channel ownership, ordered results, and error propagation. Include a worked example and explain edge cases. Give the implementation first and make the answer thorough."},
	}
	var rows []specRow
	var scores []specScore
	reference := map[string]string{}
	best := "ar"
	bestTPS := 0.0
	// Resume only completed arms from this exact local screen, preserving receipts.
	if b, e := os.ReadFile(filepath.Join(outDir, label+"-screen-summary.json")); e == nil {
		var saved struct {
			Scores []specScore `json:"scores"`
		}
		if e = json.Unmarshal(b, &saved); e != nil {
			return e
		}
		scores = saved.Scores
		b, e = os.ReadFile(filepath.Join(outDir, label+"-screen-rows.json"))
		if e != nil {
			return e
		}
		if e = json.Unmarshal(b, &rows); e != nil {
			return e
		}
		for _, r := range rows {
			if r.Arm == "ar" && r.Rep == 0 {
				reference[r.Case] = r.Content
			}
		}
		for _, s := range scores {
			if s.Matches && s.TPS > bestTPS {
				best = s.Arm
				bestTPS = s.TPS
			}
		}
	}
	for _, arm := range specArms(mtp, dflash) {
		completed := false
		for _, s := range scores {
			if s.Arm == arm.Name {
				completed = true
			}
		}
		if completed {
			continue
		}
		fmt.Printf("screen starting %s\n", arm.Name)
		s, e := specStart(exe, model, label, arm)
		if e != nil {
			specSave(label+"-failure.json", map[string]any{"arm": arm.Name, "error": e.Error()})
			return e
		}
		var score specScore
		score.Arm = arm.Name
		score.Matches = true
		e = func() error {
			defer s.Stop()
			if _, err := specRequest(s, label, arm, "warmup", 0, specPrompt("Explain virtual memory in detail."), 64); err != nil {
				return err
			}
			var decodeMS float64
			for _, task := range tasks {
				for rep := 0; rep < 2; rep++ {
					row, err := specRequest(s, label, arm, task.Name, rep, specPrompt(task.Text), 256)
					if err != nil {
						return err
					}
					if arm.Name == "ar" && rep == 0 {
						reference[task.Name] = row.Content
					}
					row.MatchesAR = row.Content == reference[task.Name]
					score.Matches = score.Matches && row.MatchesAR
					rows = append(rows, row)
					score.Generated += row.Generated
					decodeMS += specNum(row.Timings, "predicted_ms")
					score.Drafts += specNum(row.Timings, "draft_n")
					score.Accepted += specNum(row.Timings, "draft_n_accepted")
					fmt.Printf("%s %s rep=%d tokens=%.0f pp=%.1f decode=%.2f drafted=%.0f accepted=%.0f matches_ar=%v\n", arm.Name, task.Name, rep, row.Generated, row.PrefillTPS, row.DecodeTPS, specNum(row.Timings, "draft_n"), specNum(row.Timings, "draft_n_accepted"), row.MatchesAR)
					if err = specSave(label+"-screen-rows.json", rows); err != nil {
						return err
					}
				}
			}
			score.TPS = score.Generated * 1000 / decodeMS
			if score.Drafts > 0 {
				score.Acceptance = score.Accepted / score.Drafts
			}
			return nil
		}()
		if e != nil {
			return e
		}
		if len(scores) == 0 {
			score.Gain = 1
		} else {
			score.Gain = score.TPS / scores[0].TPS
		}
		scores = append(scores, score)
		if score.Matches && score.TPS > bestTPS {
			best = arm.Name
			bestTPS = score.TPS
		}
		if e = specSave(label+"-screen-summary.json", map[string]any{"scores": scores, "winner": best, "protocol": "one excluded warmup; 2 natural tasks x 2 measured repetitions; greedy; no cache; 256 output cap; fixed binary; fresh process per arm; sequential servers"}); e != nil {
			return e
		}
	}
	fmt.Printf("screen selected %s pooled_decode=%.2f\n", best, bestTPS)
	// Every speculative response must match its AR reference before adoption.
	for _, score := range scores {
		if !score.Matches {
			return fmt.Errorf("%s changed greedy output; preserve evidence and investigate before adoption", score.Arm)
		}
	}
	return nil
}

// Trim fences for a structural JSON check, without modifying stored output.
func specJSONContent(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		if i := strings.Index(s, "\n"); i >= 0 {
			s = s[i+1:]
			s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
		}
	}
	return s
}

func specAudit(args []string) error {
	if len(args) != 3 {
		return fmt.Errorf("spec-audit SCREEN_LABEL VALIDATION_LABEL SCHEMA_LABEL")
	}
	loadRows := func(label, suffix string) ([]specRow, error) {
		b, e := os.ReadFile(filepath.Join(outDir, label+suffix))
		if e != nil {
			return nil, e
		}
		var rows []specRow
		e = json.Unmarshal(b, &rows)
		return rows, e
	}
	screen, e := loadRows(args[0], "-screen-rows.json")
	if e != nil {
		return e
	}
	validation, e := loadRows(args[1], "-rows.json")
	if e != nil {
		return e
	}
	schema, e := loadRows(args[2], "-rows.json")
	if e != nil {
		return e
	}
	mismatches := []string{}
	inputMismatches := []string{}
	failures := []string{}
	allCacheZero := true
	inspect := func(rows []specRow, quality bool) {
		refs := map[string]specRow{}
		for _, r := range rows {
			if r.Arm == "ar" && r.Rep == 0 {
				refs[r.Case] = r
			}
		}
		for _, r := range rows {
			if specNum(r.Timings, "cache_n") != 0 {
				allCacheZero = false
			}
			if quality && !r.Correct {
				failures = append(failures, r.Arm+":"+r.Case)
			}
			if r.OutputHash != refs[r.Case].OutputHash {
				mismatches = append(mismatches, r.Arm+":"+r.Case)
			}
			if r.InputHash != refs[r.Case].InputHash {
				inputMismatches = append(inputMismatches, r.Arm+":"+r.Case)
			}
		}
	}
	inspect(screen, false)
	inspect(validation, true)
	inspect(schema, true)
	type totals struct{ Tokens, Millis, Drafts, Accepted float64 }
	perCase := map[string]map[string]totals{}
	for _, r := range screen {
		if perCase[r.Arm] == nil {
			perCase[r.Arm] = map[string]totals{}
		}
		v := perCase[r.Arm][r.Case]
		v.Tokens += r.Generated
		v.Millis += specNum(r.Timings, "predicted_ms")
		v.Drafts += specNum(r.Timings, "draft_n")
		v.Accepted += specNum(r.Timings, "draft_n_accepted")
		perCase[r.Arm][r.Case] = v
	}
	metrics := map[string]any{}
	for arm, cases := range perCase {
		out := map[string]any{}
		for name, v := range cases {
			a := 0.0
			if v.Drafts > 0 {
				a = v.Accepted / v.Drafts
			}
			out[name] = map[string]any{"pooled_tps": v.Tokens * 1000 / v.Millis, "acceptance": a}
		}
		metrics[arm] = out
	}
	gpu := map[string]int{}
	r := regexp.MustCompile(`gpu_device_used_mib=(\d+)`)
	for _, arm := range []string{"ar", "mtp2", "dflash3", "dflash7"} {
		b, e := os.ReadFile(filepath.Join(outDir, args[0]+"-"+arm+"-server.log"))
		if e != nil {
			return e
		}
		m := r.FindSubmatch(b)
		if len(m) > 1 {
			gpu[arm], _ = strconv.Atoi(string(m[1]))
		}
	}
	result := map[string]any{"screen_rows": len(screen), "validation_rows": len(validation), "structured_rows": len(schema), "output_mismatches": mismatches, "input_hash_differences": inputMismatches, "strict_quality_failures_retained": failures, "all_cached_tokens_zero": allCacheZero, "natural_workload_metrics": metrics, "startup_gpu_device_used_mib": gpu, "gpu_memory_note": "driver-reported startup snapshots, not peak process RAM or a dual-model residency certification"}
	if e = specSave(args[0]+"-audit.json", result); e != nil {
		return e
	}
	b, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(b))
	return nil
}

// Revalidate the lossless standalone head against the original four MTP rows.
func specIsolatedMTPVerify(args []string) error {
 if len(args)!=3{return fmt.Errorf("isolated-mtp-verify EXE IQ_TARGET STANDALONE_MTP")}
 const original=`C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
 const sidecar=`C:\Users\Butch\Documents\10-Models\models\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`
 if !strings.EqualFold(filepath.Clean(args[2]),sidecar){return fmt.Errorf("unexpected standalone MTP path")}
 if _,e:=os.Stat(original);!os.IsNotExist(e){return fmt.Errorf("original source must be held unavailable before independence test")}
 b,e:=os.ReadFile(filepath.Join(outDir,"iq-spec-natural-screen-rows.json"));if e!=nil{return e};var old []specRow;if e=json.Unmarshal(b,&old);e!=nil{return e}
 refs:=map[string]specRow{};for _,r:=range old{if r.Arm=="mtp2"{refs[fmt.Sprintf("%s:%d",r.Case,r.Rep)]=r}};if len(refs)!=4{return fmt.Errorf("missing original measured MTP reference rows")}
 label:="isolated-mtp-runtime";arm:=specArms(args[2],"")[1];s,e:=specStart(args[0],args[1],label,arm);if e!=nil{return e};defer s.Stop()
 if _,e=specRequest(s,label,arm,"warmup",0,specPrompt("Explain virtual memory in detail."),64);e!=nil{return e}
 tasks:=[]struct{Name,Text string}{{"prose","Explain how virtual memory, page faults, working sets, and memory-mapped files interact. Give practical examples and explain common performance mistakes. Write a thorough answer of at least 700 words."},{"code","Write a complete Go implementation of a bounded worker pool with context cancellation, safe channel ownership, ordered results, and error propagation. Include a worked example and explain edge cases. Give the implementation first and make the answer thorough."}}
 rows:=[]specRow{};matches,counts:=true,true;var tokens,ms,drafts,accepted float64
 for _,task:=range tasks{for rep:=0;rep<2;rep++{
  row,e:=specRequest(s,label,arm,task.Name,rep,specPrompt(task.Text),256);if e!=nil{return e};ref:=refs[fmt.Sprintf("%s:%d",task.Name,rep)]
  row.MatchesAR=row.OutputHash==ref.OutputHash&&row.InputHash==ref.InputHash&&row.Generated==ref.Generated;matches=matches&&row.MatchesAR
  counts=counts&&specNum(row.Timings,"draft_n")==specNum(ref.Timings,"draft_n")&&specNum(row.Timings,"draft_n_accepted")==specNum(ref.Timings,"draft_n_accepted")&&specNum(row.Timings,"draft_n")>0
  tokens+=row.Generated;ms+=specNum(row.Timings,"predicted_ms");drafts+=specNum(row.Timings,"draft_n");accepted+=specNum(row.Timings,"draft_n_accepted");rows=append(rows,row)
  fmt.Printf("isolated MTP %s rep=%d decode=%.2f drafted=%.0f accepted=%.0f matches_original=%v\n",task.Name,rep,row.DecodeTPS,specNum(row.Timings,"draft_n"),specNum(row.Timings,"draft_n_accepted"),row.MatchesAR)
 }}
 if e=specSave(label+"-rows.json",rows);e!=nil{return e}
 if e=specSave(label+"-summary.json",map[string]any{"all_outputs_match_original":matches,"draft_counts_match_original":counts,"old_source_unavailable":true,"standalone_mtp":args[2],"measured_rows":len(rows),"pooled_decode_tps":tokens*1000/ms,"drafted":drafts,"accepted":accepted,"acceptance":accepted/drafts});e!=nil{return e}
 if !matches||!counts{return fmt.Errorf("standalone runtime gate failed")};return nil
}
