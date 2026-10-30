package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type specFixture struct {
	Name     string         `json:"name"`
	Text     string         `json:"text"`
	Expected map[string]any `json:"expected"`
	Count    int            `json:"counted_input_tokens"`
}

func specMessages(text string) []any { return []any{map[string]any{"role": "user", "content": text}} }
func specChat(s *specServer, label string, arm specArm, name string, messages []any, tools []any, n int, formats ...map[string]any) (specRow, map[string]any, error) {
	body := map[string]any{"model": "gufo", "messages": messages, "max_tokens": n, "temperature": 0, "seed": 42, "presence_penalty": 0, "repeat_penalty": 1, "cache_prompt": false, "stream": false, "chat_template_kwargs": map[string]any{"enable_thinking": false}}
	if len(tools) > 0 {
		body["tools"] = tools
		body["tool_choice"] = "auto"
	}
	if len(formats) == 1 {
		body["response_format"] = formats[0]
	}
	encoded, _ := json.Marshal(body)
	start := time.Now()
	v, raw, status, e := specPOST(s.Base, "/v1/chat/completions", body)
	row := specRow{Arm: arm.Name, Case: name, Status: status, InputHash: specHash(encoded), WallMS: float64(time.Since(start).Microseconds()) / 1000}
	if e != nil {
		return row, nil, e
	}
	if status != 200 {
		return row, nil, fmt.Errorf("HTTP %d: %s", status, raw)
	}
	if e = specSave(label+"-"+arm.Name+"-"+name+"-response.json", v); e != nil {
		return row, nil, e
	}
	choices, ok := v["choices"].([]any)
	if !ok || len(choices) != 1 {
		return row, nil, fmt.Errorf("expected one choice")
	}
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	if msg == nil {
		return row, nil, fmt.Errorf("missing assistant message")
	}
	row.Content, _ = msg["content"].(string)
	canonical := row.Content
	if calls, ok := msg["tool_calls"].([]any); ok && len(calls) > 0 {
		functions := []any{}
		for _, c := range calls {
			call, _ := c.(map[string]any)
			functions = append(functions, call["function"])
		}
		b, _ := json.Marshal(functions)
		canonical += "\nTOOLS:" + string(b)
	}
	row.OutputHash = specHash([]byte(canonical))
	row.Timings, _ = v["timings"].(map[string]any)
	if row.Timings == nil {
		return row, nil, fmt.Errorf("missing native chat timings")
	}
	row.PromptTokens = specNum(row.Timings, "prompt_n")
	row.Generated = specNum(row.Timings, "predicted_n")
	row.PrefillTPS = specNum(row.Timings, "prompt_per_second")
	row.DecodeTPS = specNum(row.Timings, "predicted_per_second")
	return row, msg, nil
}

func specFormat(name string, properties map[string]any) map[string]any {
	required := []string{}
	for key := range properties {
		required = append(required, key)
	}
	sort.Strings(required)
	// Only constrain structure/types, never the expected answer values.
	return map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": name, "strict": true, "schema": map[string]any{"type": "object", "properties": properties, "required": required, "additionalProperties": false}}}
}
func specJSONCheck(args []string) error {
	if len(args) != 6 {
		return fmt.Errorf("spec-json-check EXE TARGET MTP_GGUF DFLASH_GGUF LABEL MODE")
	}
	exe, model, mtp, dflash, label, mode := args[0], args[1], args[2], args[3], args[4], args[5]
	arms := specArms(mtp, dflash)
	selected := specArm{}
	for _, a := range arms {
		if a.Name == mode {
			selected = a
		}
	}
	if selected.Name == "" {
		return fmt.Errorf("unknown mode")
	}
	reference := map[string]string{}
	rows := []specRow{}
	add := func(row specRow, good bool) {
		row.Correct = good
		if row.Arm == "ar" {
			reference[row.Case] = row.OutputHash
		}
		row.MatchesAR = row.OutputHash == reference[row.Case]
		rows = append(rows, row)
		fmt.Printf("structured %s %s correct=%v matches_ar=%v decode=%.1f drafted=%.0f accepted=%.0f\n", row.Arm, row.Case, row.Correct, row.MatchesAR, row.DecodeTPS, specNum(row.Timings, "draft_n"), specNum(row.Timings, "draft_n_accepted"))
	}
	for _, arm := range []specArm{arms[0], selected} {
		s, e := specStart(exe, model, label, arm)
		if e != nil {
			return e
		}
		e = func() error {
			defer s.Stop()
			multi := []any{map[string]any{"role": "system", "content": "Track the latest user preference. Return only the requested JSON."}, map[string]any{"role": "user", "content": "Remember my preferred color is blue."}, map[string]any{"role": "assistant", "content": "Your preferred color is blue."}, map[string]any{"role": "user", "content": "Change my preferred color to green."}, map[string]any{"role": "assistant", "content": "Your preferred color is now green."}, map[string]any{"role": "user", "content": "Return only JSON with my current color in the key color."}}
			row, _, e := specChat(s, label, arm, "multi_turn", multi, nil, 64, specFormat("color", map[string]any{"color": map[string]any{"type": "string"}}))
			if e != nil {
				return e
			}
			add(row, specJSONMatches(row.Content, map[string]any{"color": "green"}))
			messages := []any{map[string]any{"role": "system", "content": "Use lookup_invoice for invoice facts. Do not guess records. After the tool result, return only JSON with invoice_id, total_cents, and status."}, map[string]any{"role": "user", "content": "Look up invoice INV-1042 and report its invoice_id, total_cents, and status."}}
			row, msg, e := specChat(s, label, arm, "tool_call", messages, specTool(), 192)
			if e != nil {
				return e
			}
			calls, _ := msg["tool_calls"].([]any)
			good := false
			var call map[string]any
			if len(calls) == 1 {
				call, _ = calls[0].(map[string]any)
				f, _ := call["function"].(map[string]any)
				if f != nil {
					a, _ := f["arguments"].(string)
					good = f["name"] == "lookup_invoice" && specJSONMatches(a, map[string]any{"invoice_id": "INV-1042"})
				}
			}
			add(row, good)
			if !good {
				return fmt.Errorf("tool selection failed before structured continuation")
			}
			call["id"] = "call_invoice_1042"
			messages = append(messages, map[string]any{"role": "assistant", "content": msg["content"], "tool_calls": calls}, map[string]any{"role": "tool", "tool_call_id": call["id"], "content": `{"invoice_id":"INV-1042","subtotal_cents":4200,"tax_cents":336,"total_cents":4536,"status":"paid"}`})
			format := specFormat("invoice", map[string]any{"invoice_id": map[string]any{"type": "string"}, "total_cents": map[string]any{"type": "integer"}, "status": map[string]any{"type": "string"}})
			row, _, e = specChat(s, label, arm, "tool_result", messages, specTool(), 128, format)
			if e != nil {
				return e
			}
			add(row, specJSONMatches(row.Content, map[string]any{"invoice_id": "INV-1042", "total_cents": 4536, "status": "paid"}))
			var wg sync.WaitGroup
			parallel := make([]specRow, 2)
			errs := make([]error, 2)
			expected := make([]map[string]any, 2)
			format = specFormat("session", map[string]any{"nonce": map[string]any{"type": "string"}, "sequence": map[string]any{"type": "array", "items": map[string]any{"type": "integer"}}})
			for i, nonce := range []string{"ORANGE-18473", "PURPLE-92816"} {
				sequence := []int{}
				for n := 1; n <= 32; n++ {
					sequence = append(sequence, n)
				}
				expected[i] = map[string]any{"nonce": nonce, "sequence": sequence}
				text := fmt.Sprintf("Your private session nonce is %s. Return only JSON with keys nonce and sequence. Copy the nonce exactly and put all integers from 1 through 32 inclusive in sequence. Do not use any other session data.", nonce)
				wg.Add(1)
				go func(i int, text string) {
					defer wg.Done()
					parallel[i], _, errs[i] = specChat(s, label, arm, fmt.Sprintf("parallel_%d", i), specMessages(text), nil, 256, format)
				}(i, text)
			}
			wg.Wait()
			for i, row := range parallel {
				if errs[i] != nil {
					return errs[i]
				}
				add(row, specJSONMatches(row.Content, expected[i]))
			}
			return specSave(label+"-rows.json", rows)
		}()
		if e != nil {
			return e
		}
	}
	good, parity := true, true
	for _, row := range rows {
		good = good && row.Correct
		parity = parity && row.MatchesAR
	}
	if e := specSave(label+"-summary.json", map[string]any{"pure_json_and_known_answers_pass": good, "matches_ar": parity, "rows": len(rows), "notes": "Explicit response_format json_schema; original unconstrained-format failures retained; constraints specify types/keys only, not answer values; tool call itself remains auto/unconstrained"}); e != nil {
		return e
	}
	if !good || !parity {
		return fmt.Errorf("structured JSON check failed")
	}
	return nil
}
func specJSONMatches(content string, expected map[string]any) bool {
	var actual map[string]any
	// Require pure JSON; Markdown fences and explanatory prose fail this check.
	if json.Unmarshal([]byte(strings.TrimSpace(content)), &actual) != nil {
		return false
	}
	a, _ := json.Marshal(actual)
	b, _ := json.Marshal(expected)
	return string(a) == string(b)
}
func specCount(s *specServer, text string) (int, error) {
	v, raw, status, e := specPOST(s.Base, "/v1/messages/count_tokens", map[string]any{"model": "gufo", "messages": specMessages(text)})
	if e != nil {
		return 0, e
	}
	if status != 200 {
		return 0, fmt.Errorf("count tokens HTTP %d: %s", status, raw)
	}
	n := int(specNum(v, "input_tokens"))
	if n < 1 {
		return 0, fmt.Errorf("missing input token count")
	}
	return n, nil
}
func specArchive(records int) string {
	var b strings.Builder
	b.WriteString("Read the complete archive. Three named control records contain access codes. Ignore ordinary archive records.\nControl ALPHA has access code AX-7319.\n")
	if records == 0 {
		b.WriteString("Control BETA has access code BY-2486.\n")
	}
	for i := 0; i < records; i++ {
		if i == records/2 {
			b.WriteString("Control BETA has access code BY-2486.\n")
		}
		fmt.Fprintf(&b, "Archive entry %05d: This ordinary record describes scheduled maintenance, delivery windows, assigned workers, stored equipment, and routine status updates.\n", i)
	}
	b.WriteString("Control GAMMA has access code CZ-9054.\nReturn only a JSON object with keys alpha, beta, gamma and the exact access codes from the three control records. No explanation.\n")
	return b.String()
}
func specFixtureFor(s *specServer, name string, target int) (specFixture, error) {
	records := int(math.Max(0, float64(target-150)/30))
	var text string
	var n int
	for i := 0; i < 6; i++ {
		text = specArchive(records)
		var e error
		n, e = specCount(s, text)
		if e != nil {
			return specFixture{}, e
		}
		if abs := n - target; abs >= -32 && abs <= 32 {
			break
		}
		next := int(float64(records) * float64(target-150) / float64(max(n-150, 1)))
		if next == records {
			break
		}
		records = max(next, 0)
	}
	if n+128 > 32768 {
		return specFixture{}, fmt.Errorf("fixture %s would overflow context: %d", name, n)
	}
	return specFixture{Name: name, Text: text, Count: n, Expected: map[string]any{"alpha": "AX-7319", "beta": "BY-2486", "gamma": "CZ-9054"}}, nil
}
func specTool() []any {
	return []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup_invoice", "description": "Retrieve the authoritative invoice record by invoice_id. Use this to answer questions about invoices.", "parameters": map[string]any{"type": "object", "properties": map[string]any{"invoice_id": map[string]any{"type": "string"}}, "required": []string{"invoice_id"}, "additionalProperties": false}}}}
}
func specValidate(args []string) error {
	if len(args) != 6 {
		return fmt.Errorf("spec-validate EXE TARGET MTP_GGUF DFLASH_GGUF LABEL MODE")
	}
	exe, model, mtp, dflash, label, mode := args[0], args[1], args[2], args[3], args[4], args[5]
	selected := specArm{}
	for _, arm := range specArms(mtp, dflash) {
		if arm.Name == mode {
			selected = arm
		}
	}
	if selected.Name == "" || selected.Name == "ar" {
		return fmt.Errorf("select a speculative mode for AR comparison")
	}
	if e := os.MkdirAll(outDir, 0755); e != nil {
		return e
	}
	arms := []specArm{specArms(mtp, dflash)[0], selected}
	reference := map[string]string{}
	var rows []specRow
	var fixtures []specFixture
	appendRow := func(row specRow, good bool, reason string) {
		row.Correct = good
		row.Error = reason
		if row.Arm == "ar" {
			reference[row.Case] = row.OutputHash
		}
		row.MatchesAR = row.OutputHash == reference[row.Case]
		rows = append(rows, row)
		fmt.Printf("validate %s %s input=%.0f output=%.0f correct=%v matches_ar=%v pp=%.1f decode=%.1f %s\n", row.Arm, row.Case, row.PromptTokens, row.Generated, row.Correct, row.MatchesAR, row.PrefillTPS, row.DecodeTPS, reason)
	}
	for _, arm := range arms {
		fmt.Printf("validation starting %s\n", arm.Name)
		s, e := specStart(exe, model, label, arm)
		if e != nil {
			return e
		}
		e = func() error {
			defer s.Stop()
			if _, _, e := specChat(s, label, arm, "warmup", specMessages("Reply with the word READY."), nil, 16); e != nil {
				return e
			}
			if arm.Name == "ar" {
				fixtures = append(fixtures, specFixture{Name: "short_json", Text: "Return only JSON with keys product and answer. product is 17 multiplied by 23; answer is the capital city of Italy. No explanation.", Expected: map[string]any{"product": 391, "answer": "Rome"}})
				for _, v := range []struct {
					Name string
					N    int
				}{{"short128", 128}, {"medium2k", 2048}, {"long8k", 8192}, {"long16k", 16384}, {"long32k", 32000}} {
					f, e := specFixtureFor(s, v.Name, v.N)
					if e != nil {
						return e
					}
					fixtures = append(fixtures, f)
				}
				if e := specSave(label+"-fixtures.json", fixtures); e != nil {
					return e
				}
			}
			for _, f := range fixtures {
				row, _, e := specChat(s, label, arm, f.Name, specMessages(f.Text), nil, 128)
				if e != nil {
					return e
				}
				good := specJSONMatches(row.Content, f.Expected)
				reason := ""
				if !good {
					reason = "invalid JSON or wrong known answer"
				}
				appendRow(row, good, reason)
			}
			messages := []any{map[string]any{"role": "system", "content": "Track the latest user preference. Return only the requested JSON."}, map[string]any{"role": "user", "content": "Remember my preferred color is blue."}, map[string]any{"role": "assistant", "content": "Your preferred color is blue."}, map[string]any{"role": "user", "content": "Change my preferred color to green."}, map[string]any{"role": "assistant", "content": "Your preferred color is now green."}, map[string]any{"role": "user", "content": "Return only JSON with my current color in the key color."}}
			row, _, e := specChat(s, label, arm, "multi_turn", messages, nil, 64)
			if e != nil {
				return e
			}
			good := specJSONMatches(row.Content, map[string]any{"color": "green"})
			appendRow(row, good, "")
			// A local mock supplies data; no real invoice service or external tool runs.
			toolMessages := []any{map[string]any{"role": "system", "content": "Use lookup_invoice for invoice facts. Do not guess records. After the tool result, return only JSON with invoice_id, total_cents, and status."}, map[string]any{"role": "user", "content": "Look up invoice INV-1042 and report its invoice_id, total_cents, and status."}}
			row, msg, e := specChat(s, label, arm, "tool_call", toolMessages, specTool(), 192)
			if e != nil {
				return e
			}
			calls, _ := msg["tool_calls"].([]any)
			var call map[string]any
			var function map[string]any
			if len(calls) == 1 {
				call, _ = calls[0].(map[string]any)
				function, _ = call["function"].(map[string]any)
			}
			good = false
			if function != nil {
				a, _ := function["arguments"].(string)
				good = function["name"] == "lookup_invoice" && specJSONMatches(a, map[string]any{"invoice_id": "INV-1042"})
			}
			appendRow(row, good, "")
			if good {
				// Keep the synthetic tool-result transcript identical between arms.
				call["id"] = "call_invoice_1042"
				toolMessages = append(toolMessages, map[string]any{"role": "assistant", "content": msg["content"], "tool_calls": calls}, map[string]any{"role": "tool", "tool_call_id": call["id"], "content": `{"invoice_id":"INV-1042","subtotal_cents":4200,"tax_cents":336,"total_cents":4536,"status":"paid"}`})
				row, _, e = specChat(s, label, arm, "tool_result", toolMessages, specTool(), 128)
				if e != nil {
					return e
				}
				good = specJSONMatches(row.Content, map[string]any{"invoice_id": "INV-1042", "total_cents": 4536, "status": "paid"})
				appendRow(row, good, "")
			}
			row, msg, e = specChat(s, label, arm, "tool_not_needed", specMessages("No invoice information is needed. Return only JSON with the answer to 2+2 in key answer."), specTool(), 64)
			if e != nil {
				return e
			}
			calls, _ = msg["tool_calls"].([]any)
			good = len(calls) == 0 && specJSONMatches(row.Content, map[string]any{"answer": 4})
			appendRow(row, good, "")
			// Two simultaneous requests: distinct nonces expose session contamination.
			var wg sync.WaitGroup
			parallel := make([]specRow, 2)
			errs := make([]error, 2)
			expected := make([]map[string]any, 2)
			for i, nonce := range []string{"ORANGE-18473", "PURPLE-92816"} {
				sequence := []int{}
				for n := 1; n <= 32; n++ {
					sequence = append(sequence, n)
				}
				expected[i] = map[string]any{"nonce": nonce, "sequence": sequence}
				text := fmt.Sprintf("Your private session nonce is %s. Return only JSON with keys nonce and sequence. Copy the nonce exactly and put all integers from 1 through 32 inclusive in sequence. Do not use any other session data.", nonce)
				wg.Add(1)
				go func(i int, text string) {
					defer wg.Done()
					parallel[i], _, errs[i] = specChat(s, label, arm, fmt.Sprintf("parallel_%d", i), specMessages(text), nil, 256)
				}(i, text)
			}
			wg.Wait()
			for i, row := range parallel {
				if errs[i] != nil {
					return errs[i]
				}
				good = specJSONMatches(row.Content, expected[i])
				appendRow(row, good, "")
			}
			return specSave(label+"-rows.json", rows)
		}()
		if e != nil {
			return e
		}
	}
	correct, parity := true, true
	for _, r := range rows {
		correct = correct && r.Correct
		parity = parity && r.MatchesAR
	}
	summary := map[string]any{"all_known_answer_checks_pass": correct, "all_spec_outputs_match_ar": parity, "rows": len(rows), "speculative_mode": mode, "protocol": "same IQ4_XS and prefill ON; plain decode vs selected speculation; short through ~32K context, pure JSON, full multi-turn transcript, auto tool call and local mock result, no-tool control, 2 concurrent requests; no Strata launch"}
	if e := specSave(label+"-summary.json", summary); e != nil {
		return e
	}
	fmt.Printf("validation summary known_answers=%v exact_ar_match=%v rows=%d\n", correct, parity, len(rows))
	if !correct || !parity {
		return fmt.Errorf("validation gate failed; inspect retained responses before adoption")
	}
	return nil
}
