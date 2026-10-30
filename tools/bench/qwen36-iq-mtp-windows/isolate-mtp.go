// Lossless GGUF subset writer and narrowly scoped source retirement tool.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

const mtpFolder = `C:\Users\Butch\Documents\10-Models\models`
const mtpSource = mtpFolder + `\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
const mtpHeld = mtpSource + `.pending-delete-isolated-mtp`
const mtpSubset = mtpFolder + `\Qwen3.6-35B-A3B-MTP-only-UD-Q4_K_XL.gguf`
const mtpResults = `C:\Users\Butch\Documents\02_Apps\strycast-bench-tools\results`
const mtpReceipt = mtpResults + `\isolated-mtp-extraction.json`

type mtpKV struct {
	Name string
	Raw  []byte
}
type mtpTensor struct {
	Name   string   `json:"name"`
	Dims   []uint64 `json:"dims"`
	Type   uint32   `json:"ggml_type"`
	Offset uint64   `json:"offset"`
	Bytes  uint64   `json:"bytes"`
	SHA    string   `json:"payload_sha256,omitempty"`
}
type mtpGGUF struct {
	KV           []mtpKV
	Tensors      []mtpTensor
	Alignment    uint64
	Data         uint64
	Architecture string
	Layers       uint64
	MTP          uint64
	MetadataSHA  string
}
type mtpRecord struct {
	Source      string      `json:"source"`
	SourceBytes int64       `json:"source_bytes"`
	SourceID    string      `json:"source_file_id"`
	Subset      string      `json:"subset"`
	SubsetBytes int64       `json:"subset_bytes"`
	SubsetSHA   string      `json:"subset_sha256"`
	MetadataSHA string      `json:"metadata_sha256"`
	Tensors     []mtpTensor `json:"tensors"`
	Verified    bool        `json:"verified"`
	Time        string      `json:"time"`
}
type mtpRead struct{ f *os.File }

func (r mtpRead) num(v any) {
	if e := binary.Read(r.f, binary.LittleEndian, v); e != nil {
		panic(e)
	}
}
func (r mtpRead) pos() uint64 {
	n, e := r.f.Seek(0, io.SeekCurrent)
	if e != nil {
		panic(e)
	}
	return uint64(n)
}
func (r mtpRead) skip(n uint64) {
	if n > 1<<40 {
		panic("invalid metadata length")
	}
	if _, e := r.f.Seek(int64(n), io.SeekCurrent); e != nil {
		panic(e)
	}
}
func (r mtpRead) str() string {
	var n uint64
	r.num(&n)
	if n > 64<<20 {
		panic("oversized string")
	}
	b := make([]byte, n)
	if _, e := io.ReadFull(r.f, b); e != nil {
		panic(e)
	}
	return string(b)
}
func (r mtpRead) value(t uint32, keep bool) any {
	switch t {
	case 8:
		s := r.str()
		if keep {
			return s
		}
	case 9:
		var sub uint32
		var n uint64
		r.num(&sub)
		r.num(&n)
		if n > 10000000 {
			panic("oversized array")
		}
		sizes := map[uint32]uint64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
		if size, ok := sizes[sub]; ok {
			r.skip(n * size)
		} else {
			for i := uint64(0); i < n; i++ {
				r.value(sub, false)
			}
		}
	case 0, 1, 7:
		var v uint8
		r.num(&v)
		if keep {
			return uint64(v)
		}
	case 2, 3:
		var v uint16
		r.num(&v)
		if keep {
			return uint64(v)
		}
	case 4, 5, 6:
		var v uint32
		r.num(&v)
		if keep {
			return uint64(v)
		}
	case 10, 11, 12:
		var v uint64
		r.num(&v)
		if keep {
			return v
		}
	default:
		panic(fmt.Sprintf("unsupported metadata type %d", t))
	}
	return nil
}
func mtpAlign(n, a uint64) uint64 {
	if a == 0 || a&(a-1) != 0 || n > math.MaxUint64-(a-1) {
		panic("bad alignment")
	}
	return (n + a - 1) &^ (a - 1)
}
func mtpSize(t mtpTensor) uint64 {
	formats := map[uint32][2]uint64{0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20}, 6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176}, 14: {256, 210}, 15: {256, 292}, 20: {32, 18}, 21: {256, 110}, 23: {256, 136}, 30: {1, 2}}
	shape, ok := formats[t.Type]
	if !ok {
		panic(fmt.Sprintf("unsupported tensor encoding %d", t.Type))
	}
	n := uint64(1)
	for _, v := range t.Dims {
		if v == 0 || n > math.MaxUint64/v {
			panic("invalid tensor dimensions")
		}
		n *= v
	}
	if n%shape[0] != 0 {
		panic("tensor not whole blocks")
	}
	return n / shape[0] * shape[1]
}
func mtpParse(path string) mtpGGUF {
	f, e := os.Open(path)
	if e != nil {
		panic(e)
	}
	defer f.Close()
	r := mtpRead{f}
	var magic, version uint32
	var nt, nk uint64
	r.num(&magic)
	r.num(&version)
	r.num(&nt)
	r.num(&nk)
	if magic != 0x46554747 || version != 3 || nt > 100000 || nk > 100000 {
		panic("not supported GGUF v3")
	}
	g := mtpGGUF{Alignment: 32}
	meta := sha256.New()
	values := map[string]any{}
	for i := uint64(0); i < nk; i++ {
		start := r.pos()
		key := r.str()
		var typ uint32
		r.num(&typ)
		keep := key == "general.architecture" || key == "general.alignment" || strings.HasSuffix(key, ".block_count") || strings.HasSuffix(key, ".nextn_predict_layers")
		v := r.value(typ, keep)
		if keep {
			values[key] = v
		}
		end := r.pos()
		raw := make([]byte, end-start)
		if _, e = f.ReadAt(raw, int64(start)); e != nil {
			panic(e)
		}
		if !strings.HasPrefix(key, "split.") {
			g.KV = append(g.KV, mtpKV{key, raw})
			meta.Write(raw)
		}
	}
	g.Architecture, _ = values["general.architecture"].(string)
	g.Layers, _ = values[g.Architecture+".block_count"].(uint64)
	g.MTP, _ = values[g.Architecture+".nextn_predict_layers"].(uint64)
	if g.MTP > g.Layers {
		panic("MTP count exceeds block count")
	}
	g.Layers -= g.MTP
	if a, ok := values["general.alignment"].(uint64); ok {
		g.Alignment = a
	}
	if g.Alignment > 1<<20 {
		panic("oversized alignment")
	}
	mtpAlign(0, g.Alignment)
	g.MetadataSHA = hex.EncodeToString(meta.Sum(nil))
	for i := uint64(0); i < nt; i++ {
		var t mtpTensor
		t.Name = r.str()
		var nd uint32
		r.num(&nd)
		if nd == 0 || nd > 8 {
			panic("invalid tensor rank")
		}
		t.Dims = make([]uint64, nd)
		for j := range t.Dims {
			r.num(&t.Dims[j])
		}
		r.num(&t.Type)
		r.num(&t.Offset)
		t.Bytes = mtpSize(t)
		g.Tensors = append(g.Tensors, t)
	}
	g.Data = mtpAlign(r.pos(), g.Alignment)
	st, e := f.Stat()
	if e != nil {
		panic(e)
	}
	seen := map[string]bool{}
	for _, t := range g.Tensors {
		if seen[t.Name] || g.Data+t.Offset > uint64(st.Size()) || t.Bytes > uint64(st.Size())-g.Data-t.Offset {
			panic("invalid tensor directory")
		}
		seen[t.Name] = true
	}
	return g
}
func mtpSelect(g mtpGGUF) []mtpTensor {
	if g.Architecture != "qwen35moe" || g.Layers != 40 || g.MTP != 1 {
		panic(fmt.Sprintf("unexpected model architecture/layer count: %s %d MTP %d", g.Architecture, g.Layers, g.MTP))
	}
	var out []mtpTensor
	shared := 0
	head := 0
	for _, t := range g.Tensors {
		if t.Name == "token_embd.weight" || t.Name == "output.weight" {
			shared++
			out = append(out, t)
		} else if strings.HasPrefix(t.Name, "blk.40.") {
			head++
			out = append(out, t)
		}
	}
	if shared != 2 || head < 16 {
		panic("missing required shared/head tensors")
	}
	return out
}
func mtpFileID(path string) string {
	f, e := os.Open(path)
	if e != nil {
		panic(e)
	}
	defer f.Close()
	var info syscall.ByHandleFileInformation
	if e = syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &info); e != nil {
		panic(e)
	}
	return fmt.Sprintf("%08x:%08x%08x", info.VolumeSerialNumber, info.FileIndexHigh, info.FileIndexLow)
}
func mtpDigest(path string) string {
	f, e := os.Open(path)
	if e != nil {
		panic(e)
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		panic(e)
	}
	return hex.EncodeToString(h.Sum(nil))
}
func mtpJSON(path string, v any) {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		panic(e)
	}
	if e = os.WriteFile(path, append(b, '\n'), 0644); e != nil {
		panic(e)
	}
}
func mtpReadRecord() mtpRecord {
	b, e := os.ReadFile(mtpReceipt)
	if e != nil {
		panic(e)
	}
	var r mtpRecord
	if e = json.Unmarshal(b, &r); e != nil {
		panic(e)
	}
	return r
}
func mtpWrite(f *os.File, v any) {
	if e := binary.Write(f, binary.LittleEndian, v); e != nil {
		panic(e)
	}
}
func mtpString(f *os.File, s string) {
	mtpWrite(f, uint64(len(s)))
	if _, e := io.WriteString(f, s); e != nil {
		panic(e)
	}
}
func mtpPad(f *os.File, a uint64) {
	n, e := f.Seek(0, io.SeekCurrent)
	if e != nil {
		panic(e)
	}
	pad := mtpAlign(uint64(n), a) - uint64(n)
	if _, e = f.Write(make([]byte, pad)); e != nil {
		panic(e)
	}
}
func mtpExtract() {
	g := mtpParse(mtpSource)
	selected := mtpSelect(g)
	st, e := os.Stat(mtpSource)
	if e != nil {
		panic(e)
	}
	if st.Size() != 22853663008 {
		panic("unexpected source file size")
	}
	if _, e = os.Lstat(mtpSubset); !os.IsNotExist(e) {
		panic("destination already exists; preserving it")
	}
	src, e := os.Open(mtpSource)
	if e != nil {
		panic(e)
	}
	defer src.Close()
	tmp := mtpSubset + ".partial"
	out, e := os.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if e != nil {
		panic(e)
	}
	var relative uint64
	for i := range selected {
		selected[i].SHA = ""
		relative = mtpAlign(relative, g.Alignment)
		selected[i].Offset = relative
		relative += selected[i].Bytes
	}
	if relative > 4<<30 {
		panic("subset unexpectedly exceeds 4 GiB")
	}
	mtpWrite(out, uint32(0x46554747))
	mtpWrite(out, uint32(3))
	mtpWrite(out, uint64(len(selected)))
	mtpWrite(out, uint64(len(g.KV)))
	for _, kv := range g.KV {
		if _, e = out.Write(kv.Raw); e != nil {
			panic(e)
		}
	}
	for _, t := range selected {
		mtpString(out, t.Name)
		mtpWrite(out, uint32(len(t.Dims)))
		for _, n := range t.Dims {
			mtpWrite(out, n)
		}
		mtpWrite(out, t.Type)
		mtpWrite(out, t.Offset)
	}
	mtpPad(out, g.Alignment)
	original := map[string]mtpTensor{}
	for _, t := range g.Tensors {
		original[t.Name] = t
	}
	for i := range selected {
		t := &selected[i]
		mtpPad(out, g.Alignment)
		source := original[t.Name]
		h := sha256.New()
		n, e := io.CopyN(io.MultiWriter(out, h), io.NewSectionReader(src, int64(g.Data+source.Offset), int64(source.Bytes)), int64(source.Bytes))
		if e != nil || uint64(n) != source.Bytes {
			panic(fmt.Sprintf("copy %s: %v", t.Name, e))
		}
		t.SHA = hex.EncodeToString(h.Sum(nil))
	}
	if e = out.Sync(); e != nil {
		panic(e)
	}
	if e = out.Close(); e != nil {
		panic(e)
	}
	check := mtpParse(tmp)
	if check.MetadataSHA != g.MetadataSHA || len(check.Tensors) != len(selected) {
		panic("subset metadata/directory validation failed")
	}
	verify, e := os.Open(tmp)
	if e != nil {
		panic(e)
	}
	for i, t := range check.Tensors {
		expected := selected[i]
		if t.Name != expected.Name || t.Type != expected.Type || t.Bytes != expected.Bytes {
			panic("subset tensor mismatch")
		}
		h := sha256.New()
		if _, e = io.Copy(h, io.NewSectionReader(verify, int64(check.Data+t.Offset), int64(t.Bytes))); e != nil {
			panic(e)
		}
		if hex.EncodeToString(h.Sum(nil)) != expected.SHA {
			panic("subset payload mismatch")
		}
	}
	verify.Close()
	if e = os.Rename(tmp, mtpSubset); e != nil {
		panic(e)
	}
	side, e := os.Stat(mtpSubset)
	if e != nil {
		panic(e)
	}
	rec := mtpRecord{Source: mtpSource, SourceBytes: st.Size(), SourceID: mtpFileID(mtpSource), Subset: mtpSubset, SubsetBytes: side.Size(), SubsetSHA: mtpDigest(mtpSubset), MetadataSHA: g.MetadataSHA, Tensors: selected, Verified: true, Time: time.Now().Format(time.RFC3339)}
	mtpJSON(mtpReceipt, rec)
	fmt.Printf("extracted bytes=%d tensors=%d sha256=%s; retained payloads verified byte-for-byte\n", rec.SubsetBytes, len(selected), rec.SubsetSHA)
}
func mtpSafe(path, expected string) {
	a, e := filepath.Abs(path)
	if e != nil {
		panic(e)
	}
	root, e := filepath.Abs(mtpFolder)
	if e != nil {
		panic(e)
	}
	if !strings.EqualFold(a, expected) || !strings.EqualFold(filepath.Dir(a), root) {
		panic("path outside explicitly named model directory")
	}
	st, e := os.Lstat(path)
	if e != nil {
		panic(e)
	}
	if !st.Mode().IsRegular() {
		panic("refuse non-regular file")
	}
}
func mtpHold() {
	r := mtpReadRecord()
	if !r.Verified || r.Subset != mtpSubset || mtpDigest(mtpSubset) != r.SubsetSHA {
		panic("verified replacement required")
	}
	mtpSafe(mtpSource, mtpSource)
	st, _ := os.Stat(mtpSource)
	if st.Size() != r.SourceBytes || mtpFileID(mtpSource) != r.SourceID {
		panic("source identity changed")
	}
	if _, e := os.Lstat(mtpHeld); !os.IsNotExist(e) {
		panic("holding path already exists")
	}
	if e := os.Rename(mtpSource, mtpHeld); e != nil {
		panic(e)
	}
	fmt.Printf("held original source at %s; original path unavailable for independent validation\n", mtpHeld)
}
func mtpRestore() {
	r := mtpReadRecord()
	mtpSafe(mtpHeld, mtpHeld)
	if mtpFileID(mtpHeld) != r.SourceID {
		panic("held file identity changed")
	}
	if _, e := os.Lstat(mtpSource); !os.IsNotExist(e) {
		panic("original path occupied")
	}
	if e := os.Rename(mtpHeld, mtpSource); e != nil {
		panic(e)
	}
	fmt.Println("original source restored")
}
func mtpRetire() {
	r := mtpReadRecord()
	mtpSafe(mtpHeld, mtpHeld)
	st, _ := os.Stat(mtpHeld)
	if st.Size() != r.SourceBytes || mtpFileID(mtpHeld) != r.SourceID {
		panic("held source identity changed")
	}
	if _, e := os.Stat(mtpSource); !os.IsNotExist(e) {
		panic("source original path unexpectedly exists")
	}
	if !r.Verified || mtpDigest(mtpSubset) != r.SubsetSHA {
		panic("replacement changed")
	}
	for _, gate := range []struct {
		File string
		Keys []string
	}{{"isolated-mtp-runtime-summary.json", []string{"all_outputs_match_original", "draft_counts_match_original", "old_source_unavailable"}}, {"isolated-mtp-json-summary.json", []string{"pure_json_and_known_answers_pass", "matches_ar"}}} {
		b, e := os.ReadFile(filepath.Join(mtpResults, gate.File))
		if e != nil {
			panic(e)
		}
		var v map[string]any
		if e = json.Unmarshal(b, &v); e != nil {
			panic(e)
		}
		for _, k := range gate.Keys {
			if v[k] != true {
				panic("runtime gate missing or failed: " + k)
			}
		}
	}
	if e := os.Remove(mtpHeld); e != nil {
		panic(e)
	}
	mtpJSON(filepath.Join(mtpResults, "isolated-mtp-deletion.json"), map[string]any{"explicitly_authorized_original": mtpSource, "removed_holding_path": mtpHeld, "source_file_id": r.SourceID, "bytes_deleted": r.SourceBytes, "standalone_path": mtpSubset, "standalone_bytes": r.SubsetBytes, "net_bytes_reclaimed": r.SourceBytes - r.SubsetBytes, "standalone_sha256": r.SubsetSHA, "time": time.Now().Format(time.RFC3339), "recursive": false})
	fmt.Printf("deleted original %d bytes; net reclaimed %d bytes; standalone replacement retained\n", r.SourceBytes, r.SourceBytes-r.SubsetBytes)
}
func main() {
	defer func() {
		if v := recover(); v != nil {
			fmt.Fprintln(os.Stderr, v)
			os.Exit(1)
		}
	}()
	if len(os.Args) != 2 {
		panic("usage: isolate-mtp inspect|extract|hold|restore|retire")
	}
	switch os.Args[1] {
	case "inspect":
		g := mtpParse(mtpSource)
		t := mtpSelect(g)
		sort.Slice(t, func(i, j int) bool { return t[i].Name < t[j].Name })
		var n uint64
		for _, v := range t {
			n += v.Bytes
		}
		b, _ := json.MarshalIndent(map[string]any{"architecture": g.Architecture, "layers": g.Layers, "source_tensors": len(g.Tensors), "selected_tensors": t, "selected_payload_bytes": n, "alignment": g.Alignment}, "", "  ")
		fmt.Println(string(b))
	case "extract":
		mtpExtract()
	case "hold":
		mtpHold()
	case "restore":
		mtpRestore()
	case "retire":
		mtpRetire()
	default:
		panic("unknown action")
	}
}
