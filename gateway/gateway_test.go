package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fake is a stand-in backend speaking just enough of the OpenAI API.
type fake struct {
	models     []string
	alwaysFail atomic.Bool
	reqs       atomic.Int64
	chunkDelay time.Duration
}

func (f *fake) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		data := []map[string]any{}
		for _, m := range f.models {
			data = append(data, map[string]any{"id": m, "object": "model", "owned_by": "draftlab"})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.reqs.Add(1)
		if f.alwaysFail.Load() {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		json.Unmarshal(body, &req)
		usage := map[string]int{"prompt_tokens": 10, "completion_tokens": 5,
			"total_tokens": 15, "draft_tokens": 12, "accepted_tokens": 7}
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"id": "c1", "object": "chat.completion", "model": req.Model,
				"choices": []map[string]any{{"index": 0, "finish_reason": "stop",
					"message": map[string]any{"role": "assistant", "content": "hello"}}},
				"usage": usage,
			})
			return
		}
		fl, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i, c := range []string{"c1", "c2", "c3"} {
			chunk := map[string]any{"id": "c1", "object": "chat.completion.chunk", "model": req.Model,
				"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": c}, "finish_reason": nil}}}
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(chunk))
			fl.Flush()
			if i < 2 {
				time.Sleep(f.chunkDelay)
			}
		}
		final := map[string]any{"id": "c1", "object": "chat.completion.chunk", "model": req.Model,
			"choices": []map[string]any{{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
			"usage":   usage}
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", mustJSON(final))
		fl.Flush()
	})
	return mux
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

type testEnv struct {
	cfg    *Config
	fakes  map[string]*fake
	server *httptest.Server
}

func newEnv(t *testing.T, breakFailures int, halfOpen time.Duration, rate float64, burst int) *testEnv {
	t.Helper()
	f1, f2 := &fake{models: []string{"m1"}}, &fake{models: []string{"m2"}}
	s1 := httptest.NewServer(f1.handler())
	s2 := httptest.NewServer(f2.handler())
	t.Cleanup(func() { s1.Close(); s2.Close() })
	cfg := &Config{
		Listen: "127.0.0.1:0", UIDir: "", Retries: 0,
		HealthCheckInterval: time.Hour,
		Breaker:             BreakerCfg{Failures: breakFailures, HalfOpenAfter: halfOpen},
		Backends: []BackendCfg{
			{Name: "b1", URL: s1.URL, Models: []string{"m1"}},
			{Name: "b2", URL: s2.URL, Models: []string{"m2"}},
		},
		Keys: []KeyCfg{{Name: "dev", SHA256: KeyHash("sk-test"), Rate: rate, Burst: burst}},
	}
	g := NewGateway(cfg, NewMetrics(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	srv := httptest.NewServer(g.Handler())
	t.Cleanup(srv.Close)
	return &testEnv{cfg: cfg, fakes: map[string]*fake{"b1": f1, "b2": f2}, server: srv}
}

func post(t *testing.T, url, key, model string, stream bool) (*http.Response, []byte) {
	t.Helper()
	body := fmt.Sprintf(`{"model": %q, "stream": %v, "messages": [{"role": "user", "content": "hi"}]}`, model, stream)
	req, _ := http.NewRequest("POST", url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return resp, b
}

func TestAuth401(t *testing.T) {
	e := newEnv(t, 5, time.Second, 100, 100)
	for _, key := range []string{"", "sk-wrong"} {
		resp, _ := post(t, e.server.URL+"/v1/chat/completions", key, "m1", false)
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("key=%q: want 401, got %d", key, resp.StatusCode)
		}
	}
}

func TestRateLimit429(t *testing.T) {
	e := newEnv(t, 5, time.Second, 50, 2)
	for i, want := range []int{200, 200, 429, 429} {
		resp, _ := post(t, e.server.URL+"/v1/chat/completions", "sk-test", "m1", false)
		if resp.StatusCode != want {
			t.Fatalf("req %d: want %d, got %d", i, want, resp.StatusCode)
		}
		if want == 429 {
			ra := resp.Header.Get("Retry-After")
			if ra != "1" {
				t.Fatalf("Retry-After = %q, want 1", ra)
			}
		}
	}
}

func TestStreamOrder(t *testing.T) {
	e := newEnv(t, 5, time.Second, 100, 100)
	e.fakes["b1"].chunkDelay = 10 * time.Millisecond
	resp, body := post(t, e.server.URL+"/v1/chat/completions", "sk-test", "m1", true)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	got := []string{}
	sc := bufio.NewScanner(bytes.NewReader(body))
	for sc.Scan() {
		l := sc.Text()
		if !strings.HasPrefix(l, "data: ") || l == "data: [DONE]" {
			continue
		}
		var c struct {
			Choices []struct{ Delta struct{ Content string } }
		}
		if json.Unmarshal([]byte(l[len("data: "):]), &c) == nil && len(c.Choices) > 0 {
			if s := c.Choices[0].Delta.Content; s != "" {
				got = append(got, s)
			}
		}
	}
	if strings.Join(got, "") != "c1c2c3" {
		t.Fatalf("stream order %v, want [c1 c2 c3]", got)
	}
}

func TestBreakerOpensHalfOpensCloses(t *testing.T) {
	e := newEnv(t, 5, 150*time.Millisecond, 100, 100)
	f := e.fakes["b1"]
	f.alwaysFail.Store(true)
	url := e.server.URL + "/v1/chat/completions"
	// 5 consecutive 5xx failures open the breaker.
	for i := 0; i < 5; i++ {
		resp, _ := post(t, url, "sk-test", "m1", false)
		if resp.StatusCode != 500 {
			t.Fatalf("failure %d: want 500, got %d", i, resp.StatusCode)
		}
	}
	if f.reqs.Load() != 5 {
		t.Fatalf("upstream hits %d, want 5", f.reqs.Load())
	}
	// Open: 503 without reaching the backend.
	resp, _ := post(t, url, "sk-test", "m1", false)
	if resp.StatusCode != 503 {
		t.Fatalf("open breaker: want 503, got %d", resp.StatusCode)
	}
	if f.reqs.Load() != 5 {
		t.Fatalf("upstream hit while open")
	}
	// Half-open probe fails again -> reopens.
	time.Sleep(200 * time.Millisecond)
	resp, _ = post(t, url, "sk-test", "m1", false)
	if resp.StatusCode != 500 {
		t.Fatalf("half-open probe: want 500, got %d", resp.StatusCode)
	}
	if f.reqs.Load() != 6 {
		t.Fatalf("probe did not reach backend")
	}
	// Backend recovers; next probe closes the breaker.
	f.alwaysFail.Store(false)
	time.Sleep(200 * time.Millisecond)
	resp, _ = post(t, url, "sk-test", "m1", false)
	if resp.StatusCode != 200 {
		t.Fatalf("recovery probe: want 200, got %d", resp.StatusCode)
	}
	resp, _ = post(t, url, "sk-test", "m1", false)
	if resp.StatusCode != 200 {
		t.Fatalf("after close: want 200, got %d", resp.StatusCode)
	}
	if f.reqs.Load() != 8 {
		t.Fatalf("upstream hits %d, want 8", f.reqs.Load())
	}
}

func TestMetricsCounted(t *testing.T) {
	e := newEnv(t, 5, time.Second, 100, 100)
	url := e.server.URL + "/v1/chat/completions"
	post(t, url, "sk-test", "m1", false)
	post(t, url, "sk-test", "m1", true)
	resp, err := http.Get(e.server.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	metrics, _ := io.ReadAll(resp.Body)
	txt := string(metrics)
	want := []string{
		`draftlab_requests_total{key="dev",model="m1",status="200"} 2`,
		`draftlab_draft_tokens_total 24`,
		`draftlab_accepted_tokens_total 14`,
		`draftlab_ttft_seconds_count{model="m1"} 1`,
		`draftlab_breaker_state{backend="b1"} 0`,
	}
	for _, w := range want {
		if !strings.Contains(txt, w) {
			t.Errorf("metrics missing %q", w)
		}
	}
	if !containsFloatGT(txt, "draftlab_output_tokens_per_second ", 0) {
		t.Errorf("output_tokens_per_second not > 0")
	}
}

// containsFloatGT reports whether a metric line starts with name and its value > v.
func containsFloatGT(txt, name string, v float64) bool {
	for _, line := range strings.Split(txt, "\n") {
		if strings.HasPrefix(line, name) {
			var f float64
			if _, err := fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(line, name)), "%g", &f); err == nil {
				return f > v
			}
		}
	}
	return false
}

func TestModelsMerged(t *testing.T) {
	e := newEnv(t, 5, time.Second, 100, 100)
	req, _ := http.NewRequest("GET", e.server.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Data []map[string]any `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&out)
	ids := map[string]bool{}
	for _, d := range out.Data {
		ids[d["id"].(string)] = true
	}
	if !ids["m1"] || !ids["m2"] || len(out.Data) != 2 {
		t.Fatalf("merged models %v, want m1+m2", ids)
	}
}

func TestUnknownModel404(t *testing.T) {
	e := newEnv(t, 5, time.Second, 100, 100)
	resp, _ := post(t, e.server.URL+"/v1/chat/completions", "sk-test", "nope", false)
	if resp.StatusCode != 404 {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
}

func TestConfigLoad(t *testing.T) {
	cfg, err := LoadConfig("config.yaml")
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if len(cfg.Keys) == 0 || len(cfg.Backends) == 0 {
		t.Fatal("empty config")
	}
	if KeyHash("sk-draftlab-dev") != cfg.Keys[0].SHA256 {
		t.Fatal("sample key hash mismatch")
	}
}
