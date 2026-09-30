package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	promhttp "github.com/prometheus/client_golang/prometheus/promhttp"
)

// Gateway is the HTTP server core: auth, rate limit, routing, proxy, metrics.
type Gateway struct {
	cfg      *Config
	keys     map[string]*KeyCfg // sha256 hex -> cfg
	buckets  map[string]*bucket // sha256 hex -> bucket
	backends []*backend
	byModel  map[string]*backend
	metrics  *Metrics
	healthC  *http.Client
	log      *slog.Logger
	uiFS     http.Handler
}

func NewGateway(cfg *Config, m *Metrics, log *slog.Logger) *Gateway {
	g := &Gateway{
		cfg:     cfg,
		keys:    map[string]*KeyCfg{},
		buckets: map[string]*bucket{},
		byModel: map[string]*backend{},
		metrics: m,
		healthC: &http.Client{Timeout: 2 * time.Second},
		log:     log,
	}
	for _, k := range cfg.Keys {
		h := k.SHA256
		g.keys[h] = &k
		g.buckets[h] = newBucket(k.Rate, k.Burst)
	}
	for _, bc := range cfg.Backends {
		b := newBackend(bc, cfg.Breaker, cfg.Retries)
		g.backends = append(g.backends, b)
		for _, mname := range bc.Models {
			g.byModel[mname] = b
		}
	}
	if dir := cfg.UIDir; dir != "" {
		g.uiFS = http.StripPrefix("/ui", http.FileServer(http.Dir(dir)))
	}
	return g
}

func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.chat)
	mux.HandleFunc("GET /v1/models", g.models)
	mux.Handle("GET /metrics", http.Handler(&metricsHandler{g: g}))
	mux.HandleFunc("GET /healthz", g.healthz)
	if g.uiFS != nil {
		mux.Handle("/ui/", g.uiFS)
	}
	return logMiddleware(g.log, mux)
}

type metricsHandler struct{ g *Gateway }

func (m *metricsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	promhttp.HandlerFor(m.g.metrics.Registry, promhttp.HandlerOpts{}).ServeHTTP(w, r)
}

func logMiddleware(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rec, r)
		log.Info("request", "method", r.Method, "path", r.URL.Path,
			"status", rec.status, "dur_ms", float64(time.Since(start).Microseconds())/1e3)
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func writeErr(w http.ResponseWriter, code int, msg, codeName string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"message": msg, "type": "gateway_error", "code": codeName},
	})
}

// chat proxies /v1/chat/completions.
func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	kc := g.authenticate(r)
	if kc == nil {
		writeErr(w, http.StatusUnauthorized, "invalid API key", "invalid_api_key")
		return
	}
	ok, retryIn := g.buckets[kc.SHA256].take(time.Now())
	if !ok {
		g.metrics.RateLimited.WithLabelValues(kc.Name).Inc()
		w.Header().Set("Retry-After", fmt.Sprintf("%d", ceilSec(retryIn)))
		writeErr(w, http.StatusTooManyRequests, "rate limit exceeded", "rate_limited")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 32<<20))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading body: "+err.Error(), "invalid_request")
		return
	}
	var reqBody struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		writeErr(w, http.StatusBadRequest, "bad JSON body", "invalid_request")
		return
	}
	b, found := g.byModel[reqBody.Model]
	if !found {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not found", reqBody.Model), "model_not_found")
		return
	}
	if !b.br.allow() {
		writeErr(w, http.StatusServiceUnavailable,
			fmt.Sprintf("backend %s unavailable (circuit open)", b.cfg.Name), "breaker_open")
		return
	}

	hdr := make(http.Header)
	for k, vv := range r.Header {
		if strings.EqualFold(k, "Authorization") || isHopByHop(k) {
			continue
		}
		hdr[k] = vv
	}
	resp, _ := b.do(r.Context(), "/v1/chat/completions", body, hdr)
	g.metrics.BreakerState.WithLabelValues(b.cfg.Name).Set(float64(b.br.state()))
	status := http.StatusBadGateway
	if resp != nil {
		status = resp.StatusCode
		defer resp.Body.Close()
		stream := resp.Header.Get("Content-Type") == "text/event-stream" || reqBody.Stream
		if stream {
			g.copyStream(w, resp, reqBody.Model, start)
		} else {
			g.copyBody(w, resp, reqBody.Model, start)
		}
	} else {
		writeErr(w, status, fmt.Sprintf("backend %s unreachable", b.cfg.Name), "backend_unavailable")
	}
	g.metrics.Latency.WithLabelValues(reqBody.Model).Observe(time.Since(start).Seconds())
	g.metrics.Requests.WithLabelValues(kc.Name, reqBody.Model, fmt.Sprint(status)).Inc()
}

var hopByHop = map[string]bool{
	"Connection": true, "Keep-Alive": true, "Proxy-Authenticate": true,
	"Proxy-Authorization": true, "Te": true, "Trailer": true,
	"Transfer-Encoding": true, "Upgrade": true,
}

func isHopByHop(k string) bool { return hopByHop[strings.ToLower(k)] }

// copyStream tees the SSE stream to the client, flushing per read, and pulls
// usage from the final data chunk.
func (g *Gateway) copyStream(w http.ResponseWriter, resp *http.Response, model string, start time.Time) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(resp.StatusCode)
	fl, _ := w.(http.Flusher)
	r := bufio.NewReaderSize(resp.Body, 64<<10)
	buf := make([]byte, 64<<10)
	line := make([]byte, 0, 1024)
	var usage *Usage
	var first time.Time
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if first.IsZero() {
				first = time.Now()
			}
			w.Write(buf[:n])
			if fl != nil {
				fl.Flush()
			}
			line = scanUsage(buf[:n], line, &usage)
		}
		if err != nil {
			break
		}
	}
	g.recordUsage(w, resp, model, usage, start)
	if !first.IsZero() {
		g.metrics.TTFT.WithLabelValues(model).Observe(time.Since(start).Seconds())
	}
}

// copyBody forwards a non-stream response.
func (g *Gateway) copyBody(w http.ResponseWriter, resp *http.Response, model string, start time.Time) {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		writeErr(w, http.StatusBadGateway, "backend read error", "backend_unavailable")
		return
	}
	var u struct {
		Usage *Usage `json:"usage"`
	}
	if json.Unmarshal(body, &u) == nil {
		g.recordUsage(w, resp, model, u.Usage, start)
	}
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	if l := resp.Header.Get("Content-Length"); l != "" {
		w.Header().Set("Content-Length", l)
	}
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func (g *Gateway) recordUsage(w http.ResponseWriter, resp *http.Response, model string, u *Usage, start time.Time) {
	if u == nil {
		return
	}
	g.metrics.DraftTokens.Add(float64(u.DraftTokens))
	g.metrics.AcceptedTokens.Add(float64(u.AcceptedTokens))
	secs := time.Since(start).Seconds()
	if secs > 0 {
		g.metrics.OutputTokPerSec.Set(float64(u.CompletionTokens) / secs)
	}
}

// scanUsage appends data to the in-flight SSE line buffer and extracts usage
// from completed `data: {...}` lines. Returns the leftover partial line.
func scanUsage(data, line []byte, usage **Usage) []byte {
	line = append(line, data...)
	for {
		i := bytes.IndexByte(line, '\n')
		if i < 0 {
			break
		}
		l := line[:i]
		line = line[i+1:]
		l = bytes.TrimSpace(l)
		if !bytes.HasPrefix(l, []byte("data:")) || bytes.Equal(l, []byte("data: [DONE]")) {
			continue
		}
		payload := bytes.TrimSpace(l[len("data:"):])
		var chunk struct {
			Usage *Usage `json:"usage"`
		}
		if json.Unmarshal(payload, &chunk) == nil && chunk.Usage != nil {
			*usage = chunk.Usage
		}
		if len(line) > 1<<20 { // runaway line: drop it
			line = nil
		}
	}
	return line
}

// models merges /v1/models from all backends.
func (g *Gateway) models(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	var (
		mu   sync.Mutex
		seen = map[string]bool{}
		data []map[string]any
		errs []error
	)
	var wg sync.WaitGroup
	for _, b := range g.backends {
		wg.Add(1)
		go func(b *backend) {
			defer wg.Done()
			items, err := b.fetchModels(ctx)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				return
			}
			for _, it := range items {
				if id, _ := it["id"].(string); id != "" && !seen[id] {
					seen[id] = true
					data = append(data, it)
				}
			}
		}(b)
	}
	wg.Wait()
	w.Header().Set("Content-Type", "application/json")
	if data == nil && len(errs) == len(g.backends) {
		writeErr(w, http.StatusBadGateway, "all backends down", "backend_unavailable")
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

// healthz reports per-backend health and breaker state.
func (g *Gateway) healthz(w http.ResponseWriter, r *http.Request) {
	st := map[string]any{}
	for _, b := range g.backends {
		st[b.cfg.Name] = map[string]any{"up": b.isUp(), "breaker": breakerName(b.br.state())}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"backends": st})
}

func (g *Gateway) StartHealthChecks(ctx context.Context) {
	for _, b := range g.backends {
		name := b.cfg.Name
		go func(b *backend) {
			up := b.health(ctx)
			g.metrics.BackendUp.WithLabelValues(name).Set(boolFloat(up))
			t := time.NewTicker(g.cfg.HealthCheckInterval)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					up = b.health(ctx)
					g.metrics.BackendUp.WithLabelValues(name).Set(boolFloat(up))
					g.metrics.BreakerState.WithLabelValues(name).Set(float64(b.br.state()))
				}
			}
		}(b)
	}
}

func boolFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func ceilSec(f float64) int {
	n := int(f)
	if float64(n) < f {
		n++
	}
	if n < 1 {
		n = 1
	}
	return n
}
