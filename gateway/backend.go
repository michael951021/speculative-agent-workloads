package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

type backend struct {
	cfg     BackendCfg
	client  *http.Client
	br      *breaker
	retries int

	mu sync.Mutex
	up bool
}

func newBackend(cfg BackendCfg, bcfg BreakerCfg, retries int) *backend {
	return &backend{
		cfg:     cfg,
		client:  &http.Client{}, // no timeout: streams are long-lived
		br:      newBreaker(bcfg.Failures, bcfg.HalfOpenAfter),
		retries: retries,
	}
}

// do forwards a request, retrying on transport (connect) errors.
// failed=true means the backend is to blame (transport failure or 5xx).
func (b *backend) do(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, bool) {
	var lastErr error
	for i := 0; i <= b.retries; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.URL+path, bytes.NewReader(body))
		if err != nil {
			return nil, true
		}
		for k, vv := range headers {
			for _, v := range vv {
				req.Header.Add(k, v)
			}
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := b.client.Do(req)
		if err == nil {
			failed := resp.StatusCode >= 500
			b.br.onResult(failed)
			return resp, failed
		}
		lastErr = err
		if i < b.retries {
			time.Sleep(time.Duration(50*(1<<i)) * time.Millisecond)
		}
	}
	b.br.onResult(true)
	return nil, lastErr != nil
}

// health probes the backend's /v1/models endpoint.
func (b *backend) health(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.cfg.URL+"/v1/models", nil)
	if err != nil {
		return false
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	up := resp.StatusCode == http.StatusOK
	b.mu.Lock()
	b.up = up
	b.mu.Unlock()
	return up
}

func (b *backend) isUp() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.up
}

// fetchModels returns the backend's model list, or an error.
func (b *backend) fetchModels(ctx context.Context) ([]map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.cfg.URL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", b.cfg.Name, resp.Status)
	}
	var out struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, err
	}
	return out.Data, nil
}
