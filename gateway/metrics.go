package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
)

type Metrics struct {
	Registry        *prometheus.Registry
	Requests        *prometheus.CounterVec // key, model, status
	TTFT            *prometheus.HistogramVec
	Latency         *prometheus.HistogramVec
	OutputTokPerSec prometheus.Gauge
	DraftTokens     prometheus.Counter
	AcceptedTokens  prometheus.Counter
	BackendUp       *prometheus.GaugeVec   // backend
	BreakerState    *prometheus.GaugeVec   // backend: 0 closed, 1 open, 2 half-open
	RateLimited     *prometheus.CounterVec // key
}

func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := &Metrics{
		Registry: reg,
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "draftlab", Name: "requests_total",
			Help: "Proxied requests by key, model and status.",
		}, []string{"key", "model", "status"}),
		TTFT: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "draftlab", Name: "ttft_seconds",
			Help: "Time to first streamed byte.", Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
		}, []string{"model"}),
		Latency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "draftlab", Name: "request_latency_seconds",
			Help: "Total request latency.", Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120},
		}, []string{"model"}),
		OutputTokPerSec: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "draftlab", Name: "output_tokens_per_second",
			Help: "Completion tokens / second for the last completed request.",
		}),
		DraftTokens: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "draftlab", Name: "draft_tokens_total",
			Help: "Draft tokens proposed, from backend usage.",
		}),
		AcceptedTokens: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "draftlab", Name: "accepted_tokens_total",
			Help: "Draft tokens accepted, from backend usage.",
		}),
		BackendUp: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "draftlab", Name: "backend_up",
			Help: "Backend health: 1 up, 0 down.",
		}, []string{"backend"}),
		BreakerState: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "draftlab", Name: "breaker_state",
			Help: "Circuit breaker state: 0 closed, 1 open, 2 half-open.",
		}, []string{"backend"}),
		RateLimited: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "draftlab", Name: "rate_limited_total",
			Help: "Requests rejected by the per-key rate limit.",
		}, []string{"key"}),
	}
	reg.MustRegister(m.Requests, m.TTFT, m.Latency, m.OutputTokPerSec,
		m.DraftTokens, m.AcceptedTokens, m.BackendUp, m.BreakerState, m.RateLimited)
	return m
}

// Usage is the subset of the backend usage object the gateway consumes.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	DraftTokens      int `json:"draft_tokens"`
	AcceptedTokens   int `json:"accepted_tokens"`
}
