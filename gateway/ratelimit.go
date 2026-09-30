package main

import (
	"sync"
	"time"
)

// bucket is a per-key token bucket. take returns (ok, secondsUntilNextToken).
type bucket struct {
	mu   sync.Mutex
	rate float64
	cap  float64
	tok  float64
	last time.Time
}

func newBucket(rate float64, burst int) *bucket {
	return &bucket{rate: rate, cap: float64(burst), tok: float64(burst), last: time.Now()}
}

func (b *bucket) take(now time.Time) (bool, float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.tok += now.Sub(b.last).Seconds() * b.rate
	if b.tok > b.cap {
		b.tok = b.cap
	}
	b.last = now
	if b.tok >= 1 {
		b.tok--
		return true, 0
	}
	return false, (1 - b.tok) / b.rate
}
