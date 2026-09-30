package main

import (
	"sync"
	"time"
)

type breakerState int

const (
	closed breakerState = iota
	open
	halfOpen
)

// breaker opens after `failures` consecutive failures, stays open for `reset`,
// then lets a single probe through (half-open); success closes, failure reopens.
type breaker struct {
	mu        sync.Mutex
	threshold int
	reset     time.Duration
	st        breakerState
	fails     int
	openedAt  time.Time
}

func newBreaker(threshold int, reset time.Duration) *breaker {
	return &breaker{threshold: threshold, reset: reset}
}

// allow reports whether a request may reach the backend.
func (b *breaker) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.st {
	case closed:
		return true
	case open:
		if time.Since(b.openedAt) >= b.reset {
			b.st = halfOpen
			return true
		}
		return false
	default: // halfOpen: probe in flight
		return false
	}
}

func (b *breaker) onResult(failed bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.st {
	case closed:
		if failed {
			b.fails++
			if b.fails >= b.threshold {
				b.st = open
				b.openedAt = time.Now()
			}
		} else {
			b.fails = 0
		}
	case halfOpen:
		if failed {
			b.st = open
			b.openedAt = time.Now()
		} else {
			b.st = closed
			b.fails = 0
		}
	}
}

func (b *breaker) state() breakerState {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.st
}

func breakerName(s breakerState) string {
	switch s {
	case closed:
		return "closed"
	case open:
		return "open"
	default:
		return "half_open"
	}
}
