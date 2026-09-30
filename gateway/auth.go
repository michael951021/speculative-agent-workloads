package main

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// authenticate returns the key config for a Bearer token, or nil.
func (g *Gateway) authenticate(r *http.Request) *KeyCfg {
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, "Bearer ") {
		return nil
	}
	sum := KeyHash(strings.TrimSpace(h[len("Bearer "):]))
	for k, kc := range g.keys {
		if subtle.ConstantTimeCompare([]byte(k), []byte(sum)) == 1 {
			return kc
		}
	}
	return nil
}
