package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type KeyCfg struct {
	Name   string  `yaml:"name"`
	SHA256 string  `yaml:"sha256"` // hex sha256 of the raw API key
	Rate   float64 `yaml:"rate"`   // sustained requests/sec
	Burst  int     `yaml:"burst"`  // bucket capacity
}

type BackendCfg struct {
	Name   string   `yaml:"name"`
	URL    string   `yaml:"url"`
	Models []string `yaml:"models"`
}

type BreakerCfg struct {
	Failures      int           `yaml:"failures"` // consecutive failures to open
	HalfOpenAfter time.Duration `yaml:"half_open_after"`
}

type Config struct {
	Listen              string        `yaml:"listen"`
	UIDir               string        `yaml:"ui_dir"`
	Retries             int           `yaml:"retries"` // retries on transport errors
	HealthCheckInterval time.Duration `yaml:"health_check_interval"`
	Breaker             BreakerCfg    `yaml:"breaker"`
	Backends            []BackendCfg  `yaml:"backends"`
	Keys                []KeyCfg      `yaml:"keys"`
}

func LoadConfig(path string) (*Config, error) {
	cfg := &Config{
		Listen:              ":8200",
		UIDir:               "web/dist",
		Retries:             1,
		HealthCheckInterval: 5 * time.Second,
		Breaker:             BreakerCfg{Failures: 5, HalfOpenAfter: 2 * time.Second},
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if err := yaml.Unmarshal(b, cfg); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	for _, k := range cfg.Keys {
		if _, err := hex.DecodeString(k.SHA256); err != nil || len(k.SHA256) != 64 {
			return nil, fmt.Errorf("key %q: sha256 must be 64 hex chars", k.Name)
		}
		if k.Rate <= 0 {
			k.Rate = 10
		}
		if k.Burst <= 0 {
			k.Burst = 20
		}
	}
	if cfg.Breaker.Failures <= 0 {
		cfg.Breaker.Failures = 5
	}
	return cfg, nil
}

func KeyHash(key string) string {
	s := sha256.Sum256([]byte(key))
	return hex.EncodeToString(s[:])
}
