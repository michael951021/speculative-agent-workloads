export PATH := $(PWD)/.tools/go/bin:$(PWD)/.tools/bin:$(PATH)
export OMP_NUM_THREADS ?= 16
export XLA_FLAGS ?= --xla_cpu_multi_thread_eigen=true

.PHONY: bootstrap check fmt

bootstrap:
	bash tools/bootstrap.sh

check:
	bash tools/check.sh

fmt:
	@test -x .venv/bin/ruff || { echo "run 'make bootstrap' first"; exit 1; }
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .
	@if [ -d gateway ]; then (cd gateway && gofmt -w $$(find . -name '*.go')); fi
