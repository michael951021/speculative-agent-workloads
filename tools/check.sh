#!/usr/bin/env bash
# Sub-checks of `make check`. Each skips with a message until its directory exists.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="$PWD/.tools/go/bin:$PWD/.tools/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=true}"
PY=.venv/bin/python
ROOT=$PWD

fail=0
section() { printf '\n== %s ==\n' "$1"; }
skip() { echo "SKIP $1 ($2 absent)"; }
check() { # check <label> <cmd...>
  local label=$1; shift
  if ! "$@"; then echo "FAIL $label" >&2; fail=1; fi
}

if [ ! -x "$PY" ]; then
  echo "FAIL python venv missing — run `make bootstrap` first" >&2
  exit 1
fi

if [ -d draftlab ]; then
  section "python: ruff"
  check ruff "$PY" -m ruff check .
  section "python: pytest"
  check pytest "$PY" -m pytest -q
else
  skip "python" draftlab
fi

if [ -d gateway ]; then
  section "go: test"
  (cd gateway && go test ./...) || { echo "FAIL go test" >&2; fail=1; }
else
  skip "go" gateway
fi

if [ -d web ] && [ -f web/package.json ]; then
  section "web: build + vitest"
  if [ ! -d web/node_modules ]; then (cd web && npm install --no-audit --no-fund); fi
  (cd web && npm run build) || { echo "FAIL web build" >&2; fail=1; }
  (cd web && npx vitest run) || { echo "FAIL web vitest" >&2; fail=1; }
else
  skip "web" web
fi

if [ -d deploy/helm ]; then
  section "helm: lint + kubeconform (cpu + gpu)"
  check helm-lint helm lint deploy/helm/draftlab
  (cd deploy/helm/draftlab && \
    for extra in "" "-f values-gpu.yaml"; do
      helm template draftlab . $extra \
        | "$ROOT/.tools/bin/kubeconform" -strict -summary \
          -schema-location default \
          -schema-location "crd-schemas/{{ .ResourceKind }}{{ .KindSuffix }}.json" || exit 1
    done) || { echo "FAIL kubeconform" >&2; fail=1; }
  check grafana-sync diff -q deploy/helm/draftlab/grafana/draftlab.json deploy/grafana/draftlab.json
else
  skip "helm" deploy/helm
fi

if [ -d deploy/terraform ]; then
  section "terraform: validate + tflint"
  (cd deploy/terraform && terraform init -backend=false -input=false >/dev/null \
    && terraform validate) || { echo "FAIL terraform validate" >&2; fail=1; }
  (cd deploy/terraform && tflint --init >/dev/null && tflint) \
    || { echo "FAIL tflint" >&2; fail=1; }
else
  skip "terraform" deploy/terraform
fi

if [ -d deploy/docker ] && ls deploy/docker/Dockerfile* >/dev/null 2>&1; then
  section "hadolint"
  (cd deploy/docker && for f in Dockerfile*; do hadolint "$f" || exit 1; done) \
    || { echo "FAIL hadolint" >&2; fail=1; }
else
  skip "hadolint" deploy/docker
fi

if command -v shellcheck >/dev/null 2>&1 && ls deploy/kind/*.sh >/dev/null 2>&1; then
  section "shell: shellcheck"
  while IFS= read -r f; do
    check "shellcheck $f" shellcheck "$f"
  done < <(find deploy -type f -name '*.sh')
else
  skip "shellcheck" shellcheck
fi

if [ -d deploy/prometheus ] || [ -d deploy/grafana ]; then
  section "promtool"
  if [ -d deploy/prometheus ]; then
    check prom-config promtool check config deploy/prometheus/prometheus.yml
    check prom-rules promtool check rules deploy/prometheus/rules.yml
    check prom-test promtool test rules deploy/prometheus/rules_test.yml
  fi
  if [ -d deploy/grafana ] && ls deploy/grafana/*.json >/dev/null 2>&1; then
    # grafana dashboards: promtool can't validate them; json-parse instead
    while IFS= read -r f; do "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$f" || fail=1; done \
      < <(find deploy/grafana -name '*.json')
  fi
else
  skip "promtool" deploy/prometheus
fi

if [ -f .github/workflows/ci.yml ]; then
  section "ci.yml: yaml parse"
  check ci-yaml "$PY" -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"
fi

[ "$fail" = 0 ] || { echo; echo "make check: FAILURES" >&2; exit 1; }
echo; echo "make check: all sub-checks passed or skipped"
