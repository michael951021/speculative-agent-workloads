#!/usr/bin/env bash
# Idempotent toolchain bootstrap: uv + Python 3.12 venv + pinned binaries into .tools/.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TOOLS=.tools
BIN="$TOOLS/bin"
mkdir -p "$BIN"

case "$(uname -m)" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

get_version() { grep -E "^$1=" tools/versions.txt | cut -d= -f2; }

# --- uv (host copy preferred, else standalone binary into .tools/bin) ---
if command -v uv >/dev/null 2>&1; then
  UV=uv
else
  echo "installing uv into $BIN"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BIN" sh
  UV="$BIN/uv"
fi
"$UV" --version

# --- Python 3.12 venv ---
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'; then
  echo "venv: .venv already has CPython 3.12, keeping"
else
  echo "creating .venv with CPython 3.12"
  "$UV" venv -p 3.12 .venv
fi
PY=.venv/bin/python

# --- Python deps: torch from the CPU wheel index first, then the package extras ---
if "$PY" -c 'import torch' 2>/dev/null; then
  echo "deps: torch already installed, installing remaining extras"
else
  echo "deps: installing torch (CPU index)"
  "$UV" pip install --python "$PY" torch \
    --index-url https://download.pytorch.org/whl/cpu
fi
"$UV" pip install --python "$PY" -e ".[all]"

# --- Pinned binaries into .tools/ (skipped when versions.txt is unchanged) ---
hash=$(sha256sum tools/versions.txt | cut -d' ' -f1)
if [ "$(cat "$TOOLS/.versions" 2>/dev/null || true)" = "$hash" ]; then
  echo "binaries: up to date (tools/versions.txt unchanged)"
else
  dl() { # dl <url> <dest>
    echo "download: $1"
    curl -fL --retry 3 --retry-delay 2 -o "$2" "$1"
  }
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT

  v=$(get_version go)
  dl "https://go.dev/dl/go${v}.linux-${ARCH}.tar.gz" "$tmp/go.tgz"
  rm -rf "$TOOLS/go"
  tar xzf "$tmp/go.tgz" -C "$TOOLS"   # -> .tools/go/bin/go

  v=$(get_version helm)
  dl "https://get.helm.sh/helm-${v}-linux-${ARCH}.tar.gz" "$tmp/helm.tgz"
  tar xzf "$tmp/helm.tgz" -C "$tmp" "linux-${ARCH}/helm"
  install "$tmp/linux-${ARCH}/helm" "$BIN/helm"

  v=$(get_version kubeconform)
  dl "https://github.com/yannh/kubeconform/releases/download/${v}/kubeconform-linux-${ARCH}.tar.gz" "$tmp/kc.tgz"
  tar xzf "$tmp/kc.tgz" -C "$tmp"
  install "$tmp/kubeconform" "$BIN/kubeconform"

  v=$(get_version terraform)
  dl "https://releases.hashicorp.com/terraform/${v}/terraform_${v}_linux_${ARCH}.zip" "$tmp/tf.zip"
  (cd "$tmp" && python3 -m zipfile -e tf.zip tfx/)
  install "$tmp/tfx/terraform" "$BIN/terraform"

  v=$(get_version tflint)
  dl "https://github.com/terraform-linters/tflint/releases/download/v${v}/tflint_linux_${ARCH}.zip" "$tmp/tl.zip"
  (cd "$tmp" && python3 -m zipfile -e tl.zip tlf/)
  install "$tmp/tlf/tflint" "$BIN/tflint"

  v=$(get_version hadolint)
  ha=$([ "$ARCH" = amd64 ] && echo x86_64 || echo arm64)
  dl "https://github.com/hadolint/hadolint/releases/download/${v}/hadolint-Linux-${ha}" "$tmp/hadolint"
  install "$tmp/hadolint" "$BIN/hadolint"

  v=$(get_version shellcheck)
  dl "https://github.com/koalaman/shellcheck/releases/download/${v}/shellcheck-${v}.linux.${ha}.tar.xz" "$tmp/sc.tgz"
  tar xJf "$tmp/sc.tgz" -C "$tmp"
  install "$tmp/shellcheck-${v}/shellcheck" "$BIN/shellcheck"

  v=$(get_version promtool)
  dl "https://github.com/prometheus/prometheus/releases/download/v${v}/prometheus-${v}.linux-${ARCH}.tar.gz" "$tmp/prom.tgz"
  tar xzf "$tmp/prom.tgz" -C "$tmp" "prometheus-${v}.linux-${ARCH}/promtool"
  install "$tmp/prometheus-${v}.linux-${ARCH}/promtool" "$BIN/promtool"

  echo "$hash" > "$TOOLS/.versions"
fi

echo "bootstrap complete"
