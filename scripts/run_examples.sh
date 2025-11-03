#!/usr/bin/env bash
set -euo pipefail

# Discover and run ONLY example_*.py files.
# Usage:
#   scripts/run_examples.sh
#   scripts/run_examples.sh --grep finiteT
#   EXAMPLES_DIR=examples scripts/run_examples.sh
#   DRY_RUN=1 scripts/run_examples.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
EXAMPLES_DIR="${EXAMPLES_DIR:-$ROOT_DIR/docs/examples}"
DRY_RUN="${DRY_RUN:-0}"
GREP="${1:-}"

# Activate venv if present
if [ -d "$ROOT_DIR/.venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.venv/bin/activate"
fi

if [ ! -d "$EXAMPLES_DIR" ]; then
  echo "Examples directory not found: $EXAMPLES_DIR"
  echo "Create it and put your example_*.py files there."
  exit 1
fi

mapfile -t FILES < <(find "$EXAMPLES_DIR" -type f -name "example_*.py" | sort)

if [ -n "$GREP" ]; then
  FILES=($(printf "%s\n" "${FILES[@]}" | grep -i "$GREP" || true))
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No example_*.py files found."
  exit 0
fi

echo "Found ${#FILES[@]} example(s):"
printf " - %s\n" "${FILES[@]}"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1 -> not executing."
  exit 0
fi

# Run each example with plain Python
for f in "${FILES[@]}"; do
  echo "=============================="
  echo "▶ Running: $f"
  echo "=============================="
  python "$f"
  echo
done

echo "✅ All examples ran successfully."
