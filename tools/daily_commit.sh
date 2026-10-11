#!/usr/bin/env bash
# Usage: tools/daily_commit.sh <project-slug> <day 1-7> "<short summary>"
# Example: tools/daily_commit.sh quantlab 1 "data layer: schema, synthetic generator, loaders"
#
# Runs lint + tests for the project first, so you never push a red build.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <project-slug> <day 1-7> \"<summary>\"" >&2
  exit 1
fi

slug="$1"; day="$2"; summary="$3"
root="$(git rev-parse --show-toplevel)"
proj_dir="$(find "$root/projects" -maxdepth 1 -type d -name "*-${slug}" | head -n1)"

if [ -z "$proj_dir" ]; then
  echo "No project folder matching *-${slug} under projects/" >&2
  exit 1
fi

echo ">> Checking $(basename "$proj_dir")"
( cd "$proj_dir" && ruff check . && python -m pytest -q )

git -C "$root" add -A
git -C "$root" commit -m "feat(${slug}): day ${day}/7 - ${summary}"
git -C "$root" push
echo ">> Committed day ${day}/7 of ${slug}. Update the progress table in README.md if the project just finished."
