#!/usr/bin/env bash
# Copy the macOS app's Codex accounting cases into the desktop's tests, byte
# for byte, from a commit of cli-pulse/cli-pulse-private:
#
#   scripts/sync-codex-accounting-cases.sh <commit>
#
# Then set MAC_CODEX_CASES_COMMIT and MAC_CODEX_CASES_SHA256 in
# src-tauri/tests/scanner_integration.rs to what it prints, and run the Rust
# tests. Both apps must report the same tokens for every case; a case the
# desktop counts differently on purpose goes in DESKTOP_DIFFERS_FROM_MAC.
set -euo pipefail

ref="${1:?usage: $0 <cli-pulse-private commit>}"
cd "$(dirname "$0")/.."

url="https://raw.githubusercontent.com/cli-pulse/cli-pulse-private/${ref}/CLI%20Pulse%20Bar/CLIPulseCore/Tests/Fixtures/codex-accounting-cases.json"
dest="src-tauri/tests/fixtures/codex-accounting-cases.json"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp"
cases=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["cases"]))' "$tmp")
cp "$tmp" "$dest"

if command -v sha256sum >/dev/null 2>&1; then
    sha=$(sha256sum "$dest" | cut -d' ' -f1)
else
    sha=$(shasum -a 256 "$dest" | cut -d' ' -f1)
fi
echo "copied $cases cases to $dest"
echo "MAC_CODEX_CASES_COMMIT = \"${ref:0:8}\""
echo "MAC_CODEX_CASES_SHA256 = \"$sha\""
