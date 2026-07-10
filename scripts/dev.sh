#!/usr/bin/env bash
# Dev server with whatever assets exist so far.
set -euo pipefail
cd "$(dirname "$0")/.."
cd web && npm run dev -- --host
