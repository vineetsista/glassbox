#!/usr/bin/env bash
# Publish web/dist to the gh-pages branch (GitHub Pages).
# Run scripts/build_site.sh first. Model weights and baked dashboards are
# generated artifacts: they live on gh-pages, never on main.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f web/dist/index.html ] || { echo "web/dist missing - run scripts/build_site.sh"; exit 1; }

tmp=$(mktemp -d)
trap "rm -rf $tmp" EXIT
cp -r web/dist/* "$tmp/"
touch "$tmp/.nojekyll"

cd "$tmp"
git init -q -b gh-pages
git config user.email "vineet.sista@gmail.com"
git config user.name "Vineet Sista"
git add -A
git commit -q -m "deploy site $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push -f "https://github.com/vineetsista/glassbox.git" gh-pages
echo "deployed. Enable Pages (branch gh-pages) once:"
echo "  gh api -X POST repos/vineetsista/glassbox/pages -f 'source[branch]=gh-pages' -f 'source[path]=/'"
