#!/bin/bash
# Sync the repository to Nibi. Run this on your LAPTOP, not on the cluster.
#
#     bash hpc/nibi/push.sh
#     bash hpc/nibi/push.sh <user>@nibi.alliancecan.ca:heatwave-initial-conditions/
#
# The no-argument form assumes a `nibi` host alias in ~/.ssh/config; pass the
# destination explicitly if you do not have one.
#
# Outputs stay on the cluster, so data/ and plots/ are not pushed. The one
# exception is the Koppen classification grid, which is a static input the
# pipeline reads from the checkout: without it every event fails at
# classify_event before any optimization starts.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${1:-nibi:heatwave-initial-conditions/}"

rsync -a --stats \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    --exclude '.DS_Store' \
    --exclude 'logs/' \
    --exclude 'outputs/' \
    --exclude 'plots/' \
    --exclude 'papers/' \
    --exclude 'tim_code/' \
    --include 'data/' \
    --include 'data/koppen_*' \
    --exclude 'data/*' \
    "$REPO/" "$REMOTE"

echo
echo "Pushed $REPO -> $REMOTE"
