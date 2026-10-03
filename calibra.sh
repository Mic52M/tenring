#!/usr/bin/env bash
# tenring — calibrazione personale (mettiti in posizione di tiro, braccio su).
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tenring
cd "$(dirname "$0")"
exec python -m tenring.calibrate "$@"
