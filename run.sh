#!/usr/bin/env bash
# tenring — avvio rapido (attiva l'env conda e lancia l'app a schermo intero).
# Uso:  ./run.sh            oppure doppio-click
#       ./run.sh --width 1280 --height 720   (più fps, meno risoluzione)
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tenring
cd "$(dirname "$0")"
exec python -m tenring.app --fullscreen "$@"
