#!/usr/bin/env bash
# Stage 1 in one Kaggle GPU session: both SSL-AASIST checkpoints on every set, then
# the silence classifier. Notebook inputs: the ASVspoof 2019 mirror and the private
# dataset made by fetch_data.sh. Rerun skips sets that are already scored.
set -euo pipefail
input=${1:-/kaggle/input}
out=${2:-/kaggle/working/stage1}
# Unpacked In-the-Wild is 8 GB and must stay out of the saved working folder.
scratch=${3:-/tmp}
repo=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$repo
mkdir -p "$out"

# Walking 120k flac files on the input disk is slow, and nothing is searched there.
find_input() {
    find -L "$input" -name flac -prune -o "$@" -print -quit
}

need() {
    local path
    path=$(find_input "$@")
    if [ -z "$path" ]; then
        echo "not found in $input: $*" >&2
        return 1
    fi
    echo "$path"
}

protocols=$(need -name ASVspoof2019_LA_cm_protocols)
asvspoof=$(dirname "$protocols")
etrim=$(need -name LA2019-sil-trim.log)
la_model=$(need -name LA_model.pth)
df_model=$(need -name Best_LA_model_for_DF.pth)

# Kaggle may keep the In-the-Wild archive as is or unpack it into the dataset.
meta=$(find_input -path '*release_in_the_wild/meta.csv')
if [ -z "$meta" ]; then
    zip=$(need -name release_in_the_wild.zip)
    unpacked=$scratch/itw
    meta=$unpacked/release_in_the_wild/meta.csv
    if [ ! -f "$meta" ]; then
        echo "unpacking $zip"
        python -m zipfile -e "$zip" "$unpacked"
    fi
fi
itw=$(dirname "$meta")

{
    git -C "$repo" log -1 --format='commit %h %s'
    python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
    nvidia-smi -L 2>/dev/null || echo "no GPU"
    echo "asvspoof $asvspoof"
    echo "itw $itw"
} | tee "$out/env.txt"

# A failed step must not stop the rest: Kaggle may drop the output of a failed run.
failed=()
step() {
    local name=$1
    shift
    echo "== $name"
    "$@" || failed+=("$name")
}

score() {
    step "$1" python "$repo/scripts/score.py" --name "$1" --checkpoint "$2" \
        --asvspoof "$asvspoof" --itw "$itw" --etrim "$etrim" --out "$out"
}

score ssl_aasist_la "$la_model"
score ssl_aasist_df "$df_model"
step silence_clf python "$repo/scripts/silence_clf.py" \
    --asvspoof "$asvspoof" --itw "$itw" --out "$out"

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
else
    echo "all done, results in $out"
fi
