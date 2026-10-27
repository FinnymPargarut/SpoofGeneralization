#!/usr/bin/env bash
# Stage 2 on a Kaggle GPU session. Part frozen: SSL-AASIST on the stage 2 sets, the
# RuASD shortcut audit, frozen XLS-R features and logistic regression. Part lcnn:
# LFCC-LCNN with three seeds. Part all: both. Notebook inputs: the ASVspoof 2019
# mirror, the stage 1 and stage 2 inputs datasets; internet on for the XLS-R weights.
# A rerun skips finished steps.
set -euo pipefail
part=${1:-all}
input=${2:-/kaggle/input}
out=${3:-/kaggle/working/stage2}
# Unpacked In-the-Wild and stage 2 audio must stay out of the saved working folder.
scratch=${4:-/tmp}
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

# Kaggle may keep an uploaded zip as is or unpack it into the dataset.
unzip_once() {
    local zip=$1 dest=$2
    echo "unpacking $zip"
    python -m zipfile -e "$zip" "$dest"
}

protocols=$(need -name ASVspoof2019_LA_cm_protocols)
asvspoof=$(dirname "$protocols")
etrim=$(need -name LA2019-sil-trim.log)

meta=$(find_input -path '*release_in_the_wild/meta.csv')
if [ -z "$meta" ]; then
    meta=$scratch/itw/release_in_the_wild/meta.csv
    [ -f "$meta" ] || unzip_once "$(need -type f -name release_in_the_wild.zip)" "$scratch/itw"
fi
itw=$(dirname "$meta")

stage2=$scratch/stage2
mkdir -p "$stage2"
for name in ruasd librispeech ruls; do
    cp "$(need -name "$name.csv")" "$stage2/"
    [ -e "$stage2/$name" ] && continue
    zip=$(find_input -type f -name "$name.zip")
    if [ -n "$zip" ]; then
        unzip_once "$zip" "$stage2"
    else
        first=$(need -name '*.flac' -path "*/$name/*")
        ln -s "$(dirname "$first")" "$stage2/$name"
    fi
done

{
    git -C "$repo" log -1 --format='commit %h %s'
    python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
    nvidia-smi -L 2>/dev/null || echo "no GPU"
    echo "asvspoof $asvspoof"
    echo "itw $itw"
    echo "stage2 $stage2"
} | tee "$out/env.txt"

# A failed step must not stop the rest: Kaggle may drop the output of a failed run.
failed=()
step() {
    local name=$1
    shift
    echo "== $name"
    "$@" || failed+=("$name")
}

inputs=(--asvspoof "$asvspoof" --itw "$itw" --etrim "$etrim" --stage2 "$stage2")
new_sets=(itw_trim40 ruasd librispeech librispeech_trim40 ruls ruls_trim40)

frozen() {
    local la_model df_model
    la_model=$(need -name LA_model.pth)
    df_model=$(need -name Best_LA_model_for_DF.pth)
    step ssl_aasist_la python "$repo/scripts/score.py" --model ssl_aasist \
        --name ssl_aasist_la --checkpoint "$la_model" --sets "${new_sets[@]}" \
        "${inputs[@]}" --out "$out"
    step ssl_aasist_df python "$repo/scripts/score.py" --model ssl_aasist \
        --name ssl_aasist_df --checkpoint "$df_model" --sets "${new_sets[@]}" \
        "${inputs[@]}" --out "$out"
    step ruasd_audit python "$repo/scripts/silence_clf.py" --stage2 "$stage2" --out "$out"
    step xlsr_features python "$repo/scripts/xlsr_features.py" \
        --sets train dev eval itw "${new_sets[@]}" "${inputs[@]}" --out "$out/features"
    step frozen_lr python "$repo/scripts/frozen_lr.py" --features "$out/features" --out "$out"
}

lcnn() {
    local seed name
    for seed in 0 1 2; do
        name=lcnn_seed$seed
        step "$name" python "$repo/scripts/train_lcnn.py" --asvspoof "$asvspoof" \
            --out "$out/$name" --seed "$seed"
        step "${name}_scores" python "$repo/scripts/score.py" --model lcnn \
            --name "$name" --checkpoint "$out/$name/model.pt" \
            --sets dev eval itw "${new_sets[@]}" "${inputs[@]}" --out "$out"
    done
}

case $part in
    frozen) frozen ;;
    lcnn) lcnn ;;
    all) frozen; lcnn ;;
    *) echo "unknown part $part" >&2; exit 1 ;;
esac

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
else
    echo "all done, results in $out"
fi
