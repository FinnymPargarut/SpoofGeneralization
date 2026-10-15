#!/usr/bin/env bash
# Stage 1 inputs that are not on Kaggle. Run once in a Kaggle notebook with internet on,
# then turn the notebook output into a private dataset.
set -euo pipefail
out=${1:-/kaggle/working}
mkdir -p "$out"
cd "$out"

# Kept zipped: notebook output may be limited in the number of files.
curl -fsSL -o release_in_the_wild.zip \
    https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip
echo "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a  release_in_the_wild.zip" | sha256sum -c

curl -fsSL -o etrim.tar "https://zenodo.org/records/7314976/files/project09-ASVspoof2019LA-etrim.tar?download=1"
echo "e7faf53367b19c90d136fff806e08ee0  etrim.tar" | md5sum -c
tar xf etrim.tar
rm etrim.tar

# Google Drive gives no checksums and may answer with an HTML page, so check the size.
fetch_drive() {
    curl -fsSL -o "$1" "https://drive.usercontent.google.com/download?id=$2&export=download&confirm=t"
    test "$(stat -c %s "$1")" -eq "$3"
}
fetch_drive LA_model.pth 11vFBNKzYUtWqdK358_JEygawFzmmaZjO 1271633441
fetch_drive Best_LA_model_for_DF.pth 1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG 1271642081
sha256sum LA_model.pth Best_LA_model_for_DF.pth > checkpoints.sha256

ls -l "$out"
