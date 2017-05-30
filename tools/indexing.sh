#!/usr/bin/env sh
# Extraction & indexing of binary hash code + deep features (hdf5 file)
# By romyny, cbir_binary_code, MIT Licence
# Usage:
# ./tools/search.sh [args to index_bin48_feat.py]

set -x
set -e

export PYTHONUNBUFFERED="True"

USE_GPU=$1
MODEL_FILE=$2
FEAT_PROTO=$3
BIN_PROTO=$4
PROD_DIR=$5
DEEP_DB=$6
DATASET=$7

LOG="logs/${DATASET}_indexing_bin48_feat.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python tools/index_bin48_feat.py --use_gpu ${USE_GPU} \
 --model_file ${MODEL_FILE} \
 --feat_proto ${FEAT_PROTO} \
 --bin_proto ${BIN_PROTO} \
 --products_dir ${PROD_DIR} \
 --deep_db ${DEEP_DB}
