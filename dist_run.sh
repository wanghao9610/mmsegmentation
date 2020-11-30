#!/usr/bin/env bash

set -x

#CONFIG_FILE="configs/video/memory_gap_r50-d8_640x640_40k_camvid_video.py"
CONFIG_FILE="configs/video/memory_r50-d8_640x640_80k_camvid_video.py"
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
#CONFIG="test"
WORK_DIR="./work_dirs/${CONFIG}"
#WORK_DIR="./work_dirs/test"
SHOW_DIR="${WORK_DIR}/show"
TMPDIR="${WORK_DIR}/tmp"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULT_FILE="${WORK_DIR}/result.pkl"
#CHECKPOINT="${WORK_DIR}/iter_36000.pth,${WORK_DIR}/iter_40000.pth,${WORK_DIR}/iter_32000.pth"
GPUS=2
PORT=${PORT:-29511}

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\nconfig file: ${CONFIG}\n"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

RANDOM_SEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=4,5,6,7

# training
#echo -e '\nDistributed Training\n'
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    ./tools/train.py ${CONFIG_FILE} \
#    --seed $RANDOM_SEED \
#    --launcher 'pytorch' \
#    --work-dir $WORK_DIR \
#    --no-validate \
#    --resume-from $CHECKPOINT \

# evaluation
#echo -e "\nEvaluating ${WORK_DIR}\n"
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#  ./tools/test.py \
#  ${CONFIG_FILE} \
#  ${CHECKPOINT} \
#  --launcher 'pytorch' \
#  --eval mIoU \
#  --work-dir $WORK_DIR \
#  --tmpdir $TMPDIR
#    --out $RESULT_FILE \

# visualization
export CUDA_VISIBLE_DEVICES=1
echo -e '\nVisualization.\n'
if [ -d "${SHOW_DIR}" ]; then
  rm -rf "${SHOW_DIR}"
  mkdir "${SHOW_DIR}"
fi
python ./tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --show-dir $SHOW_DIR \
