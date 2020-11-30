#!/usr/bin/env bash

set -x
unset SLURM_JOB_ID
unset SLURM_JOBID

CONFIG_FILE="configs/video/memory_gap_guide_r50-d8_769x769_80k_cityscapes_video.py"
#CONFIG_FILE="configs/video/memory_r50-d8_640x640_80k_camvid_video.py"
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
WORK_DIR="./work_dirs/${CONFIG}"
#WORK_DIR="./work_dirs/test"
SHOW_DIR="${WORK_DIR}/show"
TMPDIR="${WORK_DIR}/tmp"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULT_FILE="${WORK_DIR}/result.pkl"

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\nconfig file: ${CONFIG_FILE}\n"

# slurm arguments
PARTITION=q02
JOB_NAME="video"
GPUS=4
GPUS_PER_NODE=2
CPUS_PER_TASK=4
NODELIST=g23,g24
#NODELIST=g21,g22
#NODELIST=g24
#--gres=gpu:${GPUS_PER_NODE} \
# training
RANDOM_SEED=0

echo -e '\nDistributed Training With Slurm.\n'
srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --nodelist=${NODELIST} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  python -u tools/train.py ${CONFIG_FILE} \
  --seed $RANDOM_SEED \
  --launcher "slurm" \
  --work-dir $WORK_DIR \
#  --options 'dist_params.port=29510' \
#  --resume-from $CHECKPOINT

# evaluation
echo -e "\nWork Dir: ${WORK_DIR}.\n"
#echo -e "\nEvaluation With Slurm.\n"
#srun -p ${PARTITION} \
#  --job-name=${JOB_NAME} \
#  --nodelist=${NODELIST} \
#  --ntasks=${GPUS} \
#  --ntasks-per-node=${GPUS_PER_NODE} \
#  --cpus-per-task=${CPUS_PER_TASK} \
#  --kill-on-bad-exit=1 \
#  python -u tools/test.py \
#  ${CONFIG_FILE} \
#  ${CHECKPOINT} \
#  --launcher="slurm" \
#  --eval mIoU \
#  --work-dir $WORK_DIR \
#  --tmpdir $TMPDIR
#    --out $RESULT_FILE \


# visualization
echo -e "\nWork Dir: ${WORK_DIR}.\n"
echo -e '\nVisualization\n'
if [ -d "${SHOW_DIR}" ]; then
  rm -rf "${SHOW_DIR}"
  mkdir "${SHOW_DIR}"
fi
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --nodelist=${NODELIST%,*} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u ./tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --show-dir $SHOW_DIR \

#echo -e "\nWork Dir: ${WORK_DIR}.\n"
