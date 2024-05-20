#!/bin/bash

SESSION_NAME="promptgenerate"

tmux new-session -d -s $SESSION_NAME

# SLURM 환경 변수 설정 및 스크립트 실행 명령 전송
tmux send-keys -t $SESSION_NAME 'export SLURMD_NODENAME="root"' C-m
tmux send-keys -t $SESSION_NAME 'conda activate prompt' C-m

tmux send-keys -t $SESSION_NAME 'export CUDA_VISIBLE_DEVICES=0,1,2' C-m

# `samplescript.bash`의 내용을 여기서 직접 실행
tmux send-keys -t $SESSION_NAME '
model_root=/workspace/model
data_path=/workspace/CUB_200_2011
output_dir=/root/nlp/vpt_skill/output

for seed in "44"; do
    python train.py \
        --config-file configs/skill/cub_binary.yaml \
        DATA.BATCH_SIZE "32" \
        MODEL.PROMPT.NUM_TOKENS "50" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "CUB" \
        DATA.NUMBER_CLASSES "200" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}" \
        MODEL.SAVE_CKPT "True"
done
' C-m

tmux attach-session -t $SESSION_NAME
