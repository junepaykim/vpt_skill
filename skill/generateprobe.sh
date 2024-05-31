#!/bin/bash

model_root=/workspace/model
data_path=/workspace/CUB_200_2011
output_dir=/root/nlp/vpt_skill/output
attributes=("has_size::medium_(9_-_16_in)" "has_wing_color::yellow" "has_wing_color::red")
target_epoch="1"

for attribute in "${attributes[@]}"; do
    for target_seed in "61" "62" "63" "64"; do
        python /root/nlp/vpt_skill/sk_probe.py \
            --config-file /root/nlp/vpt_skill/configs/skill/cub_binary.yaml \
            DATA.BATCH_SIZE "32" \
            MODEL.PROMPT.NUM_TOKENS "64" \
            SOLVER.TOTAL_EPOCH "100" \
            MODEL.PROMPT.DEEP "False" \
            MODEL.PROMPT.DROPOUT "0.1" \
            DATA.FEATURE "sup_vitb16_imagenet21k" \
            DATA.NAME "CUB" \
            DATA.ATTRIBUTE "${attribute}" \
            SEED ${target_seed} \
            MODEL.MODEL_ROOT "${model_root}" \
            DATA.DATAPATH "${data_path}" \
            OUTPUT_DIR "${output_dir}/probe/${attribute}/s${target_seed}-e${target_epoch}" \
            PROMPT_DIR "${output_dir}/${attribute}/seed${target_seed}/lr0.002/run1/prompt_ep${target_epoch}.pth"
    done
done
