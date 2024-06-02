#!/bin/bash

model_root=/workspace/model
data_path=/workspace/CUB_200_2011
output_dir=/root/nlp/vpt_skill/output
attributes=("has_size::small_(5_-_9_in)" "has_size::very_large_(32_-_72_in)" "has_size::medium_(9_-_16_in)" "has_size::very_small_(3_-_5_in)" "has_breast_pattern::solid" "has_breast_pattern::spotted" "has_breast_pattern::striped" "has_breast_pattern::multi-colored" "has_wing_color::blue" "has_wing_color::brown" "has_wing_color::iridescent" "has_wing_color::purple" "has_wing_color::rufous" "has_wing_color::grey" "has_wing_color::yellow" "has_wing_color::olive" "has_wing_color::green" "has_wing_color::pink" "has_wing_color::orange" "has_wing_color::black" "has_wing_color::white" "has_wing_color::red" "has_wing_color::buff")
target_epoch="80"
lr="0.0025"

for attribute in "${attributes[@]}"; do
    for target_seed in "70" "71" "72" "73" "74"; do
        python /root/nlp/vpt_skill/sk_probe.py \
            --config-file /root/nlp/vpt_skill/configs/skill/cub_binary.yaml \
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
            PROMPT_DIR "${output_dir}/${attribute}/seed${target_seed}/lr${lr}/run1/prompt_ep${target_epoch}.pth"
    done
done
