%%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
model_root=/workspace/model
data_path=/workspace/CUB_200_2011
output_dir=/root/nlp/vpt_skill/output
        
# vtab-structured: dmlab
# base_lr = 1.0
# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
for seed in "43"; do
    python train.py \
        --config-file configs/prompt/cub.yaml \
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

# # vtab-natural: sun397
# # base_lr = 25
# # lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
# for seed in "42" "44" "82" "100" "800"; do
#     python train.py \
#         --config-file configs/prompt/cub.yaml \
#         MODEL.TYPE "vit" \
#         DATA.BATCH_SIZE "128" \
#         MODEL.PROMPT.NUM_TOKENS "5" \
#         MODEL.PROMPT.DEEP "True" \
#         MODEL.PROMPT.DROPOUT "0.1" \
#         DATA.FEATURE "sup_vitb16_imagenet21k" \
#         DATA.NAME "vtab-sun397" \
#         DATA.NUMBER_CLASSES "397" \
#         SOLVER.BASE_LR "12.5" \
#         SOLVER.WEIGHT_DECAY "0.0001" \
#         SOLVER.TOTAL_EPOCH "100" \
#         SEED ${seed} \
#         MODEL.MODEL_ROOT "${model_root}" \
#         DATA.DATAPATH "${data_path}" \
#         OUTPUT_DIR "${output_dir}/seed${seed}" 
# done

# # vtab-specialized: vtab-eurosat
# # base_lr = 1
# # lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
# for seed in "42" "44" "82" "100" "800"; do
#     python train.py \
#         --config-file configs/prompt/cub.yaml \
#         MODEL.TYPE "vit" \
#         DATA.BATCH_SIZE "64" \
#         MODEL.PROMPT.NUM_TOKENS "100" \
#         MODEL.PROMPT.DEEP "True" \
#         MODEL.PROMPT.DROPOUT "0.1" \
#         DATA.FEATURE "sup_vitb16_imagenet21k" \
#         DATA.NAME "vtab-eurosat" \
#         DATA.NUMBER_CLASSES "10" \
#         SOLVER.BASE_LR "0.25" \
#         SOLVER.WEIGHT_DECAY "0.001" \
#         SOLVER.TOTAL_EPOCH "100" \
#         SEED ${seed} \
#         MODEL.MODEL_ROOT "${model_root}" \
#         DATA.DATAPATH "${data_path}" \
#         OUTPUT_DIR "${output_dir}/seed${seed}" 
# done
