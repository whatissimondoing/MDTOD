export CUDA_VISIBLE_DEVICES=7

VERSION=fused
BACKBONE=t5-base
TASK=e2e
DATASET=multiwoz
SEED=763904

main.py \
  -run_type predict \
  -backbone ${BACKBONE} \
  -dataset ${DATASET} \
  -seed ${SEED} \
  -task ${TASK} \
  -version ${VERSION} \
  -ckpt outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}/ckpt-epoch9 \
  -output results.json \
  -batch_size 98
