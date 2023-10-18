export CUDA_VISIBLE_DEVICES=3

VERSION=2.0
BACKBONE=t5-base
TASK=e2e
DATASET=multiwoz


if [ $DATASET = 'multiwoz' ]; then
  EPOCH=8 # 0.01-20 | 0.05-10
  BATCH_SIZE=16
  SEED=-1
  LOGGIN_STEP=500
  LR=8e-4 # default 5e-4
  WARMUP_RATIO=0.3
  ADD_DU_DUAL=1
  ADD_RU_DUAL=1
  GRAD_ACCUM_STEPS=1
  DU_COEFF=0.1
  RU_COEFF=0.2
  TRAIN_RATIO=0.2
  PARA_NUM=0
  EX_DOMAINS="attraction"
  MEMO="case_analysis_dual"
  MODEL_DIR=outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}
elif [ $DATASET = 'incar' ]; then
  EPOCH=10
  SEED=-1 # 144720
  BATCH_SIZE=16
  LOGGIN_STEP=500
  LR=8e-4
  WARMUP_RATIO=0.4 # 0.4 best so far
  ADD_DU_DUAL=1
  ADD_RU_DUAL=1
  GRAD_ACCUM_STEPS=1
  DU_COEFF=0.1
  RU_COEFF=0.2
  TRAIN_RATIO=1.0
  PARA_NUM=1
  EX_DOMAINS="schedule"
  MEMO="few-shot5-para1"
  MODEL_DIR=outputs/${DATASET}_${BACKBONE}_${MEMO}_${CUDA_VISIBLE_DEVICES}
fi

python main.py \
  -backbone ${BACKBONE} \
  -dataset ${DATASET} \
  -task ${TASK} \
  -batch_size ${BATCH_SIZE} \
  -seed ${SEED} \
  -epochs $EPOCH \
  -learning_rate ${LR} \
  -warmup_ratio ${WARMUP_RATIO} \
  -version ${VERSION} \
  -add_du_dual ${ADD_DU_DUAL} \
  -add_ru_dual ${ADD_RU_DUAL} \
  -du_coeff ${DU_COEFF} \
  -ru_coeff ${RU_COEFF} \
  -grad_accum_steps ${GRAD_ACCUM_STEPS} \
  -train_ratio ${TRAIN_RATIO} \
  -para_num ${PARA_NUM} \
  -run_type train \
  -output results.json \
  -log_frequency ${LOGGIN_STEP} \
  -model_dir $MODEL_DIR/ \
  -memo ${MEMO} \

