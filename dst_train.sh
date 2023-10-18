export CUDA_VISIBLE_DEVICES=2

VERSION=2.1
BACKBONE=t5-base
TASK=dst
SEED=-1
DATASET=multiwoz

if [ $DATASET = 'multiwoz' ]; then
  EPOCH=8
  BATCH_SIZE=6
  LOGGIN_STEP=500
  LR=5e-4 # default 5e-4
  ADD_DU_DUAL=1
  ADD_RU_DUAL=0
  GRAD_ACCUM_STEPS=1
  DU_COEFF=0.2
  RU_COEFF=0.1
  TRAIN_RATIO=1.0
  PARA_NUM=1
  MEMO="ours"
  MODEL_DIR=outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}
fi

python main.py \
  -backbone ${BACKBONE} \
  -dataset ${DATASET} \
  -task ${TASK} \
  -batch_size ${BATCH_SIZE} \
  -seed ${SEED} \
  -epochs $EPOCH \
  -learning_rate ${LR} \
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
  -ururu \
  -use_true_dbpn \
  -use_true_prev_aspn \
  -use_true_prev_resp \
  -memo ${MEMO} #
