# Multijugate Dual Learning for Low-Resource Task-Oriented Dialogue System

This is the code for the paper "Multijugate Dual Learning for Low-Resource Task-Oriented Dialogue System".

## Environment setting

Our python version is 3.7.11.

The package to reproduce the results can be installed by running the following command.

```
pip install -r requirements.txt
```

## Data Preprocessing

For the experiments, we use MultiWOZ2.0 and MultiWOZ2.1.

- (MultiWOZ2.0) annotated_user_da_with_span_full.json: A fully annotated version of the original MultiWOZ2.0 data released by developers of Convlab
  available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).
- (MultiWOZ2.1) data.json: The original MultiWOZ 2.1 data released by researchers in University of Cambrige
  available [here](https://github.com/budzianowski/multiwoz/tree/master/data).

> We have preprocessed the MultiWOZ2.0 and MultiWOZ2.1 and uploaded the data for fast reproduce, so this command can be skipped.

```
python preprocess.py -version $VERSION
```


## Training and Prediction

We uploaded the pre-processed MultiWOZ2.0 and MultiWOZ2.1 for fast reproduce, so we can directly start training and the results will be output when
the training is complete.

- We use ```fitlog``` as training monitor for visualization and results recoder, so please run the following command before training:
```shell
fitlog init
```

- For fast reproduce of end-to-end MultiWOZ2.0 under full-setting, you may check the following hyper-parameters in ```train.sh```:

```shell
VERSION=2.0
TASK=e2e
SEED=709635
BATCH_SIZE=4
EPOCH=10
LR=2e-4
ADD_DU_DUAL=1
ADD_RU_DUAL=1
DU_COEFF=0.1
RU_COEFF=0.2
TRAIN_RATIO=1.0
PARA_NUM=0
```

If all goes well, the following results will be obtained.

|       | Inform | Success | BLEU  | Combined | 
|-------|--------|---------|-------|----------|
| MDTOD | 92.80  | 84.70   | 19.83 | 108.58   | 

- For fast reproduce of end-to-end MultiWOZ2.0 under 5% training set, you may check the following hyper-parameters in ```train.sh```:

```shell
VERSION=2.0
TASK=e2e
SEED=242094
BATCH_SIZE=4
EPOCH=15
LR=5e-4
ADD_DU_DUAL=1
ADD_RU_DUAL=1
DU_COEFF=0.1
RU_COEFF=0.2
TRAIN_RATIO=0.05
PARA_NUM=2
```


If all goes well, the following results will be obtained.

|       | Inform | Success | BLEU  | Combined | 
|-------|--------|---------|-------|----------|
| MDTOD | 88.00  | 64.90   | 14.20 | 90.65    | 

- For fast reproduce of end-to-end MultiWOZ2.1 under full-setting, you may check the following hyper-parameters in ```train.sh```:

```shell
VERSION=2.1
TASK=e2e
SEED=326210
BATCH_SIZE=4
EPOCH=10
LR=2e-4
ADD_DU_DUAL=1
ADD_RU_DUAL=1
DU_COEFF=0.1
RU_COEFF=0.2
TRAIN_RATIO=1.0
PARA_NUM=0
```

If all goes well, the following results will be obtained.

|       | Inform | Success | BLEU  | Combined | 
|-------|--------|---------|-------|----------|
| MDTOD | 92.70  | 84.60   | 19.03 | 107.68   | 



- For fast reproduce of dialogue state tracking of MultiWOZ2.0, you may check the following hyper-parameters in ```dst_train.sh```:

```shell
VERSION=2.0
TASK=dst
SEED=677514
BATCH_SIZE=6
EPOCH=8
LR=5e-4
ADD_DU_DUAL=1
ADD_RU_DUAL=0
DU_COEFF=0.2
RU_COEFF=0.1
TRAIN_RATIO=1.0
PARA_NUM=1

```

If all goes well, the following results will be obtained.

|       | JGA   | f1    | ACC   |
|-------|-------|-------|-------|
| MDTOD | 54.41 | 91.20 | 97.09 | 

Checkpoints are saved after each epoch and only the latest five checkpoints are retained.

## Evaluation

- For fast reproduce of the results, we output the predictions as soon as the training is complete, so there is no need to run additional prediction
  scripts.

## Acknowledgements

This code is referenced from the [MTTOD](https://github.com/bepoetree/MTTOD) implementation, and we appreciate their contribution.
