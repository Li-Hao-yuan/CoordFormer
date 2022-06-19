# TRAIN_CONFIGS='configs/v1.yml'

# GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
# DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
# TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=0 nohup python -u -m romp.train --GPUS=0 --configs_yml='configs/v1.yml' > 'C:/Users/Administrator/Desktop/vomp/log/hrnet_3pdw_g0'.log 2>&1 &