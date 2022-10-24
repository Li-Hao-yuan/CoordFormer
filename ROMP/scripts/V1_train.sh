# TRAIN_CONFIGS='configs/v1.yml' -> sh scripts/V1_train.sh

# GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
# DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
# TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

# CUDA_VISIBLE_DEVICES=2,3 nohup python -u -m romp.train --GPUS=2,3 --configs_yml='configs/v1.yml' > '/home/lihaoyuan/vomp/log/hrnet_3pdw_g2,3'.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3,4,5 nohup python -u -m romp.train --GPUS=3,4,5 --configs_yml='configs/v1.yml' > '/data1/lihaoyuan/vomp/log/resnet_3pdw_g0'.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -m romp.train --GPUS=2 --configs_yml='configs/v1_another.yml' > '/data1/lihaoyuan/vomp/log/another_resnet_3pdw_g0'.log 2>&1 &

# sh scripts/V1_train.sh
# kill pid

# python -u -m romp.train_video --GPUS=0 --configs_yml='configs/v1_video.yml' > '/data1/lihaoyuan/vomp/log/resnet_3pdw_only_video_g0'.log 2>&1 &


