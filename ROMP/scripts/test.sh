# CUDA_VISIBLE_DEVICES=0 nohup python -u -m romp.lib.visualization.visualization --GPUS=0 --configs_yml='configs/v1.yml'
CUDA_VISIBLE_DEVICES=5 nohup python -u -m romp.test --GPUS=5 --configs_yml='configs/test.yml' > '/data1/lihaoyuan/vomp/log/test'.log 2>&1 &

# sh scripts/test.sh
