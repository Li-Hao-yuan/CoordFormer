
TEST_CONFIGS='configs/test.yml'

GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.GPUS)
CUDA_VISIBLE_DEVICES=${GPUS} python -u romp/lib/evaluation/collect_3DPW_results.py --GPUS=${GPUS} --configs_yml=${TEST_CONFIGS} > '/data1/lihaoyuan/vomp/log/3dpw_p1'.log 2>&1 &

# sh scripts/test_3dpwchallenge.sh