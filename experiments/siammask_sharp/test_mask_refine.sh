if [ -z "$4" ]
  then
    echo "Need input parameter!"
    echo "Usage: bash `basename "$0"` \$CONFIG \$MODEL \$DATASET \$GPUID"
    exit
fi

ROOT=/Users/yaoxiaoying/Documents/01-工作/03.计算机视觉/03.智慧交通/04.单目标追踪/SiamMask-master
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

config=$1
model=$2
dataset=$3
gpu=$4

CUDA_VISIBLE_DEVICES=$gpu python -u $ROOT/tools/test.py \
    --config $config \
    --resume $model \
    --mask --refine \
    --dataset $dataset 2>&1 | tee logs/test_$dataset.log

