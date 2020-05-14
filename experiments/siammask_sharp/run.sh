if [ -z "$1" ]
  then
    echo "Need input base model!"
    echo "Usage: bash `basename "$0"` \$BASE_MODEL"
    exit
fi


ROOT=/Users/yaoxiaoying/Documents/01-工作/03.计算机视觉/03.智慧交通/04.单目标追踪/SiamMask-master
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

base=$1

python -u $ROOT/tools/train_siammask_refine.py \
    --config=config.json -b 8\
    -j 0 --pretrained $base \
    --epochs 8 \
    2>&1 | tee logs/train.log
