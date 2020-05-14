ROOT=/Users/yaoxiaoying/Documents/01-工作/03.计算机视觉/03.智慧交通/04.单目标追踪/SiamMask-master
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siammask.py \
    --config=config.json -b 8 \
    -j 0 \
    --epochs 8 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

# bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
