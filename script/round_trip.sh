#!/bin/bash
query=$1
collection=$2
dataset=$3
for round in 1;do
    echo $((round - 1))
    mkdir round_${round} 
    sh script/filter.sh ${round} ${query} ${collection} ${dataset}
    sh script/kill_gpu.sh
    mv top50.npy round_${round}
    mv output/log/train.log round_${round}/filter.log 
    sh script/train_dual_encoder.sh ${round} ${query} ${collection} ${dataset}
    mv output/log/train.log round_${round}/train.log 
    mv output/best.p round_${round}/best.p
    sh script/kill_gpu.sh
done 
