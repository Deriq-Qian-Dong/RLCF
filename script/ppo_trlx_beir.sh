#!/bin/bash
# for dataset in trec-covid nfcorpus fiqa dbpedia-entity scidocs fever scifact;do
mkdir checkpoints
for dataset in giga;do
    sh script/ppo_trlx.sh ${dataset}
    sh script/kill_gpu.sh 
    mv output/ checkpoints/${dataset}.output/
done 
