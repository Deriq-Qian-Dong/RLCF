#!/bin/bash
# for dataset in trec-covid nfcorpus nq hotpotqa fiqa dbpedia-entity scidocs fever scifact;do
generated_dir=generated
mkdir -p ${generated_dir}-result
for dataset in lecard cail;do
    query=${generated_dir}/${dataset}.generated.tsv
    echo ${query}
    collection=../beir/${dataset}/collection.tsv
    echo ${collection}
    sh script/train_dual_encoder.sh 1 ${query} ${collection} ${dataset}
    sh script/kill_gpu.sh 
    mv output/log/train.log ${generated_dir}-result/${dataset}.log
done 
