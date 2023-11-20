#!/bin/bash
# for dataset in trec-covid nfcorpus hotpotqa fiqa dbpedia-entity scidocs fever scifact;do
mkdir -p generated
for dataset in giga;do
    sh script/generate_trlx.sh ${dataset}
    sh script/kill_gpu.sh 
    mv output/generated.tsv generated/${dataset}.generated.tsv
done 
