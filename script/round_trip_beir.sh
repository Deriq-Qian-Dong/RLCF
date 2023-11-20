#!/bin/bash
# for dataset in trec-covid nfcorpus nq hotpotqa fiqa dbpedia-entity scidocs fever scifact;do
generated_dir=chatglm-legal-summary
mkdir -p ${generated_dir}-result
for dataset in lecard cail;do
    query=${generated_dir}/${dataset}.generated.tsv
    echo ${query}
    collection=../beir/${dataset}/collection.summary.tsv
    echo ${collection}
    sh script/round_trip.sh ${query} ${collection} ${dataset}
    sh script/kill_gpu.sh 
    mkdir ${dataset}
    mv round_* ${dataset}
    mv ${dataset} ${generated_dir}-result
done 
