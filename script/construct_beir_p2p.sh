#!/bin/bash
# for dataset in trec-covid nfcorpus hotpotqa fiqa dbpedia-entity scidocs fever scifact;do
mkdir p2p
for dataset in giga;do
    sh script/construct_p2p.sh ${dataset}
    sh script/kill_gpu.sh 
    mv top50.npy p2p/top50.${dataset}.npy
done 
