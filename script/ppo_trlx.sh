#!/bin/bash
task=$1
batch_size=32
eval_batch_size=1024
echo "batch size ${batch_size}"
# model_name_or_path=best_checkpoint/
# model_name_or_path=../flan-t5-large/
model_name_or_path=../flan-t5-xl/
# model_name_or_path=../llama_7B_zh/
# model_name_or_path=../BELLE-7B-2M/
# model_name_or_path=../Ziya-LLaMA-13B-v1/
# model_name_or_path=../BELLE-LLaMA-EXT-7B-decrypted/
recalled_list=../beir/p2p/top50.${task}.npy
reranking_model_name_or_path=../data/electra-base-discriminator/
reranker_warm_start_from=../data/checkpoint_pt/teacher_electra_base_435/reranker.p
# retriever_model_name_or_path=../data/SAILER_zh/
retriever_model_name_or_path=../data/contriever/
# retriever_warm_start_from=../data/T2Ranking/dual-encoder-trained-with-hard-negatives.p
learning_rate=3e-4
max_seq_len=160
collection=../beir/${task}/collection.tsv
eval_qp=../beir/${task}/collection.tsv
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=65533 --config_file script/default_config.yaml\
    src/ppo_trlx.py \
    --model_name_or_path=${model_name_or_path} \
    --batch_size=${batch_size} \
    --eval_batch_size=${eval_batch_size} \
    --collection=${collection} \
    --learning_rate=${learning_rate} \
    --max_seq_len=${max_seq_len} \
    --reranking_model_name_or_path=${reranking_model_name_or_path} \
    --reranker_warm_start_from=${reranker_warm_start_from} \
    --retriever_model_name_or_path=${retriever_model_name_or_path} \
    --retriever_warm_start_from=${retriever_warm_start_from} \
    --eval_qp=${eval_qp} \
    --recalled_list=${recalled_list} \
    --task=${task} \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="


