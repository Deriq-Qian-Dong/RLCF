#!/bin/bash
task=$1
# lora_checkpoint_dir=lora-xl-beir/${task}.output/best_checkpoint/
lora_checkpoint_dir=checkpoints/giga.output/output/best_checkpoint/
batch_size=32
eval_batch_size=1024
echo "batch size ${batch_size}"
# model_name_or_path=best_checkpoint/
# model_name_or_path=../flan-t5-large/
model_name_or_path=../flan-t5-xl/
# model_name_or_path=../chatglm-6b-int4-qe/
# model_name_or_path=flan-t5-xxl-RLCL-MRR-87/
# model_name_or_path=flan-t5-large-few-shot-RLCL-MRR-85/
# model_name_or_path=../llama_7B
# recalled_list='../neighbors.P2P.DPR34-1.top50.dev.npy'
recalled_list='../neighbors.P2P.DPR34-1.top50.npy'
reranking_model_name_or_path=../data/electra-base-discriminator/
reranker_warm_start_from=../data/checkpoint_pt/teacher_electra_base_435/reranker.p
retriever_model_name_or_path=../data/distillBERT/
retriever_warm_start_from=26-9.p
learning_rate=1e-5
max_seq_len=160
# collection=../beir/${dataset}/collection.tsv
collection=../data/marco/dev_relevant_psgs.tsv 
# eval_qp=../segmented_passages.tsv
eval_qp=../beir/${task}/collection.tsv
# eval_qp=../beir/${task}/test.query.tsv
# eval_qp=../dev.query.tsv
# eval_qp=../data/marco/dev_sft.tsv
train_qp=../data/marco/dev_sft_with_hard_samples.tsv
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=29500 --config_file script/default_config.yaml\
    src/generate_trlx.py \
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
    --train_qp=${train_qp} \
    --recalled_list=${recalled_list} \
    --task=${task} \
    --lora_checkpoint_dir=${lora_checkpoint_dir} \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="


