#!/bin/bash
round=$1
query=$2
collection=$3
dataset=$4
top1000=round_${round}/top50.npy
retriever_model_name_or_path=../SAILER_zh
# retriever_model_name_or_path=../data/coco/coco-${dataset}-base/
if [ ${round} = 1 ]; then
    warm_start_from=''
else
    warm_start_from=round_$((round - 1))/best.p
fi

dataset=marco
sample_num=1
batch_size=256
dev_batch_size=256
max_seq_len=176
q_max_seq_len=512
p_max_seq_len=512
echo "batch size ${batch_size}"
max_index=64
FN_threshold=.9
# query=../flan-t5-xl-sft-generated/generated.tsv
# retriever_model_name_or_path=../data/contriever/
# retriever_model_name_or_path=../data/cocodr-base/
n_head_layers=0
p2p=../neighbors.P2P.DPR34-1.top50.npy
generated_query=${query}
# top1000=../marco_pretrain.tsv
learning_rate=2e-5
### 下面是永远不用改的
min_index=0
EN_threshold=-100
dev_query=../data/${dataset}/dev.query.txt
warmup_proportion=0.1
eval_step_proportion=0.01
report_step=1
epoch=200
qrels=../data/${dataset}/qrels.mrr43-5.tsv
fp16=true
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
gpu_partial=1
master_port=29500
# if [[ ${gpu_partial} =~ "0" ]]; then
#     export CUDA_VISIBLE_DEVICES=0,1,2,3
#     master_port=29500
# fi 
# if [[ ${gpu_partial} =~ "1" ]]; then
#     export CUDA_VISIBLE_DEVICES=4,5,6,7
#     master_port=29501
# fi 
# pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl 
# pip install transformers
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m torch.distributed.launch \
    --log_dir ${log_dir} \
    --nproc_per_node=8 \
    --master_port=${master_port} \
    src/train_dual_encoder.py \
    --vocab_file=${vocab_file} \
    --retriever_model_name_or_path=${retriever_model_name_or_path} \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --query=${query} \
    --dev_query=${dev_query} \
    --collection=${collection} \
    --top1000=${top1000} \
    --min_index=${min_index} \
    --max_index=${max_index} \
    --epoch=${epoch} \
    --sample_num=${sample_num} \
    --dev_batch_size=${dev_batch_size} \
    --pretrain_input_file=${pretrain_input_file} \
    --max_seq_len=${max_seq_len} \
    --learning_rate=${learning_rate} \
    --q_max_seq_len=${q_max_seq_len} \
    --p_max_seq_len=${p_max_seq_len} \
    --warm_start_from=${warm_start_from} \
    --n_head_layers=${n_head_layers} \
    --FN_threshold=${FN_threshold} \
    --EN_threshold=${EN_threshold} \
    --p2p=${p2p} \
    --generated_query=${generated_query} \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="

