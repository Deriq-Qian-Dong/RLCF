import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import argparse
import random
import subprocess
import tempfile
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import distributed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
tqdm.pandas()
from transformers import AutoConfig, AutoTokenizer, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import trlx
from trlx.trainer.accelerate_sft_trainer import SFTConfig
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
from config import default_t5_ppo_config

import dataset_factory
import utils
from modeling import QueryGenerator, Reranker, DualEncoder
from utils import add_prefix, build_engine, load_qid, merge, read_embed, search
from accelerate import Accelerator
from promptor import Promptor

SEED = 2023
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

def compute_mrr(scores):
    # 仅保留每一行对应位置的得分
    relevant_scores = torch.diag(scores).unsqueeze(1)
    # 将每行得分与相关passage的得分进行比较，计算排名
    ranks = (scores >= relevant_scores).sum(dim=1).float()
    # 计算每个query的MRR
    reciprocal_ranks = 1.0 / ranks
    # 计算所有query的平均MRR
    return reciprocal_ranks.tolist()

def read_qp(qp_file, promptor):
    # instruction = "Write a question related to this document: %s"  比如
    print(qp_file)
    with open(qp_file,encoding='utf8') as f:
        qry_psgs = f.readlines()
        samples = []
        labels = []
        pids = []
        for i in tqdm(range(len(qry_psgs))):
            qry, psg = qry_psgs[i].strip().split('\t')
            samples.append(promptor.build_prompt(psg))
            labels.append(qry)
            pids.append(i)
    with open("output/samples.tsv","w") as f:
        f.writelines(samples)
    return samples, labels, pids

def define_args():
    parser = argparse.ArgumentParser('BERT-retrieval model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--model_name_or_path', type=str, default="../data/co-condenser-marco-retriever/")
    parser.add_argument('--reranking_model_name_or_path', type=str, default="../data/co-condenser-marco-retriever/")
    parser.add_argument('--retriever_warm_start_from', type=str, default="")
    parser.add_argument('--retriever_model_name_or_path', type=str, default="../data/distillBERT/")
    parser.add_argument('--model_out_dir', type=str, default="output")
    parser.add_argument('--train_qp', type=str, default="../data/marco/train_sft.tsv")
    parser.add_argument('--eval_qp', type=str, default="../data/marco/dev_sft.tsv")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--collection', type=str, default="/home/dongqian06/hdfs_data/data_train/marco/collection.remains.tsv")
    parser.add_argument('--task', type=str, default="/home/dongqian06/hdfs_data/data_train/marco/collection.remains.tsv")
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--gradient_checkpoint', type=bool, default=False)
    parser.add_argument('--add_decoder', type=bool, default=False)
    parser.add_argument('--negatives_x_device', type=bool, default=False)
    parser.add_argument('--negatives_in_device', type=bool, default=True)
    parser.add_argument('--reranker_warm_start_from', type=str, default="")
    parser.add_argument('--recalled_list', type=str, default="")
    parser.add_argument('--lora_checkpoint_dir', type=str, default="")

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args 

def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        elif x is None:
            prepared[key] = x
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared

def ppo():
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank==0:
        args.print_config()
    promptor = Promptor(args.task)
    config = default_t5_ppo_config(promptor, model_name_or_path=args.model_name_or_path, learning_rate=args.learning_rate)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    train_samples, train_labels, train_pids = read_qp(args.collection, args)
    eval_samples, eval_labels, eval_pids = read_qp(args.eval_qp, args) # 在测试集上做RLHF
    # idx = 2560
    # eval_samples = eval_samples[:idx]
    # eval_labels = eval_labels[:idx]
    # eval_pids = eval_pids[:idx]

    de_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
    retriever = DualEncoder(args)
    retriever.eval()
    # accelerator = Accelerator()
    if args.retriever_warm_start_from:
        print('retriever warm start from ', args.retriever_warm_start_from)
        state_dict = torch.load(args.retriever_warm_start_from, map_location='cpu')
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','')] = state_dict.pop(k)
        retriever.load_state_dict(state_dict)

    retriever.to(device)
    os.makedirs(args.model_out_dir, exist_ok=True)
    @torch.no_grad()
    def rank_model_fn(samples, prompts, outputs):
        '''
        samples: [instruction:passage query]
        prompts: [instruction:passage]
        outputs: [query]
        '''
        # psgs = [psg[len(instruction)-2:] for psg in prompts]
        psgs = [psg.split("\n")[-1][promptor.skip_length:] for psg in prompts]
        q_records = de_tokenizer(outputs, padding=True, truncation=True, return_tensors="pt", max_length=32)
        p_records = de_tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=160)
        scores = retriever(_prepare_inputs(q_records), _prepare_inputs(p_records))
        rewards = compute_mrr(scores)
        return rewards
    recalled_list = np.load(args.recalled_list)
    trainer = trlx.train(
        reward_fn=rank_model_fn,
        prompts=train_samples,
        recalled_list=recalled_list,
        eval_prompts=eval_samples,
        eval_labels=eval_labels,
        eval_pids=eval_pids,
        config=config,
    )
    trainer.learn()


def sft():
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank==0:
        args.print_config()
    promptor = Promptor(args.task)
    config = default_t5_ppo_config(promptor, model_name_or_path=args.model_name_or_path, learning_rate=args.learning_rate)
    # instruction = "Please rephrase this document: %s"
    # instruction = "Please write the title of this document: %s"
    # instruction = "Please summarize this document: %s"
    # instruction = "Please answer the question step by step: %s"
    eval_samples, eval_labels, eval_pids = read_qp(args.eval_qp, promptor)
    train_samples, train_labels, train_pids = eval_samples, eval_labels, eval_pids
    # idx = 2560
    # eval_samples = eval_samples[:idx]
    # eval_labels = eval_labels[:idx]
    # eval_pids = eval_pids[:idx]

    # ce_tokenizer = AutoTokenizer.from_pretrained(args.reranking_model_name_or_path)
    # reranker = Reranker(args)
    # reranker.eval()
    # accelerator = Accelerator(mixed_precision='fp16')
    # if args.reranker_warm_start_from:
        # print('reranker warm start from ', args.reranker_warm_start_from)
        # state_dict = torch.load(args.reranker_warm_start_from, map_location='cpu')
        # reranker.lm.load_state_dict(state_dict)
    os.makedirs(args.model_out_dir, exist_ok=True)
    # # reranker.cuda()
    @torch.no_grad()
    def rank_model_fn(samples, prompts, outputs):
        '''
        samples: [instruction:passage query]
        prompts: [instruction:passage]
        outputs: [query]
        '''
        reranker.cuda()
        psgs = [psg.split("\n")[-1][promptor.skip_length:] for psg in prompts]
        features = ce_tokenizer(outputs, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=args.max_seq_len) 
        rewards = reranker(_prepare_inputs(features))
        rewards = [r.detach() for r in rewards]
        return rewards
    config.model.checkpoint_dir = args.lora_checkpoint_dir
    trainer = trlx.train(
        metric_fn=None,
        samples=train_samples,
        train_labels=train_labels,
        train_pids=train_pids,
        eval_prompts=eval_samples,
        eval_labels=eval_labels,
        eval_pids=eval_pids,
        config=config,
        promptor=promptor,
    )
    trainer.inference4qgen()
    # trainer.learn()


if __name__ == '__main__':
    # ppo()
    sft()
