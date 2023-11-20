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
from modeling import QueryGenerator, Reranker, DualEncoder, DualEncoderMeanPooling
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
    with open(qp_file) as f:
        qry_psgs = f.readlines()
        samples = []
        labels = []
        pids = []
        for i in tqdm(range(len(qry_psgs))):
            try:
                qry, psg = qry_psgs[i].strip().split('\t')
                samples.append(promptor.build_prompt(psg))
                labels.append(qry)
                pids.append(i)
            except:
                print(qry_psgs[i])
    with open("output/samples.tsv","w") as f:
        f.writelines(samples[:1000])
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
    parser.add_argument('--gradient_checkpoint', type=bool, default=True)
    parser.add_argument('--add_decoder', type=bool, default=False)
    parser.add_argument('--negatives_x_device', type=bool, default=False)
    parser.add_argument('--negatives_in_device', type=bool, default=True)
    parser.add_argument('--reranker_warm_start_from', type=str, default="")
    parser.add_argument('--recalled_list', type=str, default="")


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

    train_samples, train_labels, train_pids = read_qp(args.collection, promptor) 
    p2p = np.load(args.recalled_list)
    idx = 2560
    tmp = p2p[:idx//4,:4,1].reshape(-1)
    eval_samples = [train_samples[pid] for pid in tmp]
    eval_labels = [train_labels[pid] for pid in tmp]
    eval_pids = [train_pids[pid] for pid in tmp]

    de_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
    retriever = DualEncoderMeanPooling(args)
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
        psgs = [psg.split("Passage:")[-1] for psg in prompts]
        q_records = de_tokenizer(outputs, padding=True, truncation=True, return_tensors="pt", max_length=64)
        p_records = de_tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=256)
        scores = retriever(_prepare_inputs(q_records), _prepare_inputs(p_records))
        rewards = compute_mrr(scores)
        return rewards
    trainer = trlx.train(
        reward_fn=rank_model_fn,
        prompts=train_samples,
        recalled_list=p2p[:,:,1],
        eval_prompts=eval_samples,
        eval_labels=eval_labels,
        eval_pids=eval_pids,
        config=config,
        promptor=promptor,
    )
    trainer.learn()

if __name__ == '__main__':
    ppo()
