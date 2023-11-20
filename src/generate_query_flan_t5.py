import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import argparse
import random
import subprocess
import tempfile
import time
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import distributed
import torch_optimizer as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

import dataset_factory
import utils
from modeling import QueryGenerator
from msmarco_eval import calc_mrr
from utils import add_prefix, build_engine, load_qid, merge, read_embed, search

SEED = 2023
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
def define_args():
    parser = argparse.ArgumentParser('BERT-retrieval model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dev_batch_size', type=int, default=2)
    parser.add_argument('--model_name_or_path', type=str, default="../data/co-condenser-marco-retriever/")
    parser.add_argument('--model_out_dir', type=str, default="output")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=1.0)
    parser.add_argument('--report', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--qrels', type=str, default="/home/dongqian06/hdfs_data/data_train/qrels.train.remains.tsv")
    parser.add_argument('--top1000', type=str, default="/home/dongqian06/codes/anserini/runs/run.msmarco-passage.train.remains.tsv")
    parser.add_argument('--collection', type=str, default="/home/dongqian06/hdfs_data/data_train/marco/collection.remains.tsv")
    parser.add_argument('--query', type=str, default="/home/dongqian06/hdfs_data/data_train/train.query.remains.txt")
    parser.add_argument('--dev_query', type=str, default="/home/dongqian06/hdfs_data/data_train/train.query.remains.txt")
    parser.add_argument('--min_index', type=int, default=0)
    parser.add_argument('--max_index', type=int, default=256)
    parser.add_argument('--sample_num', type=int, default=256)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_checkpoint', type=bool, default=False)
    parser.add_argument('--negatives_x_device', type=bool, default=True)
    parser.add_argument('--negatives_in_device', type=bool, default=True)
    parser.add_argument('--untie_encoder', type=bool, default=True)
    parser.add_argument('--warm_start_from', type=str, default="")
    parser.add_argument('--FN_threshold', type=float, default=0.9)
    parser.add_argument('--EN_threshold', type=float, default=0.1)

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args


def main_multi(args, model, optimizer):
    epoch = 0
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        print(f'Starting training, upto {args.epoch} epochs, LR={args.learning_rate}', flush=True)

    # 加载数据集
    passage_dataset = dataset_factory.PassageDataset4QueryGeneration(args)
    passage_loader = DataLoader(passage_dataset, batch_size=args.dev_batch_size, collate_fn=passage_dataset._collate_fn, num_workers=3)
    # validate_multi_gpu(model, query_loader, passage_loader, epoch, args)
    # generate_query(model, passage_loader, args)
    train_query_generation(args, model, optimizer)

def train_query_generation(args, model, optimizer):
    epoch = 0
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        print(f'Starting training, upto {args.epoch} epochs, LR={args.learning_rate}', flush=True)


    train_dataset = dataset_factory.PassageDataset4QueryGenerationTraining(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    for epoch in range(1, args.epoch+1):
        train_sampler.set_epoch(epoch)  # shuffle batch
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset._collate_fn, sampler=train_sampler, num_workers=4, drop_last=True) 
        train_iteration_multi_gpu(model, optimizer, train_loader, epoch, args)
        torch.distributed.barrier()

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

def train_iteration_multi_gpu(model, optimizer, data_loader, epoch, args):
    total = 0
    model.train()
    total_loss = 0.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    start = time.time()
    local_start = time.time()
    all_steps_per_epoch = len(data_loader)
    step = 0
    scaler = GradScaler()
    for record in data_loader:
        record = _prepare_inputs(record)
        loss = model(input_ids=record['input_ids'], attention_mask=record['attention_mask'], labels=record['labels'])
        torch.distributed.barrier() 
        reduced_loss = reduce_tensor(loss.data)
        total_loss += reduced_loss.item()
        # optimize
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step+=1
        if step%args.report==0 and local_rank==0:
            seconds = time.time()-local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print("epoch:%d training step: %d/%d, mean loss: %.5f, current loss: %.5f,"%(epoch, step, all_steps_per_epoch, total_loss/step, loss.cpu().detach().numpy()),"report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time()-start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    if local_rank==0:
        # model.save(os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        torch.save(model.module.state_dict(), os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        seconds = time.time()-start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f'train epoch={epoch} loss={total_loss}')
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))



def generate_query(model, passage_loader, args):
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    num_return_sequences = 5
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    with open("output/generated_queries_%d.tsv"%local_rank,"w") as fout:
        with torch.no_grad():
            model.eval()
            for records, pids in tqdm(passage_loader, disable=args.local_rank>0):
                inputs = _prepare_inputs(records)
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=32,
                    do_sample=True,
                    top_k=num_return_sequences,
                    num_return_sequences=num_return_sequences
                )
                tmp = []
                for i in range(outputs.shape[0]):
                    # print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')
                    if i%num_return_sequences==0:
                        pid = pids.pop(0)
                    tmp.append(str(pid)+'\t'+tokenizer.decode(outputs[i], skip_special_tokens=True)+"\n")
                fout.writelines(tmp)
                    
                    
                    
                    

def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        elif x is None or isinstance(x, str):
            prepared[key] = x
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared

def main_cli():
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    # 加载到多卡
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        args.print_config()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = QueryGenerator(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    # optimizer = torch.optim.AdamW([params], lr=args.learning_rate, weight_decay=0.0)
    optimizer = optim.Adafactor([params], lr=args.learning_rate, weight_decay=0.0)

    if args.warm_start_from:
        print('warm start from ', args.warm_start_from)
        state_dict = torch.load(args.warm_start_from, map_location=device)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','')] = state_dict.pop(k)
        model.lm.load_state_dict(state_dict)

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print("model loaded on GPU%d"%local_rank)
    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)

    # we use the same qrels object for both training and validation sets
    main_multi(args, model, optimizer)

if __name__ == '__main__':
    main_cli()
