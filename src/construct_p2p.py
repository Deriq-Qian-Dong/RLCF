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

import dataset_factory
import utils
from modeling import DualEncoder, DualEncoderMeanPooling
from msmarco_eval import calc_mrr
from utils import add_prefix, build_engine, load_qid, read_embed, search

SEED = 2023
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
def define_args():
    parser = argparse.ArgumentParser('BERT-retrieval model')
    parser.add_argument('--bert_config_file', type=str, default="/home/dongqian06/codes/dpr/conf/config.json")
    parser.add_argument('--vocab_file', type=str, default="/home/dongqian06/codes/dpr/conf/vocab.txt")
    parser.add_argument('--pretrain_input_file', type=str, default="/home/dongqian06/hdfs_data/data_train/pretrain/*")
    parser.add_argument('--pretrain_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--q_max_seq_len', type=int, default=160)
    parser.add_argument('--p_max_seq_len', type=int, default=160)
    parser.add_argument('--retriever_model_name_or_path', type=str, default="../data/co-condenser-marco-retriever/")
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
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--n_head_layers', type=int, default=2)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_checkpoint', type=bool, default=True)
    parser.add_argument('--negatives_x_device', type=bool, default=True)
    parser.add_argument('--negatives_in_device', type=bool, default=True)
    parser.add_argument('--untie_encoder', type=bool, default=True)
    parser.add_argument('--add_pooler', type=bool, default=False)
    parser.add_argument('--add_decoder', type=bool, default=False)
    parser.add_argument('--warm_start_from', type=str, default="")
    parser.add_argument('--p2p', type=str, default="")
    parser.add_argument('--generated_query', type=str, default="")
    parser.add_argument('--generated_psg', type=str, default="")
    parser.add_argument('--q_gen', type=bool, default=False)
    parser.add_argument('--generated_query_len', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
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
    # query_dataset = dataset_factory.QueryDataset4Filter(args)
    # query_loader = DataLoader(query_dataset, batch_size=17000, collate_fn=query_dataset._collate_fn, num_workers=3)
    query_loader = None
    passage_dataset = dataset_factory.PassageDataset(args, args.collection)
    passage_loader = DataLoader(passage_dataset, batch_size=args.dev_batch_size, collate_fn=passage_dataset._collate_fn, num_workers=3)
    validate_multi_gpu(model, query_loader, passage_loader, epoch, args)

    # train_dataset = dataset_factory.DualEncoderTrainDataset(args)
    # # train_dataset = dataset_factory.DualEncoderTrainPassageGenerationDataset(args)

    # for epoch in range(1, args.epoch+1):
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     train_sampler.set_epoch(epoch)
    #     train_dataset.set_epoch(epoch)
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset._collate_fn, sampler=train_sampler, num_workers=4)
    #     loss = train_iteration_multi_gpu(model, optimizer, train_loader, epoch, args)
    #     del train_loader
    #     torch.distributed.barrier()
    #     if epoch%1==0:
    #         validate_multi_gpu(model, query_loader, passage_loader, epoch, args)
    #         torch.distributed.barrier()

def validate_multi_gpu(model, query_loader, passage_loader, epoch, args):
    global best_mrr
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    _output_file_name = 'output/_para.index.part%d'%local_rank
    output_file_name = 'output/para.index.part%d'%local_rank
    top_k = 50
    q_output_file_name = 'output/query.emb.step%d.npy'%epoch
    with torch.no_grad():
        model.eval()
        para_embs = []
        for records in tqdm(passage_loader):
            if args.fp16:
                with autocast():
                    p_reps = model(passage_inputs=_prepare_inputs(records))
            else:
                p_reps = model(passage_inputs=_prepare_inputs(records))
            para_embs.append(p_reps.cpu().detach().numpy())
    para_embs = np.concatenate(para_embs, axis=0)
    print("predict embs cnt: %s" % len(para_embs))
    np.save('output/_para.emb.part%d.npy'%local_rank, para_embs)
    engine = torch.from_numpy(para_embs)
    engine = engine.cuda()
    print(engine.shape)
    print('create index done!')
    qid_list = load_qid(args.collection)
    torch.distributed.barrier() 
    if local_rank==0:
        tmp = []
        for i in range(world_size):
            tmp.append(np.load('output/_para.emb.part%d.npy'%i))
        tmp = np.concatenate(tmp, axis=0)
        np.save(q_output_file_name, tmp)
    torch.distributed.barrier() 
    search(engine, q_output_file_name, qid_list, "output/res.top%d.part%d.step%d"%(top_k, local_rank, epoch), top_k=top_k)
    torch.distributed.barrier() 
    if local_rank==0:
        f_list = []
        for part in range(world_size):
            f_list.append('output/res.top%d.part%d.step%d' % (top_k, part, epoch))
        shift = np.load("output/_para.emb.part0.npy").shape[0]
        merge(world_size, shift, top_k, epoch)
        top50=np.loadtxt('output/res.top50.step0')
        # top50=top50[:-(len(top50)%50)]
        top50 = top50[:,:-1]
        top50=top50.reshape((-1,50,3))
        top50 = top50.astype(int)
        total_samples = len(top50)
        np.save('top50.npy',top50)
        print("accept rate:", len(top50)/total_samples)

        
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

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

def train_iteration_multi_gpu(model, optimizer, data_loader, epoch, args):
    total = 0
    model.train()
    total_loss = 0.
    total_generation_loss = 0.
    total_ce_loss = 0.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    start = time.time()
    local_start = time.time()
    all_steps_per_epoch = len(data_loader)
    step = 0
    scaler = GradScaler()
    for record in data_loader:
        record = _prepare_inputs(record)
        with autocast():
            retriever_ce_loss = model(**record)
        loss = args.alpha*retriever_ce_loss
        torch.distributed.barrier() 
        reduced_loss = reduce_tensor(loss.data)
        total_loss += reduced_loss.item()
        total_ce_loss += float(args.alpha*retriever_ce_loss.cpu().detach().numpy())

        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step+=1
        if step%args.report==0 and local_rank==0:
            seconds = time.time()-local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print(f"epoch:{epoch} training step: {step}/{all_steps_per_epoch}, mean loss: {total_loss/step}, ce loss: {total_ce_loss/step}, ", "report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time()-start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    if local_rank==0:
        # model.save(os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        torch.save(model.state_dict(), os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        seconds = time.time()-start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f'train epoch={epoch} loss={total_loss}')
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    return total_loss


def main_cli():
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    # 加载到多卡
    import datetime
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=100000))
    local_rank = torch.distributed.get_rank()
    args.local_rank = local_rank
    if local_rank==0:
        args.print_config()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # validate_multi_gpu(None, None, None, 0, args)

    model = DualEncoderMeanPooling(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    # optimizer = torch.optim.Adam([params], lr=args.learning_rate, weight_decay=0.0)
    optimizer = optim.Lamb([params], lr=args.learning_rate, weight_decay=0.0)

    if args.warm_start_from:
        print('warm start from ', args.warm_start_from)
        state_dict = torch.load(args.warm_start_from, map_location=device)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','').replace('bert.', '')] = state_dict.pop(k)
            if 'pooler' in k or 'cls' in k:
                state_dict.pop(k.replace('module.',''))
        model.load_state_dict(state_dict)

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print("model loaded on GPU%d"%local_rank)
    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)

    # we use the same qrels object for both training and validation sets
    # main(model, dataset, train_pairs, qrels, valid_run, qrels, args.model_out_dir)
    main_multi(args, model, optimizer)


def merge(total_part, shift, top, eval_cnts):
    f_list = []
    for part in range(total_part):
        f0 = open('output/res.top%d.part%d.step%d' % (top, part, eval_cnts))
        f_list.append(f0)

    line_list = []
    for part in range(total_part):
        line = f_list[part].readline()
        line_list.append(line)

    out = open('output/res.top%d.step%d' % (top, eval_cnts), 'w')
    last_q = ''
    ans_list = {}
    while line_list[-1]:
        cur_list = []
        for line in line_list:
            sub = line.strip().split('\t')
            cur_list.append(sub)

        if last_q == '':
            last_q = cur_list[0][0]
        if cur_list[0][0] != last_q:
            rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
            for i in range(top):
                out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
            ans_list = {}
        for i, sub in enumerate(cur_list):
            ans_list[int(sub[1]) + shift*i] = float(sub[-1])
        last_q = cur_list[0][0]

        line_list = []
        for f0 in f_list:
            line = f0.readline()
            line_list.append(line)

    rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
    for i in range(top):
        out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
    out.close()

if __name__ == '__main__':
    main_cli()

