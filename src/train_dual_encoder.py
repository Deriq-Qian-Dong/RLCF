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
from eval import evaluate
from modeling import DualEncoder, DualEncoderMeanPooling
from msmarco_eval import calc_mrr
from utils import add_prefix, build_engine, load_qid, merge, read_embed, search

def filter_non_cands(qrels_path, results_path):
    qrels_path = qrels_path.replace("test.qrels.tsv", "cands.tsv")
    qrels = defaultdict(list)
    results = defaultdict(list)
    with open(qrels_path) as f:
        lines = f.readlines()
    for line in lines:
        qid,pid,label = line.strip().split("\t")
        qrels[qid].append(pid)
    with open(results_path) as f:
        lines = f.readlines()
    for line in lines:
        try:
            qid, pid, _, label = line.strip().split("\t")
        except:
            # 264014 Q0 8012101 1 1.5171098709106445 rank
            qid, _, pid, _, label, _ = line.strip().split(" ")
        if pid in qrels[qid]:
            results[qid].append((pid, float(label)))
    with open(results_path, 'w') as fout:
        for qid in results:
            ranks = results[qid]
            ranks.sort(key=lambda x:x[1], reverse=True)
            for i in range(len(ranks)):
                idx = i+1
                pid, score = ranks[i]
                fout.write(f'{qid}\t{pid}\t{idx}\t{score}\n')
SEED = 2023
best_ndcg=-1
top_k = 1000000
early_stop = 0
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
    parser.add_argument('--negatives_x_device', type=bool, default=False)
    parser.add_argument('--negatives_in_device', type=bool, default=True)
    parser.add_argument('--untie_encoder', type=bool, default=True)
    parser.add_argument('--add_pooler', type=bool, default=False)
    parser.add_argument('--add_decoder', type=bool, default=False)
    parser.add_argument('--warm_start_from', type=str, default="")
    parser.add_argument('--p2p', type=str, default="")
    parser.add_argument('--generated_query', type=str, default="")
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
    query_dataset_names =  ['trec-covid', 'nfcorpus','nq','hotpotqa','fiqa','dbpedia-entity','scidocs','fever','scifact', 'lecard', 'cail']
    query_dataset_names = [q for q in query_dataset_names if q in args.collection]
    query_loader_mp = {}
    passage_loader_mp = {}
    all_query_paths = {}
    all_qrels = {}
    for query_dataset_name in query_dataset_names:
        print(query_dataset_name)
        query_path = os.path.join('../beir', query_dataset_name, 'test.query.tsv')
        passage_path = os.path.join('../beir', query_dataset_name, 'collection.summary.tsv')
        qrels = os.path.join('../beir', query_dataset_name, 'test.qrels.tsv')
        all_query_paths[query_dataset_name] = query_path
        all_qrels[query_dataset_name] = qrels
        query_dataset = dataset_factory.QueryDataset(args, query_path)
        passage_dataset = dataset_factory.PassageDataset(args, passage_path)
        global top_k
        top_k = len(passage_dataset)
        query_loader_mp[query_dataset_name] = DataLoader(query_dataset, batch_size=args.dev_batch_size, collate_fn=query_dataset._collate_fn, num_workers=1)
        passage_loader_mp[query_dataset_name] = DataLoader(passage_dataset, batch_size=args.dev_batch_size, collate_fn=passage_dataset._collate_fn, num_workers=1)
    validate_multi_dataset(model, query_loader_mp, passage_loader_mp, epoch, args, all_query_paths, all_qrels)

    # train_dataset = dataset_factory.DualEncoderTrainDataset(args)
    # train_dataset = dataset_factory.DualEncoderTrainReadFromHardNegDataset(args)

    # for epoch in range(1, args.epoch+1):
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     train_sampler.set_epoch(epoch)
    #     train_dataset.set_epoch(epoch)
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset._collate_fn, sampler=train_sampler, num_workers=1)
    #     loss = train_iteration_multi_gpu(model, optimizer, train_loader, epoch, args)
    #     del train_loader
    #     torch.distributed.barrier()
    #     if epoch%1==0:
    #         validate_multi_dataset(model, query_loader_mp, passage_loader_mp, epoch, args, all_query_paths, all_qrels)
    #         torch.distributed.barrier()

def validate_multi_dataset(model, query_loader_mp, passage_loader_mp, epoch, args, all_query_paths, all_qrels):
    global best_ndcg
    global early_stop
    ndcgs = []
    local_rank = torch.distributed.get_rank()
    for query_dataset_name in query_loader_mp:
        query_loader = query_loader_mp[query_dataset_name]
        passage_loader = passage_loader_mp[query_dataset_name]
        dev_query = all_query_paths[query_dataset_name]
        qrels = all_qrels[query_dataset_name]
        ndcg = validate_multi_gpu(model, query_loader, passage_loader, epoch, args, dev_query, qrels, query_dataset_name)
        ndcgs.append(ndcg)
    if local_rank==0:
        ndcg = sum(ndcgs)/len(ndcgs)
        print("average ndcg@10:", ndcg)
        if ndcg>best_ndcg:
            print("*"*50)
            print("new top")
            print("*"*50)
            best_ndcg = ndcg
            torch.save(model.state_dict(), "output/best.p")
            print("******************eval, ndcg@10: %.10f,"%(ndcg))
            early_stop = 0
        else:
            early_stop += 1
            if early_stop>3:
                assert False, 'early stop'


def validate_multi_gpu(model, query_loader, passage_loader, epoch, args, dev_query, qrels, query_dataset_name):
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    _output_file_name = 'output/_para.index.part%d'%local_rank
    output_file_name = 'output/para.index.part%d'%local_rank
    global top_k
    q_output_file_name = 'output/query.emb.step%d.npy'%epoch
    if local_rank==0:
        q_embs = []
        with torch.no_grad():
            model.eval()
            for records in query_loader:
                if args.fp16:
                    with autocast():
                        q_reps = model(query_inputs=_prepare_inputs(records))
                else:
                    q_reps = model(query_inputs=_prepare_inputs(records))
                q_embs.append(q_reps.cpu().detach().numpy())
        emb_matrix = np.concatenate(q_embs, axis=0)
        np.save(q_output_file_name, emb_matrix)
        print("predict q_embs cnt: %s" % len(emb_matrix))
    with torch.no_grad():
        model.eval()
        para_embs = []
        for records in tqdm(passage_loader, disable=args.local_rank>0):
            if args.fp16:
                with autocast():
                    p_reps = model(passage_inputs=_prepare_inputs(records))
            else:
                p_reps = model(passage_inputs=_prepare_inputs(records))
            para_embs.append(p_reps.cpu().detach().numpy())
    torch.distributed.barrier() 
    para_embs = np.concatenate(para_embs, axis=0)
    engine = torch.from_numpy(para_embs).cuda()
    np.save('output/_para.emb.part%d.npy'%local_rank, para_embs)
    qid_list = load_qid(dev_query)
    search(engine, q_output_file_name, qid_list, "output/res.top%d.part%d.step%d.%s"%(top_k, local_rank, epoch, query_dataset_name), top_k=top_k)
    torch.distributed.barrier() 
    ret = 0
    if local_rank==0:
        f_list = []
        for part in range(world_size):
            f_list.append('output/res.top%d.part%d.step%d.%s' % (top_k, part, epoch, query_dataset_name))
        shift = np.load("output/_para.emb.part0.npy").shape[0]
        merge(world_size, shift, top_k, epoch, query_dataset_name)
        filter_non_cands(qrels, 'output/res.top%d.step%d.%s'%(top_k, epoch, query_dataset_name))
        # metrics = calc_mrr(args.qrels, 'output/res.top%d.step%d'%(top_k, epoch))
        ndcg, _map, recall, precision = evaluate(qrels, 'output/res.top%d.step%d.%s'%(top_k, epoch, query_dataset_name))
        print("*"*50)
        print(query_dataset_name)
        print('MAP@1000:', _map['MAP@1000'])
        print('NDCG@10:', ndcg['NDCG@10'])
        print('NDCG@30:', ndcg['NDCG@30'])
        print('Recall@1000:', recall['Recall@1000'])
        print("*"*50)
        for run in f_list:
            os.remove(run)
        ret = ndcg['NDCG@10']
    torch.distributed.barrier() 
    return ret


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
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    args.local_rank = local_rank
    if local_rank==0:
        args.print_config()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = DualEncoder(args)
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
            state_dict[k.replace('module.','')] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print("model loaded on GPU%d"%local_rank)
    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)

    # we use the same qrels object for both training and validation sets
    # main(model, dataset, train_pairs, qrels, valid_run, qrels, args.model_out_dir)
    main_multi(args, model, optimizer)

if __name__ == '__main__':
    main_cli()
