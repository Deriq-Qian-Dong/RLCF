import sys
import pytrec_eval
import logging
from collections import defaultdict
from typing import Type, List, Dict, Union, Tuple
def evaluate(qrels_path, results_path, ignore_identical_ids=True):
    qrels = defaultdict(dict)
    results = defaultdict(dict)
    with open(qrels_path) as f:
        lines = f.readlines()
    for line in lines:
        qid,pid,label = line.strip().split("\t")
        qrels[qid][pid] = int(label)
    with open(results_path) as f:
        lines = f.readlines()
    for line in lines:
        try:
            qid, pid, _, label = line.strip().split("\t")
        except:
            # 264014 Q0 8012101 1 1.5171098709106445 rank
            qid, _, pid, _, label, _ = line.strip().split(" ")
        results[qid][pid] = float(label)
    k_values = [1, 10, 30, 1000]
    if ignore_identical_ids:
        logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

if __name__=='__main__':
    qrels_path = sys.argv[1]
    results_path = sys.argv[2]
    ndcg, _map, recall, precision = evaluate(qrels_path, results_path)  
    print('MAP@1000:', _map['MAP@1000'])
    print('NDCG@10:', ndcg['NDCG@10'])
    print('Recall@1000:', recall['Recall@1000'])
    # print(ndcg, _map, recall, precision)


