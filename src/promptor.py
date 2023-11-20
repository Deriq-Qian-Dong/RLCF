

from beir.datasets.data_loader import GenericDataLoader
import os
import random
from transformers import AutoConfig, AutoTokenizer, BertModel

class Promptor:
    def __init__(self, task: str, language: str = 'en'):
        self.task = task
        self.language = language
        self.level = 1
        if task in ['trec-covid','nfcorpus','webis-touche2020','dbpedia-entity']:
            self.level = 2
        if task in ['lecard', 'cail', 'lcsts', 'cnndaily', 'email', 'marco', 'marco-psg', 'giga']:
            self.mean_qry_len = 64
            self.mean_psg_len = 64
        else:
            corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join('/root/paddlejob/workspace/env_run/data/beir',task)).load(split="test")
            qp_pairs = []
            for qid in qrels:
                for pid in qrels[qid]:
                    if qrels[qid][pid]==self.level:
                        try:
                            qp_pairs.append((queries[qid], corpus[pid]['title']+corpus[pid]['text']))
                            break
                        except:
                            pass
            qp_pairs.sort(key=lambda x:len(x[1].split()))
            lens = [len(qp[1].split()) for qp in qp_pairs]
            self.mean_psg_len = sum(lens)/len(lens)
            lens = [len(qp[0].split()) for qp in qp_pairs]
            self.mean_qry_len = sum(lens)/len(lens)
            for idx, qp in enumerate(qp_pairs):
                length = len(qp[1].split())
                if length>=self.mean_psg_len:
                    break
            qp_pairs_ = qp_pairs[idx:idx-4]+qp_pairs[idx:idx+4]
            while len(qp_pairs_)<8:
                qp_pairs_.append(random.choice(qp_pairs))
            self.qp_pairs = qp_pairs_
        self.mp={
'WEB_SEARCH':['Please write a question based on the passage.\n', 
"""Passage: {}
Question: {}""", "Question:"],

'cnndaily':['Please write a summarization for the passage.\n', 
"""Passage: {}
TL;DR: {}""", "TL;DR:"],

'email':['Please write a summarization for the passage.\n', 
"""Passage: {}
TL;DR: {}""", "TL;DR:"],

'marco':['Please write a title for the passage.\n', 
"""Passage: {}
Title: {}""", "Title:"],

'giga':['Please write a title for the passage.\n', 
"""Passage: {}
Title: {}""", "Title:"],

'marco-psg':['Please write a summarization for the passage.\n', 
"""Passage: {}
Please write a summarization for the passage: {}""", "Please write a summarization for the passage:"],

'scifact':['Please write a scientific claim which could be supported/refuted by the passage.\n',
"""Passage: {}
Claim: {}""", "Claim:"],

'lcsts':['',
"""文章：{}
请生成一个标题：{}""", "请生成一个标题："],

'lecard':['',
"""案例：{}
请生成一个标题：{}""", "请生成一个标题："],

'cail':['',
"""案例：{}
请生成一个标题：{}""", "请生成一个标题："],

'nfcorpus':['Please write a medical question based on the passage.\n', 
"""Passage: {}
Question: {}""", "Question:"],

'fever':['Please write a claim which could be supported/refuted by the passage.\n',
"""Passage: {}
Claim: {}""", "Claim:"],

'arguana':['Please write a counter argument for the passage.\n',
"""Passage: {}
Counter Argument: {}""", "Counter Argument:"],

'trec-covid':['Please write a question about COVID-19 based on the passage.\n', 
"""Passage: {}
Question: {}""", "Question:"],

'fiqa':['Please write a financial question based on the passage.\n', 
"""Passage: {}
Question: {}""", "Question:"],

'trec-news':['Please write a topic based on the news passage.\n', 
"""Passage: {}
Topic: {}""", "Topic:"],

'dbpedia-entity':['Please write a question about the main entity in the passage.\n', 
"""Passage: {}
Question: {}""", "Question:"]}
        self.template = self.mp.get(self.task, self.mp['WEB_SEARCH'])
        self.skip_length = len(self.template[1].split("\n")[0].format(""))
        self.pad_tokens = self.template[2]
    
    def build_prompt(self, psg: str):
        if self.task in ['lecard', 'cail', 'lcsts', 'cnndaily', 'email', 'marco', 'marco-psg', 'giga']:
            template = self.mp.get(self.task)
            instruction = template[0]
            tail = template[1].format(psg, "").split("\n")[0]
            return instruction+tail+'\n'
        psg = " ".join(psg.split()[:160])
        template = self.mp.get(self.task, self.mp['WEB_SEARCH'])
        instruction = template[0]
        tail = template[1].format(psg, "").split("\n")[0]
        tmp = instruction+tail
        icl_examples = []
        for qp in self.qp_pairs:
            q,p = qp
            p = " ".join(p.split()[:160])
            sample = template[1].format(p, q)
            tmp += sample
            if len(tmp.split())<=350:
                icl_examples.append(sample)
            else:
                break
        prompt = instruction
        for sample in icl_examples:
            prompt+=sample
            prompt+='\n'
        prompt+=tail
        return prompt+'\n'
                