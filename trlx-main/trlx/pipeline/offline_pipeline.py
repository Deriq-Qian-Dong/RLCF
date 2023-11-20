from typing import Iterable, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline

max_seq_len=256
def tokenize_dialogue(dialogue: Union[str, List[str]], tokenizer, max_length=2048) -> List[int]:  # noqa: C901
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        dialogue = [tokenizer.bos_token, dialogue]
    elif isinstance(dialogue, tuple):
        dialogue = list(dialogue)
    dialogue[-1] += tokenizer.eos_token

    out = []
    ctx_length = max_length
    if tokenizer.truncation_side == "left":
        for phrase in reversed(dialogue):
            # Manually added BOS and EOS above so we don't want to add special tokens here
            tokens = tokenizer(phrase, add_special_tokens=False).input_ids[-ctx_length:]
            ctx_length -= len(tokens)
            out.insert(0, tokens)
            if ctx_length == 0:
                break

        # in case of odd number of phrases (possibly due to truncation)
        # since the first phrase always has to be a prompt, force it to be <bos>
        if len(out) % 2 == 1:
            if sum(map(len, out)) == max_length:
                out[0].pop(0)
            out.insert(0, [tokenizer.bos_token_id])

    elif tokenizer.truncation_side == "right":
        for phrase in dialogue:
            # Manually added BOS and EOS above so we don't want to add special tokens here
            tokens = tokenizer(phrase, add_special_tokens=False).input_ids[:ctx_length]
            ctx_length -= len(tokens)
            out.append(tokens)
            if ctx_length == 0:
                break
    return out


@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer: PreTrainedTokenizer):
        super().__init__()

        model_inputs = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length, add_special_tokens=True
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask} for tokens, mask in zip(prompts_tokens, attention_mask)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

@register_datapipeline
class SFTPipeline(BasePipeline):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts: List[str], labels: List[str], pids: List[int], max_prompt_length: int, tokenizer: PreTrainedTokenizer, pad_tokens: str):
        super().__init__()
        self.pad_tokens = pad_tokens
        # model_inputs = tokenizer(
        #     prompts, truncation=True, padding=True, max_length=200, add_special_tokens=True
        # )

        # prompts_tokens = model_inputs["input_ids"]
        # attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        # labels = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt", max_length=32).input_ids
        # labels[labels == self.tokenizer.pad_token_id] = -100
        self.prompts = []
        for prompt, label, pid in zip(prompts, labels, pids):
            self.prompts.append({"prompt": prompt, "label":label, "pid":pid})

    def __getitem__(self, ix: int):
        return self.prompts[ix]
    
    def _collate_fn(self, samples):
        psgs = []
        qrys = []
        pids = []
        for sample in samples:
            psgs.append(sample['prompt'])
            qrys.append(sample['label'])
            pids.append(sample['pid'])
        p_records = self.tokenizer(psgs, [f"{self.pad_tokens}"]*len(psgs), padding='longest', truncation='only_first', return_tensors="pt", max_length=max_seq_len)
        labels = self.tokenizer(qrys, padding='longest', truncation=True, return_tensors="pt", max_length=max_seq_len).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        p_records['label'] = labels
        p_records['pids'] = pids
        return p_records

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        # collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self, batch_size=batch_size, collate_fn=self._collate_fn, shuffle=shuffle)


@register_datapipeline
class RecalledPipeline(BasePipeline):
    def __init__(self, collection: List[str], recalled_list: List[List[int]], sample_num: int, max_prompt_length: int, tokenizer: PreTrainedTokenizer, pad_tokens: str,):
        super().__init__()
        self.pad_tokens = pad_tokens
        self.tokenizer = tokenizer
        self.collection = collection
        self.recalled_list = recalled_list
        self.sample_num = sample_num-1  # 采样的对比样本个数需要减1
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, ix: int):
        topk = self.recalled_list[ix]
        pid = topk[0]
        pids = topk[1:21]
        interval = len(pids)//self.sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:self.sample_num]
        data = [(self.collection[pid], pid)]
        for contrastive_pid in sample_pids:
            data.append((self.collection[contrastive_pid], contrastive_pid))
        return data

    def _collate_fn(self, samples):
        psgs = []
        pids = []
        for sample in samples:
            for psg,pid in sample:
                psgs.append(psg)
                pids.append(pid)
        p_records = self.tokenizer(psgs, [f"{self.pad_tokens}"]*len(psgs), padding='longest', truncation='only_first', return_tensors="pt", max_length=max_seq_len)
        p_records['pids'] = pids
        return p_records

    def __len__(self) -> int:
        return len(self.recalled_list)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, collate_fn=self._collate_fn, shuffle=shuffle)


def ilql_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_collate_fn,
            drop_last=drop_last,
        )
