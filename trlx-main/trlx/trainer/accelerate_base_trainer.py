import json
import os
import sys
from abc import abstractmethod
from time import time
from typing import Dict, List, Optional, Tuple

import ray
import torch
from accelerate import Accelerator  # type: ignore
from ray.air import session
from ray.air.checkpoint import Checkpoint
from rich.console import Console
from rich.table import Table, Text
import pandas as pd
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    significant,
)
from trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    get_delta_model_class,
    parse_delta_kwargs,
)
from nltk.translate.bleu_score import sentence_bleu
import jieba
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

def tokenize_and_lemmatize(text):
    # 分词
    try:
        tokens = word_tokenize(text)
    except:
        tokens = ['pad']
    
    # 初始化词形归并器
    lemmatizer = WordNetLemmatizer()
    
    # 过滤标点符号
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # 加载停用词列表
    stop_words = set(stopwords.words('english'))
    
    # 提取词根并过滤停用词
    lemmas = [lemmatizer.lemmatize(token).lower() for token in tokens if token.lower() not in stop_words]
    
    return lemmas

def calculate_difference(paragraphs):
    # 创建一个字典，用于存储每个段落与其他段落的差集
    difference_dict = {}

    # 遍历每个段落
    for i, paragraph in enumerate(paragraphs):
        # 将当前段落拆分为单个字或词
        # current_paragraph = set(paragraph)
        current_paragraph = set(tokenize_and_lemmatize(paragraph))


        # 遍历其他段落
        for j, other_paragraph in enumerate(paragraphs):
            if i != j:  # 排除当前段落
                # 将其他段落拆分为单个字或词
                # other_words = set(other_paragraph)
                other_words = set(tokenize_and_lemmatize(other_paragraph))

                # 计算当前段落与其他段落的差集
                current_paragraph = current_paragraph.difference(other_words)

        # 将差集存储到字典中
        difference_dict[i] = (paragraph, current_paragraph)

    return difference_dict

def calculate_rouge1(reference, hypothesis):
    reference_words = list(reference)
    hypothesis_words = list(hypothesis)
    # print("summarys:", reference)
    # print("diff:", hypothesis)

    # 计算匹配的单词数
    matching_words = set(reference_words) & set(hypothesis_words)
    num_matching_words = len(matching_words)

    # 计算参考摘要中的单词数
    num_diff_words = len(hypothesis_words)

    # 计算 Rouge-1 分数
    rouge1_score = num_matching_words / num_diff_words if num_diff_words > 0 else 0

    return rouge1_score

def compute_batch_rouge1(texts, summarys):
    diffs = calculate_difference(texts)
    rouges = []
    for i in diffs:
        text, diff = diffs[i]
        # print(diff)
        # print(summarys[i])
        rouge1_score = calculate_rouge1(tokenize_and_lemmatize(summarys[i]), diff)
        rouges.append(rouge1_score)
    return rouges

logger = logging.get_logger(__name__)

def decode_from_markup(x):
    ret = []
    for y in x.cells:
        try:
            ret.append(Text.from_markup(y).plain)
        except:
            ret.append('Padded tokens') 
    return ret 

def table_to_df(rich_table: Table) -> pd.DataFrame:
    """Convert a rich.Table obj into a pandas.DataFrame obj with any rich formatting removed from the values.
    Args:
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
    Returns:
        DataFrame: A pandas DataFrame with the Table data as its values."""

    table_data = {
        x.header: decode_from_markup(x) for x in rich_table.columns
    }
    return pd.DataFrame(table_data)

def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = False,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to False.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    # for column in pandas_dataframe.columns:
    #     rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table

@register_trainer
class AccelerateRLTrainer(BaseRLTrainer):
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, config, **kwargs):  # noqa: C901
        super().__init__(config, **kwargs)
        self.max_length = config.train.seq_length
        self.sample_num = config.train.sample_num
        self.accelerator = Accelerator(log_with=config.train.tracker, project_dir=config.train.logging_dir)
        self.num_return_sequences = config.method.gen_kwargs['num_return_sequences']
        os.makedirs("output/trlx/",exist_ok=True)

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["auto_cast"] = False

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.model = self.setup_model()
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info("Initialized in Rank %d"%self.local_rank)

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, trust_remote_code=True, padding_side=config.tokenizer.padding_side)
        # self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, trust_remote_code=True, padding_side=config.tokenizer.padding_side, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        self.tokenizer.sep_token = "<sep>"
        if config.model.model_arch_type != "seq2seq":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(config.model.model_path, str):
            model_name = str(config.model.model_path).split()[0]
        else:
            model_name = config.model.model_path.split("/")[-1]

        if self.accelerator.num_processes == 1:
            num_gpus = "1gpu"
        else:
            num_gpus = f"{self.accelerator.num_processes}gpus"
        branch = get_git_tag()[0]

        run_name = "/".join([script_name, model_name, num_gpus]) + f":{branch}"

        if self.accelerator.is_main_process and not ray.is_initialized():
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}

            if config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
            elif config.train.tracker == "tensorboard":
                # flatten config for tensorboard, split list in hparams into flatten config
                config_dict_flat = flatten_dict(config_dict)
                config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat["optimizer/kwargs/betas"][0]
                config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat["optimizer/kwargs/betas"][1]
                config_dict_flat.pop("optimizer/kwargs/betas", None)
                gen_kwargs_suppress_tokens = config_dict_flat.pop("method/gen_kwargs/suppress_tokens", None)
                gen_experience_kwargs_suppress_tokens = config_dict_flat.pop("method/gen_experience_kwargs/suppress_tokens", None)    
                modified_modules = config_dict_flat.pop("model/delta_kwargs/modified_modules", None)
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict_flat,
                )
                config_dict_flat["method/gen_kwargs/suppress_tokens"] = gen_kwargs_suppress_tokens
                config_dict_flat["method/gen_experience_kwargs/suppress_tokens"] = gen_experience_kwargs_suppress_tokens
                config_dict_flat['model/delta_kwargs/modified_modules'] = modified_modules
            elif config.train.tracker is None:
                self.accelerator.init_trackers(project_name=self.config.train.project_name)
            else:
                raise ValueError(
                    f"Only supported trackers are `wandb` and `tensorboard`. Got: `{config.train.tracker}`. "
                    "Set `tracker` to `None` to disable tracking."
                )

    def setup_model(self):
        """
        Returns a model derived from an instance's TRLConfig
        """
        logger.info(f"Initializing model: {self.config.model.model_path}")

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        if self.config.model.model_arch_type == "seq2seq":
            freeze_bottom_seq2seq_layers(model.base_model, self.config.model.num_layers_unfrozen)
        else:
            freeze_bottom_causal_layers(model.base_model, self.config.model.num_layers_unfrozen)
        # Set the delta tuning strategies
        if self.config.model.delta_kwargs is not None:
            delta_type, delta_kwargs = parse_delta_kwargs(
                model.base_model.config,
                self.config.model.delta_kwargs,
                self.config.model.num_layers_unfrozen,
            )
            delta_model_class = get_delta_model_class(delta_type)
            delta_model = delta_model_class(model.base_model, **delta_kwargs)
            delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
            if self.config.model.checkpoint_dir:
                logger.info(f"Loading LORA model from {self.config.model.checkpoint_dir}/pytorch_model.bin!")
                delta_model._load_state_dict_into_backbone(model.base_model, torch.load(f'{self.config.model.checkpoint_dir}/pytorch_model.bin'))
            if self.accelerator.is_main_process:
                delta_model.log()
        return model

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )

        if "bitsandbytes" in optimizer.__class__.__module__:
            # Force 32-bit `nn.Embedding` weights for stability. See discussion:
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1016017746
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(module, "weight", {"optim_bits": 32})

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)
        return scheduler

    def decode(
        self,
        prompts: List[torch.LongTensor],
        samples: List[torch.LongTensor],
        prompt_sizes: torch.LongTensor = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            if self.config.model.model_arch_type == "seq2seq":
                output_start_ix = 0
            else:
                output_start_ix = prompt_size

            str_prompt = self.tokenizer.decode(prompt[:prompt_size], skip_special_tokens=True)
            str_output = self.tokenizer.decode(sample[output_start_ix:], skip_special_tokens=True)

            # Trim outputs up to `self.stop_sequences` if any are present
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()
            str_output = str_output.split("###")[0].strip()
            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            if self.config.model.model_arch_type == "seq2seq":
                sample = str_prompt + self.tokenizer.sep_token + str_output
            else:
                sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)
        kwargs.pop("token_type_ids", None)
        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)
        # logger.info("input_ids: %s"%(str(input_ids.shape)))
        kwargs.pop("token_type_ids", None)
        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir)

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """Save the underlying Hugging Face model, tokenizer, and configuration files to a directory for
        later use.

        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = f"{self.config.train.checkpoint_dir}/hf_model"
        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).save_pretrained(directory, **kwargs)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)

    def load(self, directory=None):
        """Load checkpoint of optimizer, scheduler and a model"""
        self.accelerator.load_state(directory or self.config.train.checkpoint_dir)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating model")
        # assert self.num_return_sequences==1

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []
        # instruction = self.config.train.instruction
        skip_length = self.config.train.skip_length

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_outputs = []
            all_prompt_sizes = []
            all_rewards = []
            all_metrics = []
            all_pids = []
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                prompts.pop("label")
                pids = prompts.pop("pids")
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(**prompts, **{gen_sweep_arg: gen_sweep_value})
                else:
                    samples = self.generate_eval(**prompts)

                # TODO(reciprocated): this should be moved into `decode`
                # but that needs to be synced with indexing in `make_experience`
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()
                # if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                #     torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                pids = torch.tensor(pids)
                # compute reward for a test batch
                str_samples, str_prompts, str_outputs = self.decode(prompts.input_ids, samples, prompt_sizes)

                if self.reward_fn:
                    rewards = torch.tensor(
                            self.reward_fn(
                                samples=str_samples,
                                prompts=str_prompts,
                                outputs=str_outputs,
                            ),
                            dtype=float,
                        )
                    # rewards = self.accelerator.gather_for_metrics(rewards.to(samples.device))
                    all_rewards.extend(rewards.tolist())
                if self.metric_fn:
                    metrics = torch.tensor(
                            self.metric_fn(
                                samples=str_samples,
                                prompts=str_prompts,
                                outputs=str_outputs,
                            ),
                            dtype=float,
                        )
                    # metrics = self.accelerator.gather_for_metrics(metrics.to(samples.device))
                    all_metrics.extend(metrics.tolist())
                # prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                #     self.accelerator.pad_across_processes(
                #         [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                #         dim=1,
                #         pad_index=self.tokenizer.pad_token_id,
                #     )
                # )
                # pids = self.accelerator.gather_for_metrics(pids.to(samples.device))
                # all_samples.extend(str_samples)
                all_prompts.extend(str_prompts)
                all_outputs.extend(str_outputs)
                # all_prompt_sizes.extend(prompt_sizes.tolist())
                all_pids.extend(pids.tolist())


                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            # if self.accelerator.is_main_process:
            # str_samples, str_prompts, str_outputs = self.decode(all_prompts, all_samples, all_prompt_sizes)

            all_prompts = [prompts.split("Passage:")[-1] for prompts in all_prompts]
            columns = ["prompt", "output", "pids"]
            columns_data = [all_prompts, all_outputs, all_pids]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                # logger.info("Computing rewards")
                # rewards = torch.tensor(
                #     self.reward_fn(
                #         samples=str_samples,
                #         prompts=str_prompts,
                #         outputs=str_outputs,
                #     ),
                #     dtype=float,
                # )
                all_rewards = torch.tensor(all_rewards)
                mean_reward = all_rewards.mean().item()
                columns.append("reward")
                # if not isinstance(rewards, list):
                #     rewards = rewards.tolist()
                columns_data.append(all_rewards.tolist())
                # stats[f"reward/mean{sweep_suffix}"] = mean_reward

            # additionally log any other metrics
            if self.metric_fn:
                # logger.info("Computing metrics")
                metric_time = time()
                # metrics = self.metric_fn(
                    # samples=str_samples,
                    # prompts=str_prompts,
                    # outputs=str_outputs,
                # )
                all_metrics = torch.tensor(all_metrics)
                mean_metric = all_metrics.mean().item()
                columns.append("metric")
                columns_data.append(all_metrics.tolist())
                # stats[f"metric/mean{sweep_suffix}"] = mean_metric

            # # # Prepend the sweep argument along with samples
            # if self.generate_sweep_kwarg:
            #     columns.insert(0, gen_sweep_arg)
            #     columns_data.insert(0, [gen_sweep_value] * len(samples))

            table.append(list(zip(*columns_data)))
            columns.append("rouge")

        # Log and display evaluation metrics
        rows = sum(list(map(list, zip(*table))), [])

        # Add metrics/rewards to the table's title
        table_title = f"Evaluation #{self.nth_evaluation}"
        rich_table = Table(*columns, title=table_title, show_lines=True)
        for ix in range(len(rows)):
            rich_table.add_row(*[str(significant(x, ndigits=10)) for x in rows[ix]])
        df = table_to_df(rich_table)
        df.to_csv(f'output/trlx/Psg_GeneratedQry_Reward-{self.nth_evaluation}-{self.local_rank}.tsv', sep='\t', index=False)

        if not ray.is_initialized():
            if self.config.train.tracker == "wandb":
                import wandb

                stats["samples"] = wandb.Table(columns, rows)
        
        logger.info("Summarizing evaluation")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            tmp = []
            for i in range(int(os.environ.get("WORLD_SIZE", 1))):
                tmp.append(pd.read_csv(f'output/trlx/Psg_GeneratedQry_Reward-{self.nth_evaluation}-{i}.tsv', sep='\t'))
                os.remove(f'output/trlx/Psg_GeneratedQry_Reward-{self.nth_evaluation}-{i}.tsv')
            df = pd.concat(tmp)
            prompts = df['prompt']
            outputs = df['output']
            rouges = []
            sample_num = self.sample_num
            for i in range(len(df)//sample_num):
                p = prompts[i*sample_num:(i+1)*sample_num]
                o = outputs[i*sample_num:(i+1)*sample_num]
                p = list(p)
                o = list(o)
                rouges += compute_batch_rouge1(p,o)
            df['rouge'] = rouges
            df.to_csv(f'output/trlx/Psg_GeneratedQry_Reward-{self.nth_evaluation}.tsv', sep='\t', index=False)
            table_title+=f"reward: {df.reward.mean()} rouge: {df.rouge.mean()}"
            rich_table = Table(*columns, title=table_title, show_lines=True)
            rich_table = df_to_table(df.iloc[:4], rich_table)
            stats[f"reward/mean{sweep_suffix}"] = df.reward.mean()
            Console().print(rich_table)

        self.nth_evaluation += 1
        return stats


    def inference4qgen(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Inference for generation")
        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        gen_sweep_values = [1]
        gen_sweep_arg = 1
        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        with open("output/trlx/generated_%d.tsv"%self.local_rank, "w") as fout:
            for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
                # A dedicated suffix for wandb logging
                if gen_sweep_value is not None:
                    sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
                else:
                    sweep_suffix = ""
                generate_time = time()
                for i_prompt, prompts in enumerate(self.eval_dataloader):
                    prompts.pop("label")
                    pids = prompts.pop("pids")
                    samples = self.generate_eval(**prompts, **self.config.method.gen_kwargs)

                    # TODO(reciprocated): this should be moved into `decode`
                    # but that needs to be synced with indexing in `make_experience`
                    if self.config.model.model_arch_type == "seq2seq":
                        samples = samples[:, 1:].contiguous()
                    
                    batch_size, seq_length = prompts.input_ids.shape
                    prompts.input_ids = prompts.input_ids.unsqueeze(1)
                    prompts.input_ids = prompts.input_ids.repeat(1, self.num_return_sequences, 1)
                    prompts.input_ids = prompts.input_ids.reshape(-1, seq_length)

                    prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                    pids = torch.tensor(pids)
                    pids = pids.unsqueeze(1)
                    pids = pids.repeat(1, self.num_return_sequences)
                    pids = pids.reshape(-1).tolist()

                    str_samples, str_prompts, str_outputs = self.decode(prompts.input_ids, samples, prompt_sizes)
                    tmp = []
                    for output, pid in zip(str_outputs, pids):
                        # output = output.split("###")[0].strip()
                        tmp.append(str(pid)+'\t'+output.replace("\n", ' ')+'\n')
                    fout.writelines(tmp)

                    desc = [
                        f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                        f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                    ]
                    tbar.set_description(f"[{' | '.join(desc)}]")
                    tbar.update()
                tbar.close()

                stats["time/generate"] = time() - generate_time
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            merged = []
            for i in range(int(os.environ.get("WORLD_SIZE", 1))):
                with open("output/trlx/generated_%d.tsv"%i, encoding='utf8') as fin:
                    merged+=fin.readlines()
            with open("output/generated.tsv","w", encoding='utf8') as fout:
                fout.writelines(merged)
        return stats



    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        logger.info("Starting training")

        self.generate_sweep_kwarg = None
        for k, v in self.config.method.gen_kwargs.items():
            if isinstance(v, list):
                if self.generate_sweep_kwarg is not None:
                    logger.info("Only a single sweep is allowed, {k} is going to be set to {v[0]}")
                    self.generate_kwargs[k] = v[0]
                else:
                    self.generate_sweep_kwarg = (k, v)

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, "state.json")) as f:
                        state = json.load(f)
                        self.iter_count = state["iter_count"]
        else:
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

        tbar = logging.tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float("inf")

        # For each epoch
        for epoch in range(self.config.train.epochs):
            # For each batch
            for batch in self.train_dataloader:
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    forward_time = time()
                    loss, stats = self.loss(batch)
                    forward_time = time() - forward_time
                    backward_time = time()
                    self.accelerator.backward(loss)
                    backward_time = time() - backward_time

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if self.iter_count % self.config.train.checkpoint_interval == 0:
                        self.save_pretrained("output/iter_count-%d/"%self.iter_count)

                    stats["time/forward"] = forward_time
                    stats["time/backward"] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f"learning_rate_group_{group_number}"] = lr

                    if self.iter_count % self.config.train.eval_interval == 0:
                        results = self.evaluate()
                        stats.update(results)

                        # always save checkpoint with the greatest mean reward
                        if self.config.train.save_best:
                            if stats.get("reward/mean", -float("inf")) > best_reward:
                                best_reward = stats.get("reward/mean")
                                do_save = True
                            elif stats.get("metric/mean", -float("inf")) > best_reward:
                                best_reward = stats.get("metric/mean")
                                do_save = True
                            # in case ILQL reports reward estimate as one of its metrics
                            elif stats.get("metrics/reward", -float("inf")) > best_reward:
                                best_reward = stats.get("metrics/reward")
                                do_save = True
                            else:
                                do_save = False
                            do_save = torch.tensor(do_save, device=self.accelerator.device)
                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(do_save, torch.distributed.ReduceOp.MAX)
                            if do_save:
                                best_path = f"{self.config.train.checkpoint_dir}/best_checkpoint"
                                logger.info(f"Saving the best state so far into {best_path}")
                                self.save_pretrained(best_path)
                                self.save_pretrained("output/iter_count-%d/"%self.iter_count)

                    if not ray.is_initialized():
                        self.accelerator.log(stats, step=self.iter_count)

                    desc = " | ".join(f"{k}: {v:.2f}" for k, v in stats.items() if k.startswith("loss"))
                    tbar.set_description(f"[{desc}]")
                    tbar.update()
                    if self.iter_count>=self.total_steps:
                        return 
                self.post_backward_callback()
            # results = self.evaluate()
            # stats.update(results)
            # self.save_pretrained("output/epoch-%d/"%epoch)
            self.post_epoch_callback()
        tbar.close()

    
    @abstractmethod
    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper of the decoder architecture"""
        pass

    @abstractmethod
    def loss(self, batch) -> Tuple[float, Dict]:
        """Compute loss on a batch from `store` and return some statistics"""
        pass

    @abstractmethod
    def post_backward_callback(self):
        """Do something after model update"""
        pass

    @abstractmethod
    def post_epoch_callback(self):
        """Do something after exhausting/single pass over `self.store`"""
        pass
