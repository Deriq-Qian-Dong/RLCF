# RLCF
<!-- 请写一个处理数据，训练模型，评价结果的readme，用英文-->
## Introduction
This is a PyTorch implementation of the paper Unsupervised Alignment of Large Language Model for Information Retrieval via Contrastive Feedback.
Information Retrieval (IR), the process of finding information to satisfy user’s information needs, plays an essential role in modern people’s life. Recently, large language models (LLMs) have demonstrated remarkable capabilities across various tasks, some of which are important for IR. Nonetheless, LLMs frequently confront the issue of generating responses that lack specificity. This has limited the overall effectiveness of LLMs for IR in many cases. To address this issue, we present an unsupervised alignment framework called Reinforcement Learning from Contrastive Feedback (RLCF), which empowers LLMs to generate both high-quality and context-specific responses that suit the needs of IR tasks. Specifically, we first propose to construct groups of similar documents for computing unsupervised contrastive feedback signals. Next, we propose to leverage Batched-RR as the reward function to optimize LLMs to generate responses that captures the fine-grained information that distinguish documents from their similar ones. To demonstrate the effectiveness of RLCF, we conducted experiments in two typical applications of LLMs in IR, i.e., data augmentation and summarization. The experimental results show that RLCF can effectively improve the performance of LLMs in IR context.

## Datasets
- Gigaword
- LCSTS
- MS MARCO
- BEIR

## Preparation
### Retrieving similar documents for contrastive feedback
```
sh scripts/construct_beir_p2p.sh
```
### Generating summaries or queries for documents
```
sh scripts/generate_beir.sh
```
### Training the LLMs
```
sh scripts/ppo_trlx_beir.sh
```
### Evaluation
Firstly, you need to regenerate the summaries or queries for documents
```
sh scripts/generate_beir.sh
```
then, you can evaluate the performance of the LLMs with two tasks, i.e., data augmentation and summarization.
For data augmentation, you can run the following command
```
sh scripts/round_trip_beir.sh
```
For summarization, you can run the following command to conduct human evaluation
```
python src/data_evaluation/gui.py
```
