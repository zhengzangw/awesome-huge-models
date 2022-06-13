# awesome-big-models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A collection of AWESOME things about big AI models.

## Survey

- On the Opportunities and Risk of Foundation Models
- A Roadmap to Big Model

## Models

- Imagen
- DALL·E 2
- Gato
- PaLM
- OPT-175B
- YUAN 1.0
- ERNIE 3.0
- WuDao 2.0
- Switch Transformer
- 阿里达摩院M6
- 快手1.9万亿参数推荐精排模型
- PLUG
- 鹏程盘古-α
- Pangu
- BAGUALU
- CogView
- WuDao
- BriVL
- DALL·E
- CLIP
- Switch Transformer
- **GPT-3** [[OpenAI]](https://openai.com/api/) <ins>May 2020</ins>  
    Language Models are Few-Shot Learners [[NeurIPS'20]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)  

    ```yaml
    Field: NLP  
    Params: 175B  
    Training Data: ~680B Tokens (45TB)  
    Training Time: 95 A100 GPU years (355 V100 GPU years)
    Training Cost: $4.6M
    Architecture: Transformer
    Obective: Autoregressive
    ```

- Turing-NLG
- T5
- Megatron-LM
- **XLNet** [Google] <ins>June 2019</ins> [[code]](https://github.com/zihangdai/xlnet)  
    XLNet: Generalized Autoregressive Pretraining for Language Understanding [[NeurIPS'19]](https://papers.nips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)

    ```yaml
    Field: NLP
    Params: 340M
    Training Data: 33B words (113GB)
    Training Time: 1280 TPUv3 days
    Training Cost: $245k
    Architecture: Transformer
    Objective: PLM
    ```

- **RoBERTa** [[Meta AI]](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) <ins>July 2019</ins> [[code]](https://github.com/facebookresearch/fairseq)  
    RoBERTa: A Robustly Optimized BERT Pretraining Approach [[Preprint]](https://arxiv.org/abs/1907.11692)  

    ```yaml
    Field: NLP
    Params: 354M
    Training Data: (160GB)
    Training Time: 1024 V100 GPU days
    Architecture: Transformer
    Objective: MLM
    ```

- **GPT-2** [[OpenAI]](https://openai.com/blog/better-language-models/) <ins>Feb 2019</ins> [[code]](https://github.com/openai/gpt-2)  
    Language Models are Unsupervised Multitask Learners [[Preprint]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

    ```yaml
    Field: NLP  
    Params: 1.5B
    Training Data: ~8M web pages (40GB)
    Architecture: Transformer
    Objective: Autoregressive
    ```

- **BERT** [Google] <ins>Oct 2018</ins> [[code]](https://github.com/google-research/bert)  
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[NAACL'18]](https://arxiv.org/pdf/1810.04805.pdf)

    ```yaml
    Field: NLP
    Params: 330M
    Training Data: 3.3B words (16GB)
    Training Time: 64 TPUv2 days (280 V100 GPU days)
    Training Cost: ~$7k
    Architecture: Transformer
    Objective: MLM, NSP
    ```

- **GPT** [[OpenAI]](https://openai.com/blog/language-unsupervised/) <ins>June 2018</ins>  
    Improving Language Understanding by Generative Pre-Training [[Preprint]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

    ```yaml
    Field: NLP  
    Params: 117M 
    Training Data: 7k books (1GB)
    Architecture: Transformer
    Objective: Autoregressive
    ```

## Notes

Training cost:

- TPUv2 hour: $4.5
- TPUv3 hour: $8
- V100 GPU hour: $0.55 (2022)
- A100 GPU hoor: $1.10 (2022)

Related company, parameters of the largest model
