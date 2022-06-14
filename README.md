# awesome-big-models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A collection of AWESOME things about BIG AI models.

## Survey

- [A Roadmap to Big Model](https://arxiv.org/abs/2203.14101)
- [On the Opportunities and Risk of Foundation Models](https://arxiv.org/abs/2108.07258)

## Models

- **OPT-175B** [[Meta]](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) <ins>May 2022</ins> [[code]](https://github.com/facebookresearch/metaseq)  
    OPT: Open Pre-trained Transformer Language Models [[preprint]](https://arxiv.org/abs/2205.01068)

    ```yaml
    Field: NLP
    Params: 175B
    Training Data: 800GB
    Architecutre: De
    Objective: LTR
    ```

- **CPM-2** [BAAI] <ins>June 2021</ins> [[code]](https://github.com/TsinghuaAI/CPM)  
    CPM-2: Large-scale Cost-effective Pre-trained Language Models [[Preprint]](https://arxiv.org/abs/2106.10715)

    ```yaml
    Field: NLP
    Params: 198B
    Training Data: Chinese 2.3TB, English 300GB (2.6TB)
    Architecture: En-De
    Objective: MLM
    ```

- **PLUG** [Alibaba] <ins>Apr 2021</ins>  

    ```yaml
    Field: NLP
    Params: 25B
    Training Data: Chinese (1TB)
    Architecture: En-De
    ```

- **PanGu-α** [Huawei] <ins>Apr 2021</ins>  
    PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation [[Preprint]](https://arxiv.org/abs/2104.12369)

    ```yaml
    Field: NLP
    Params: 200B
    Training Data: Chinese (1.1TB)
    Architecture: De
    Objective: LTR
    ```

- **Switch Transformer**  [Google] <ins>Jan 2021</ins>  
    Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity [[Preprint]](https://arxiv.org/abs/2101.03961)

    ```yaml
    Field: NLP
    Params: 1.6T
    Training Data: (750GB)
    Architecture: En-De
    Objective: MLM
    ```

- **CPM** [BAAI] <ins>Dec 2020</ins> [[code]](https://github.com/TsinghuaAI/CPM)  
    CPM: A Large-scale Generative Chinese Pre-trained Language Model [[Preprint]](https://arxiv.org/abs/2012.00413)

    ```yaml
    Field: NLP (Chinese)
    Params: 2.6B
    Training Data: Chinese (~100G)
    Architecture: De
    Objective: LTR
    ```

- **GPT-3** [[OpenAI]](https://openai.com/api/) <ins>May 2020</ins>  
    Language Models are Few-Shot Learners [[NeurIPS'20]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)  

    ```yaml
    Field: NLP  
    Params: 175B  
    Training Data: ~680B Tokens (45TB)  
    Training Time: 95 A100 GPU years (355 V100 GPU years)
    Training Cost: $4.6M
    Architecture: De
    Obective: LTR
    ```

- **Turing-NLG** [[Microsoft]](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) <ins>Feb 2020</ins>

    ```yaml
    Field: NLP
    Params: 17B
    Architecture: De
    Obective: LTR
    ```

- **T5** [[Google]](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) <ins>Oct 2019</ins> [[code]](https://github.com/google-research/text-to-text-transfer-transformer)  
    Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [JMLR'19](https://arxiv.org/abs/1910.10683)

    ```yaml
    Field: NLP
    Params: 11B
    Training Data: (~800 GB)
    Training Cost: $1.5M
    Architecture: En-De
    Obective: MLM
    ```

- **Megatron-LM** [Nvidia] <ins>Sept 2019</ins> [[code]](https://github.com/NVIDIA/Megatron-LM)  
    Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism [[Preprint]](https://arxiv.org/abs/1909.08053)

    ```yaml
    Field: NLP
    Params: 8.3B
    Training Data: (174 GB)
    Architecture: En-De; En
    Obective: LTR; MLM
    ```

- **XLNet** [Google] <ins>June 2019</ins> [[code]](https://github.com/zihangdai/xlnet)  
    XLNet: Generalized Autoregressive Pretraining for Language Understanding [[NeurIPS'19]](https://papers.nips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)

    ```yaml
    Field: NLP
    Params: 340M
    Training Data: 33B words (113GB)
    Training Time: 1280 TPUv3 days
    Training Cost: $245k
    Architecture: En
    Objective: PLM
    ```

- **RoBERTa** [[Meta]](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) <ins>July 2019</ins> [[code]](https://github.com/facebookresearch/fairseq)  
    RoBERTa: A Robustly Optimized BERT Pretraining Approach [[Preprint]](https://arxiv.org/abs/1907.11692)  

    ```yaml
    Field: NLP
    Params: 354M
    Training Data: (160GB)
    Training Time: 1024 V100 GPU days
    Architecture: En
    Objective: MLM
    ```

- **GPT-2** [[OpenAI]](https://openai.com/blog/better-language-models/) <ins>Feb 2019</ins> [[code]](https://github.com/openai/gpt-2)  
    Language Models are Unsupervised Multitask Learners [[Preprint]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

    ```yaml
    Field: NLP  
    Params: 1.5B
    Training Data: ~8M web pages (40GB)
    Architecture: De
    Objective: LTR
    ```

- **BERT** [Google] <ins>Oct 2018</ins> [[code]](https://github.com/google-research/bert)  
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[NAACL'18]](https://arxiv.org/pdf/1810.04805.pdf)

    ```yaml
    Field: NLP
    Params: 330M
    Training Data: 3.3B words (16GB)
    Training Time: 64 TPUv2 days (280 V100 GPU days)
    Training Cost: ~$7k
    Architecture: En
    Objective: MLM, NSP
    ```

- **GPT** [[OpenAI]](https://openai.com/blog/language-unsupervised/) <ins>June 2018</ins>  
    Improving Language Understanding by Generative Pre-Training [[Preprint]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

    ```yaml
    Field: NLP  
    Params: 117M 
    Training Data: 7k books (1GB)
    Architecture: De
    Objective: LTR
    ```

## Distributed Training Framework

> Deep Learning frameworks supportting distributed training are marked with *.

- **Pathways** [[Google]](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/) <ins>Mar 2021</ins>  
    Pathways: Asynchronous Distributed Dataflow for ML [[Preprint]](https://arxiv.org/abs/2203.12533)
- **Colossal-AI** [[HPC-AI TECH]](https://colossalai.org/) <ins>Nov 2021</ins> [[code]](https://github.com/hpcaitech/ColossalAI)  
    Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training [[Preprint]](https://arxiv.org/abs/2110.14883)
- **OneFlow*** [[OneFlow]](https://docs.oneflow.org/master/index.html) <ins>July 2020</ins> [[code]](https://github.com/OneFlow-Inc/oneflow)  
    OneFlow: Redesign the Distributed Deep Learning Framework from Scratch [[Preprint]](https://arxiv.org/abs/2110.15032)
- **MindSpore*** [[Huawei]](https://e.huawei.com/en/products/cloud-computing-dc/atlas/mindspore) <ins>Mar 2020</ins> [[code]](https://github.com/mindspore-ai/mindspore)
- **DeepSpeed** [[Microsoft]](https://www.microsoft.com/en-us/research/project/deepspeed/) <ins>Oct 2019</ins> [[code]](https://github.com/microsoft/DeepSpeed)  
    ZeRO: Memory Optimizations Toward Training Trillion Parameter Models [[SC'20]](https://arxiv.org/abs/1910.02054)
- **Megatron** [Nivida] <ins>Sept 2019</ins> [[code]](https://github.com/NVIDIA/Megatron-LM)  
    Megatron: Training Multi-Billion Parameter Language Models Using Model Parallelism [[Preprint]](https://arxiv.org/abs/1909.08053)
- **PaddlePaddle** [[Baidu]](https://www.paddlepaddle.org.cn/) <ins>Nov 2018</ins> [[code]](https://github.com/PaddlePaddle/Paddle)  
    End-to-end Adaptive Distributed Training on PaddlePaddle [[Preprint]](https://arxiv.org/abs/2112.02752)
- **Horovod** [[Uber]](https://horovod.ai/) <ins>Feb 2018</ins> [[code]](https://github.com/horovod/horovod)  
    Horovod: fast and easy distributed deep learning in TensorFlow [[Preprint]](https://arxiv.org/abs/1802.05799)
- **PyTorch*** [[Meta]](https://pytorch.org/) <ins>Sept 2016</ins> [[code]](https://github.com/pytorch/pytorch)  
    PyTorch: An Imperative Style, High-Performance Deep Learning Library [[NeurIPS'19]](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
- **Tensorflow*** [[Google]](https://www.tensorflow.org/) <ins>Nov 2015</ins> [[code]](https://github.com/tensorflow/tensorflow)  
    TensorFlow: A system for large-scale machine learning [[OSDI'16]](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

## Others

- PaLM
- YUAN 1.0
- ERNIE 3.0
- WuDao
- BriVL
- DALL·E
- CLIP
- CogView
- Imagen
- DALL·E 2
- Gato
- WuDao 2.0
- 快手1.9万亿参数推荐精排模型
- 阿里达摩院M6
- Jurassic-1 AI21 178B
- Wenxin
- Gopher
- GLaM
- **BaGuaLu** [BAAI, Alibaba] <ins>Apr 2022</ins>  
    BaGuaLu: targeting brain scale pretrained models with over 37 million cores [PPoPP'22](https://keg.cs.tsinghua.edu.cn/jietang/publications/PPOPP22-Ma%20et%20al.-BaGuaLu%20Targeting%20Brain%20Scale%20Pretrained%20Models%20w.pdf)

    ```yaml
    Field:
    Params: 174T
    ```

- EVA [BAAI] <ins>Aug 2021</ins> [[code]](https://github.com/BAAI-WuDao/EVA)  
    EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training [[Preprint]](https://arxiv.org/abs/2108.01547)

## Keys Explanations

- Company tags: the related company name. Other institudes may also involve in the job.
- Params: number of parameters of the largest model
- Training cost:
  - TPUv2 hour: $4.5
  - TPUv3 hour: $8
  - V100 GPU hour: $0.55 (2022)
  - A100 GPU hoor: $1.10 (2022)
- Architecture:
  - If not mentioned, models are built with transformers.
  - En: Encoder-based Language Model
  - De: Decoder-based Language Model
  - En-De=Encoder-Decoder-based Language Model
  - MoE: Mixture of Experts
- Objective
  - MLM: Masked Language Modeling
  - LTR: Left-To-Right Language Modeling
  - NSP: Next Sentence Prediction
  - PLM: Permuted Language Modeling
