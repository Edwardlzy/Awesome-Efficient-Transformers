# Awesome-Efficient-Transformers
This repo is a collection of awesome efficient transformers.

## Table of Contents
- Papers
- Blogs
- Tutorials

## Papers
- Attention Is All You Need (NeurIPS 2017) [[paper](https://arxiv.org/abs/1706.03762)] [[code](https://github.com/tensorflow/tensor2tensor)]
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019) [[paper](https://arxiv.org/abs/1810.04805)]
- Language Models are Few-Shot Learners [[paper](https://arxiv.org/abs/2005.14165)]

### Low Rank Methods
- Rethinking Attention with Performers (ICLR 2021) [[paper](https://arxiv.org/abs/2009.14794)]
- Linformer: Self-Attention with Linear Complexity [[paper](https://arxiv.org/abs/2006.04768)] [[code](https://github.com/tatp22/linformer-pytorch)]
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (ICML 2020) [[paper](https://arxiv.org/abs/2006.16236)] [[code](https://linear-transformers.com)]
- Synthesizer: Rethinking Self-Attention in Transformer Models (ICML 2021) [[paper](https://arxiv.org/abs/2005.00743)]
 
### Memory Compression
- Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (ICML 2019) [[paper](https://arxiv.org/abs/1810.00825)]
- Generating Wikipedia by Summarizing Long Sequences (ICLR 2018) [[paper](https://arxiv.org/abs/1801.10198)]
- Longformer: The Long-Document Transformer [[paper](https://arxiv.org/abs/2004.05150)] [[code](https://github.com/allenai/longformer)]
- ETC: Encoding Long and Structured Inputs in Transformers (EMNLP 2020) [[paper](https://arxiv.org/abs/2004.08483)]
- Big Bird: Transformers for Longer Sequences (NeurIPS 2020) [[paper](https://arxiv.org/abs/2007.14062)]
- Efficient Content-Based Sparse Attention with Routing Transformers (TACL 2020) [[paper](https://arxiv.org/abs/2003.05997)]
- Compressive Transformers for Long-Range Sequence Modelling [[paper](https://arxiv.org/abs/1911.05507)]
- Lite Transformer with Long-Short Range Attention (ICLR 2020) [[paper](https://arxiv.org/abs/2004.11886)]

### Attention Matrix Improvements
- Blockwise Self-Attention for Long Document Understanding (Workshop at EMNLP 2020) [[paper](https://arxiv.org/abs/1911.02972)]
- Generating Long Sequences with Sparse Transformers [[paper](https://arxiv.org/abs/1904.10509)]
- Axial Attention in Multidimensional Transformers [[paper](https://arxiv.org/abs/1912.12180)]
- Image Transformer (ICML 2018) [[paper](https://arxiv.org/abs/1802.05751)]
- Sparse Sinkhorn Attention [[paper](https://arxiv.org/abs/2002.11296)]

### Other Efficient Transformers / BERT
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (ICLR 2020) [[paper](https://arxiv.org/abs/1909.11942)]
- Patient Knowledge Distillation for BERT Model Compression (EMNLP 2019) [[paper](https://arxiv.org/abs/1908.09355)]
- TinyBERT: Distilling BERT for Natural Language Understanding (EMNLP 2020) [[paper](https://arxiv.org/abs/1909.10351)]
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Workshop in NeurIPS 2019) [[paper](https://arxiv.org/abs/1910.01108)]
- Contrastive Distillation on Intermediate Representations for Language Model Compression (EMNLP 2020) [[paper](https://arxiv.org/abs/2009.14167)]
- MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices (ACL 2020) [[paper](https://arxiv.org/abs/2004.02984)]
- Dynamic Knowledge Distillation for Pre-trained Language Models (EMNLP 2021) [[paper](https://arxiv.org/abs/2109.11295)]
- FastBERT: a Self-distilling BERT with Adaptive Inference Time (ACL 2020) [[paper](https://arxiv.org/abs/2004.02178)]
- XtremeDistil: Multi-stage Distillation for Massive Multilingual Models (ACL 2020) [[paper](https://arxiv.org/abs/2004.05686)]
- AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models (ACL 2021) [[paper](https://arxiv.org/abs/2107.13686)] [[code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/AutoTinyBERT)]
- DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering (ACL 2020) [[paper](https://arxiv.org/abs/2005.00697)]
- Q8BERT: Quantized 8Bit BERT (Workshop in NeurIPS 2019) [[paper](https://arxiv.org/abs/1910.06188)]
- Training with Quantization Noise for Extreme Model Compression (ICLR 2021) [[paper](https://arxiv.org/abs/2004.07320)]
- BinaryBERT: Pushing the Limit of BERT Quantization (ACL 2021) [[paper](https://arxiv.org/abs/2012.15701)]
- ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques (AAAI 2021) [[paper](https://arxiv.org/abs/2103.11367)]
- HAT: Hardware-Aware Transformers for Efficient Natural Language Processing (ACL 2020) [[paper](https://arxiv.org/abs/2005.14187)]
- Efficient Transformers: A Survey [[paper](https://arxiv.org/abs/2009.06732)]

## Blogs
- [Training a single AI model can emit as much carbon as five cars in their lifetimes](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/)
- [Model Compression via Pruning](https://towardsdatascience.com/model-compression-via-pruning-ac9b730a7c7b)
- [A developer-friendly guide to model pruning in PyTorch](https://spell.ml/blog/model-pruning-in-pytorch-X9pXQRAAACIAcH9h)
- [Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [Distilling Knowledge in Neural Networks](https://wandb.ai/authors/knowledge-distillation/reports/Distilling-Knowledge-in-Neural-Networks--VmlldzoyMjkxODk)
- [Compressing BERT for faster prediction](https://rasa.com/blog/compressing-bert-for-faster-prediction-2/)
- [BERT-related Papers](https://github.com/tomohideshibata/BERT-related-papers)
- [awesome-transformers](https://github.com/shizhediao/awesome-transformers)

## Tutorials
- [High Performance Natural Language Processing](http://gabrielilharco.com/publications/EMNLP_2020_Tutorial__High_Performance_NLP.pdf)
- [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization)
- [TensorFlow Lite example apps](https://www.tensorflow.org/lite/examples)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [TensorFlow Lite Transformers w/ Android demos](https://github.com/huggingface/tflite-android-transformers)
