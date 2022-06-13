<p align="center">
<h2 align="center"> Contrastive Learning for Natural Language Processing</h2>
</p>

Current NLP models heavily rely on effective representation learning algorithms. Contrastive learning is one such technique to learn an embedding space such that similar data sample pairs have close representations while dissimilar samples stay far apart from each other. It can be used in supervised or unsupervised settings using different loss functions to produce task-specific or general-purpose representations. While it has originally enabled the success for vision tasks, recent years have seen a growing number of publications in contrastive NLP. This first line of works not only delivers promising performance improvements in various NLP tasks, but also provides desired characteristics such as task-agnostic sentence representation, faithful text generation, data-efficient learning in zero-shot and few-shot settings, interpretability and explainability.

- [Tutorial and Survey](#1-tutorial-and-survey)
- [Talk, Presentation, and Blog](#2-talk-presentation-and-blog)
- [Foundation of Contrastive Learning](#3-foundation-of-contrastive-learning)
    - [Contrastive Learning Objective](#contrastive-learning-objective)
    - [Sampling Strategy for Contrastive Learning](#sampling-strategy-for-contrastive-learning)
    - [Most Notable Applications of Contrastive Learning](#most-notable-applications-of-contrastive-learning)
    - [Analysis of Contrastive Learning](#analysis-of-contrastive-learning)
    - [Graph Contrastive Learning](#graph-contrastive-learning)
- [Contrastive Learning for NLP](#4-contrastive-learning-for-nlp)
    - [Contrastive Data Augmentation for NLP](#contrastive-data-augmentation-for-nlp)
    - [Text Classification](#text-classification)
    - [Sentence Embeddings and Phrase Embeddings](#sentence-embeddings-and-phrase-embeddings)
    - [Information Extraction](#information-extraction)
    - [Sequence Labeling](#sequence-labeling)
    - [Machine Translation](#machine-translation)
    - [Question Answering](#question-answering)
    - [Summarization](#summarization)
    - [Text Generation](#text-generation)
    - [Data-Efficient Learning](#data-efficient-learning)
    - [Contrastive Pretraining](#contrastive-pretraining)
    - [Interpretability and Explainability](#interpretability-and-explainability)
    - [Commonsense Knowledge and Reasoning](#commonsense-knowledge-and-reasoning)
    - [Vision-and-Language](#vision-and-language)
    - [Others](#others)

## 1. Tutorial and Survey

* **Contrastive Data and Learning for Natural Language Processing** *Rui Zhang, Yangfeng Ji, Yue Zhang, Rebecca J. Passonneau* `NAACL 2022 Tutorial` [[website]](https://contrastive-nlp-tutorial.github.io/)

* **A Primer on Contrastive Pretraining in Language Processing: Methods, Lessons Learned and Perspectives** *Nils Rethmeier, Isabelle Augenstein* [[pdf]](https://arxiv.org/abs/2102.12982)

* **A Survey on Contrastive Self-Supervised Learning** *Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh, Debapriya Banerjee, Fillia Makedon* [[pdf]](https://www.mdpi.com/2227-7080/9/1/2/htm)

## 2. Talk, Presentation, and Blog

* **Contrastive Representation Learning in Text** *Danqi Chen* [[slide]](https://cds.nyu.edu/wp-content/uploads/2021/11/TaD-Slides-Danqi-Chen-compressed.pdf)

* **Contrastive pairs are better than independent samples, for both learning and evaluation** *Matt Gardner* [[video]](https://drive.google.com/file/d/1DWMDeUzy9m0Z5a1gzQm4I78ZEQp8gyhm/view)

* **Contrastive Representation Learning** *Lilian Weng* [[blog]](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

* **Contrastive Self-Supervised Learning** *Ankesh Anand* [[blog]](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)

* **Triplet Loss and Online Triplet Mining in TensorFlow** *Olivier Moindrot* [[blog]](https://omoindrot.github.io/triplet-loss)

* **Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names** *Raúl Gómez* [[blog]](https://gombru.github.io/2019/04/03/ranking_loss/)

* **Contrastive Learning in 3 Minutes** *Ta-Ying Cheng* [[blog]](https://towardsdatascience.com/contrastive-learning-in-3-minutes-89d9a7db5a28)

* **Demystifying Noise Contrastive Estimation** *Jack Morris* [[blog]](https://jxmo.io/posts/nce)

* **Phrase Retrieval and Beyond** *Jinhyuk Lee* [[blog]](https://princeton-nlp.github.io/phrase-retrieval-and-beyond/)

* **Advances in Understanding, Improving, and Applying Contrastive Learning** *Dan Fu* [[blog]](https://hazyresearch.stanford.edu/blog/2022-04-19-contrastive-1)

* **Improving Transfer and Robustness in Supervised Contrastive Learning** *Mayee Chen* [[blog]](https://hazyresearch.stanford.edu/blog/2022-04-19-contrastive-2)

* **TABi: Type-Aware Bi-Encoders for Open-Domain Entity Retrieval** *Megan Leszczynski* [[blog]](https://hazyresearch.stanford.edu/blog/2022-04-19-contrastive-3)



## 3. Foundation of Contrastive Learning

### Contrastive Learning Objective
1. **Learning a similarity metric discriminatively, with application to face verification** *Sumit Chopra, Raia Hadsell, Yann LeCun* `CVPR 2005` [[pdf]](https://ieeexplore.ieee.org/abstract/document/1467314)

1. **Facenet: A unified embedding for face recognition and clustering** *Florian Schroff, Dmitry Kalenichenko, and James Philbin* `CVPR 2015` [[pdf]](https://arxiv.org/abs/1503.03832)

1. **Deep metric learning via lifted structured feature embedding** *Hyun Oh Song, Yu Xiang, Stefanie Jegelka, Silvio Savarese* `CVPR 2016` [[pdf]](https://arxiv.org/abs/1511.06452)

1. **Improved deep metric learning with multi-class n-pair loss objective** *Kihyuk Sohn* `NeurIPS 2016` [[pdf]](https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf)

1. **Noise-contrastive estimation: A new estimation principle for unnormalized statistical models** *Michael Gutmann and Aapo Hyvärinen* `AISTATS 2010` [[pdf]](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

1. **Representation learning with contrastive predictive coding** *Aaron van den Oord, Yazhe Li, Oriol Vinyals* `arXiv` [[pdf]](https://arxiv.org/abs/1807.03748)

1. **Learning a nonlinear embedding by preserving class neighbourhood structure** *Ruslan Salakhutdinov, Geoff Hinton* `AISTATS 2007` [[pdf]](http://proceedings.mlr.press/v2/salakhutdinov07a/salakhutdinov07a.pdf)

1. **Analyzing and improving representations with the soft nearest neighbor loss** *Nicholas Frosst, Nicolas Papernot, Geoffrey Hinton* `ICML 2019` [[pdf]](http://proceedings.mlr.press/v97/frosst19a/frosst19a.pdf)

### Sampling Strategy for Contrastive Learning 
1. **Learning deep representations by mutual information estimation and maximization** *R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, Yoshua Bengio* `ICLR 2019` [[pdf]](https://arxiv.org/abs/1808.06670) [[code]](https://github.com/rdevon/DIM)

1. **Debiased Contrastive Learning** *Ching-Yao Chuang, Joshua Robinson, Lin Yen-Chen, Antonio Torralba, Stefanie Jegelka* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2007.00224)

1. **Contrastive Learning with Hard Negative Samples** *Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, Stefanie Jegelka* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2010.04592)

1. **Supervised Contrastive Learning** *Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2004.11362)

1. **Adversarial Self-Supervised Contrastive Learning** *Minseon Kim, Jihoon Tack, Sung Ju Hwang* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2006.07589) [[code]](https://github.com/Kim-Minseon/RoCL)

1. **Decoupled Contrastive Learning** *Chun-Hsiao Yeh, Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen, Yann LeCun* `arXiv` [[pdf]](https://arxiv.org/abs/2110.06848) [[code]](https://github.com/Kim-Minseon/RoCL)

1. **Momentum Contrast for Unsupervised Visual Representation Learning** *Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick* `CVPR 2020` [[pdf]](https://arxiv.org/abs/1911.05722) [[code]](https://github.com/facebookresearch/moco)

1. **Unsupervised Learning of Visual Features by Contrasting Cluster Assignments** *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2006.09882) [[code]](https://github.com/facebookresearch/swav)

1. **Contrastive Multiview Coding** *Yonglong Tian, Dilip Krishnan, Phillip Isola* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1906.05849) [[code]](http://github.com/HobbitLong/CMC/)

1. **Prototypical Contrastive Learning of Unsupervised Representations** *Junnan Li, Pan Zhou, Caiming Xiong, Steven C.H. Hoi* `ICLR 2021` [[pdf]](https://arxiv.org/abs/1906.05849) [[code]](https://github.com/salesforce/PCL)

### Most Notable Applications of Contrastive Learning 
1. **Efficient Estimation of Word Representations in Vector Space** *Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean* `arXiv` [[pdf]](https://arxiv.org/abs/1301.3781)

1. **A Simple Framework for Contrastive Learning of Visual Representations** *Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton* `ICML 2020` [[pdf]](https://arxiv.org/abs/2002.05709) [[code]](https://github.com/google-research/simclr)

1. **Learning Transferable Visual Models From Natural Language Supervision** *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever* `arXiv` [[pdf]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP)

### Analysis of Contrastive Learning

1. **A Theoretical Analysis of Contrastive Unsupervised Representation Learning** *Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, Nikunj Saunshi* `ICML 2019` [[pdf]](https://arxiv.org/abs/1902.09229)

1. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere** *Tongzhou Wang, Phillip Isola* `ICML 2020` [[pdf]](https://arxiv.org/abs/2005.10242) [[code]](https://github.com/SsnL/align_uniform)

1. **What Makes for Good Views for Contrastive Learning?** *Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, Phillip Isola* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2005.10243) [[code]](https://hobbitlong.github.io/InfoMin/)

1. **Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases** *Senthil Purushwalkam, Abhinav Gupta* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2007.13916)

1. **What Should Not Be Contrastive in Contrastive Learning** *Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2008.05659)

1. **Dissecting Supervised Contrastive Learning** *Florian Graf, Christoph D. Hofer, Marc Niethammer, Roland Kwitt* `ICML 2021` [[pdf]](https://arxiv.org/abs/2102.08817) [[code]](https://github.com/plus-rkwitt/py_supcon_vs_ce)

1. **A Broad Study on the Transferability of Visual Representations with Contrastive Learning** *Ashraful Islam, Chun-Fu Chen, Rameswar Panda, Leonid Karlinsky, Richard Radke, Rogerio Feris* `ICCV 2021` [[pdf]](https://arxiv.org/abs/2103.13517)

1. **Poisoning and Backdooring Contrastive Learning** *Nicholas Carlini, Andreas Terzis* `ICLR 2022` [[pdf]](https://arxiv.org/abs/2106.09667)

1. **Understanding Dimensional Collapse in Contrastive Self-supervised Learning** *Li Jing, Pascal Vincent, Yann LeCun, Yuandong Tian* `ICLR 2022` [[pdf]](https://openreview.net/forum?id=YevsQ05DEN7)

1. **Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss** *Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2106.04156)

1. **Beyond Separability: Analyzing the Linear Transferability of Contrastive Representations to Related Subpopulations** *Jeff Z. HaoChen, Colin Wei, Ananya Kumar, Tengyu Ma* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2204.02683)

1. **Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation** *Kendrick Shen, Robbie Jones, Ananya Kumar, Sang Michael Xie, Jeff Z. HaoChen, Tengyu Ma, Percy Liang* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2204.00570)

1. **Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning** *Mayee F. Chen, Daniel Y. Fu, Avanika Narayan, Michael Zhang, Zhao Song, Kayvon Fatahalian, Christopher Ré* `arXiv` [[pdf]](https://arxiv.org/abs/2204.07596)

1. **Intriguing Properties of Contrastive Losses** *Ting Chen, Calvin Luo, Lala Li* `NeurIPS 2021` [[pdf]](https://proceedings.neurips.cc/paper/2021/file/628f16b29939d1b060af49f66ae0f7f8-Paper.pdf) [[code]](https://contrastive-learning.github.io/intriguing/)

1. **Rethinking InfoNCE: How Many Negative Samples Do You Need?** *Chuhan Wu, Fangzhao Wu, Yongfeng Huang* `arXiv` [[pdf]](https://arxiv.org/abs/2105.13003)

### Graph Contrastive Learning

1. **Graph Contrastive Learning with Augmentations** *Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2010.13902)[[code]](https://github.com/Shen-Lab/GraphCL)

1. **Contrastive Multi-View Representation Learning on Graphs** *Kaveh Hassani, Amir Hosein Khasahmadi* `ICML 2020` [[pdf]](https://arxiv.org/abs/2006.05582)

1. **Deep Graph Contrastive Representation Learning** *Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, Liang Wang* `ICML Workshop on Graph Representation Learning and Beyond` [[pdf]](https://arxiv.org/abs/2006.04131)[[code]](https://github.com/CRIPAC-DIG/GRACE)

## 4. Contrastive Learning for NLP

### Contrastive Data Augmentation for NLP

1. **Learning the Difference that Makes a Difference with Counterfactually-Augmented Data** *Divyansh Kaushik, Eduard Hovy, Zachary C. Lipton* `ICLR 2020` [[pdf]](https://arxiv.org/abs/1909.12434) [[code]](https://github.com/acmi-lab/counterfactually-augmented-data)

1. **NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation** *Kaustubh D. Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Srivastava, Samson Tan, Tongshuang Wu, Jascha Sohl-Dickstein, Jinho D. Choi, Eduard Hovy, Ondrej Dusek, Sebastian Ruder, Sajant Anand, Nagender Aneja, Rabin Banjade, Lisa Barthe, Hanna Behnke, Ian Berlot-Attwell, Connor Boyle, Caroline Brun, Marco Antonio Sobrevilla Cabezudo, Samuel Cahyawijaya, Emile Chapuis, Wanxiang Che, Mukund Choudhary, Christian Clauss, Pierre Colombo, Filip Cornell, Gautier Dagan, Mayukh Das, Tanay Dixit, Thomas Dopierre, Paul-Alexis Dray, Suchitra Dubey, Tatiana Ekeinhor, Marco Di Giovanni, Rishabh Gupta, Rishabh Gupta, Louanes Hamla, Sang Han, Fabrice Harel-Canada, Antoine Honore, Ishan Jindal, Przemyslaw K. Joniak, Denis Kleyko, Venelin Kovatchev, Kalpesh Krishna, Ashutosh Kumar, Stefan Langer, Seungjae Ryan Lee, Corey James Levinson, Hualou Liang, Kaizhao Liang, Zhexiong Liu, Andrey Lukyanenko, Vukosi Marivate, Gerard de Melo, Simon Meoni, Maxime Meyer, Afnan Mir, Nafise Sadat Moosavi, Niklas Muennighoff, Timothy Sum Hon Mun, Kenton Murray, Marcin Namysl, Maria Obedkova, Priti Oli, Nivranshu Pasricha, Jan Pfister, Richard Plant, Vinay Prabhu, Vasile Pais, Libo Qin, Shahab Raji, Pawan Kumar Rajpoot, Vikas Raunak, Roy Rinberg, Nicolas Roberts, Juan Diego Rodriguez, Claude Roux, Vasconcellos P. H. S., Ananya B. Sai, Robin M. Schmidt, Thomas Scialom, Tshephisho Sefara, Saqib N. Shamsi, Xudong Shen, Haoyue Shi, Yiwen Shi, Anna Shvets, Nick Siegel, Damien Sileo, Jamie Simon, Chandan Singh, Roman Sitelew, Priyank Soni , Taylor Sorensen, William Soto, Aman Srivastava, KV Aditya Srivatsa, Tony Sun, Mukund Varma T, A Tabassum, Fiona Anting Tan, Ryan Teehan, Mo Tiwari, Marie Tolkiehn, Athena Wang, Zijian Wang, Gloria Wang, Zijie J. Wang, Fuxuan Wei, Bryan Wilie, Genta Indra Winata, Xinyi Wu, Witold Wydmański, Tianbao Xie, Usama Yaseen, M. Yee, Jing Zhang, Yue Zhang* `arXiv` [[pdf]](https://arxiv.org/abs/2112.02721) [[code]](https://github.com/GEM-benchmark/NL-Augmenter)

1. **A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation** *Dinghan Shen, Mingzhi Zheng, Yelong Shen, Yanru Qu, Weizhu Chen* `arXiv` [[pdf]](https://arxiv.org/abs/2009.13818) [[code]](https://github.com/dinghanshen/cutoff)

1. **Efficient Contrastive Learning via Novel Data Augmentation and Curriculum Learning** *Seonghyeon Ye, Jiseon Kim, Alice Oh* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.05941) [[code]](https://github.com/vano1205/EfficientCL)

1. **CoDA: Contrast-enhanced and Diversity-promoting Data Augmentation for Natural Language Understanding** *Yanru Qu, Dinghan Shen, Yelong Shen, Sandra Sajeev, Jiawei Han, Weizhu Chen* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2010.08670)

### Text Classification

1. **CERT: Contrastive Self-supervised Learning for Language Understanding** *Hongchao Fang, Sicheng Wang, Meng Zhou, Jiayuan Ding, Pengtao Xie* `arXiv` [[pdf]](https://arxiv.org/abs/2005.12766) [[code]](https://github.com/UCSD-AI4H/CERT)

1. **Self-Supervised Contrastive Learning for Efficient User Satisfaction Prediction in Conversational Agents** *Mohammad Kachuee, Hao Yuan, Young-Bum Kim, Sungjin Lee* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2010.11230)

1. **Not All Negatives are Equal: Label-Aware Contrastive Loss for Fine-grained Text Classification** *Varsha Suresh, Desmond C. Ong* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.05427)

1. **Constructing Contrastive samples via Summarization for Text Classification with limited annotations** *Yangkai Du, Tengfei Ma, Lingfei Wu, Fangli Xu, Xuhong Zhang, Bo Long, Shouling Ji* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.05094)

1. **Semantic Re-Tuning via Contrastive Tension** *Fredrik Carlsson, Amaru Cuba Gyllensten, Evangelia Gogoulou, Erik Ylipää Hellqvist, Magnus Sahlgren* `ICLR 2021` [[pdf]](https://openreview.net/forum?id=Ov_sMNau-PF) [[code]](https://github.com/FreddeFrallan/Contrastive-Tension)

1. **Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval** *Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2007.00808)

1. **Improving Gradient-based Adversarial Training for Text Classification by Contrastive Learning and Auto-Encoder** *Yao Qiu, Jinchao Zhang, Jie Zhou* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2109.06536)

1. **Contrastive Document Representation Learning with Graph Attention Networks** *Peng Xu, Xinchi Chen, Xiaofei Ma, Zhiheng Huang, Bing Xiang* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2110.10778)

1. **Attention-based Contrastive Learning for Winograd Schemas** *Tassilo Klein, Moin Nabi* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.05108) [[code]](https://github.com/SAP-samples/emnlp2021-attention-contrastive-learning/)

1. **CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding** *Dong Wang, Ning Ding, Piji Li, Hai-Tao Zheng* `ACL 2021` [[pdf]](https://arxiv.org/abs/2107.00440) [[code]](https://github.com/kandorm/CLINE)

1. **Contrastive Learning-Enhanced Nearest Neighbor Mechanism for Multi-Label Text Classification** *Xi’ao Su, Ran Wang, Xinyu Dai* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-short.75/)

1. **Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification** *Zihan Wang, Peiyi Wang, Lianzhe Huang, Xin Sun, Houfeng Wang* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.491/)

1. **Label Anchored Contrastive Learning for Language Understanding** *Zhenyu Zhang, Yuming Zhao, Meng Chen, Xiaodong He* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.10227)

1. **Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks** *Anton Chernyavskiy, Dmitry Ilvovsky, Pavel Kalinin, Preslav Nakov* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2110.15725)

### Sentence Embeddings and Phrase Embeddings

1. **Towards Universal Paraphrastic Sentence Embeddings** *John Wieting, Mohit Bansal, Kevin Gimpel, Karen Livescu* `ICLR 2016` [[pdf]](https://arxiv.org/abs/1511.08198) [[code]](https://github.com/jwieting/iclr2016)

1. **An Efficient Framework for Learning Sentence Representations** *Lajanugen Logeswaran, Honglak Lee* `ICLR 2018` [[pdf]](https://arxiv.org/abs/1803.02893) [[code]](https://github.com/lajanugen/S2V)

1. **SimCSE: Simple Contrastive Learning of Sentence Embeddings** *Tianyu Gao, Xingcheng Yao, Danqi Chen* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.08821) [[code]](https://github.com/princeton-nlp/simcse)

1. **Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders** *Fangyu Liu, Ivan Vulić, Anna Korhonen, Nigel Collier* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.08027) [[code]](https://github.com/cambridgeltl/mirror-bert)

1. **Learning Dense Representations of Phrases at Scale** *Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, Danqi Chen* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.12624) [[code]](https://github.com/princeton-nlp/DensePhrases)

1. **Phrase Retrieval Learns Passage Retrieval, Too** *Jinhyuk Lee, Alexander Wettig, Danqi Chen* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.08133) [[code]](https://github.com/princeton-nlp/DensePhrases)

1. **Self-Guided Contrastive Learning for BERT Sentence Representations** *Taeuk Kim, Kang Min Yoo, Sang-goo Lee* `	ACL 2021` [[pdf]](https://arxiv.org/abs/2106.07345)

1. **Pairwise Supervised Contrastive Learning of Sentence Representations** *Dejiao Zhang, Shang-Wen Li, Wei Xiao, Henghui Zhu, Ramesh Nallapati, Andrew O. Arnold, Bing Xiang* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.05424) [[code]](https://github.com/amazon-research/sentence-representations)

1. **SupCL-Seq: Supervised Contrastive Learning for Downstream Optimized Sequence Representations** *Hooman Sedghamiz, Shivam Raval, Enrico Santus, Tuka Alhanai, Mohammad Ghassemi* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.07424) [[code]](https://github.com/hooman650/SupCL-Seq)

1. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** *Nils Reimers, Iryna Gurevych* `EMNLP 2019` [[pdf]](https://arxiv.org/abs/1908.10084) [[code]](https://github.com/UKPLab/sentence-transformers)

1. **An Unsupervised Sentence Embedding Method by Mutual Information Maximization** *Yan Zhang, Ruidan He, Zuozhu Liu, Kwan Hui Lim, Lidong Bing* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2009.12061) [[code]](https://github.com/yanzhangnlp/IS-BERT)

1. **DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations** *John Giorgi, Osvald Nitski, Bo Wang, Gary Bader* `ACL 2021` [[pdf]](https://arxiv.org/abs/2006.03659) [[code]](https://github.com/JohnGiorgi/DeCLUTR)

1. **ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer** *Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, Weiran Xu* `ACL 2021` [[pdf]](https://arxiv.org/abs/2105.11741) [[code]](https://github.com/yym6472/ConSERT)

1. **DialogueCSE: Dialogue-based Contrastive Learning of Sentence Embeddings** *Che Liu, Rui Wang, Jinghua Liu, Jian Sun, Fei Huang, Luo Si* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.12599) [[code]](https://github.com/wangruicn/DialogueCSE)

1. **Pretraining with Contrastive Sentence Objectives Improves Discourse Performance of Language Models** *Dan Iter, Kelvin Guu, Larry Lansing, Dan Jurafsky* `ACL 2020` [[pdf]](https://arxiv.org/abs/2005.10389) [[code]](https://github.com/google-research/language/tree/master/language/conpono)

1. **Contextualized and Generalized Sentence Representations by Contrastive Self-Supervised Learning: A Case Study on Discourse Relation Analysis** *Hirokazu Kiyomaru, Sadao Kurohashi* `NAACL 2021` [[pdf]](https://aclanthology.org/2021.naacl-main.442.pdf)

1. **DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings** *Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljačić, Shang-Wen Li, Wen-tau Yih, Yoon Kim, James Glass* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2204.10298) [[code]](https://github.com/voidism/DiffCSE)

1. **Exploring the Impact of Negative Samples of Contrastive Learning: A Case Study of Sentence Embedding** *Rui Cao, Yihao Wang, Yuxin Liang, Ling Gao, Jie Zheng, Jie Ren, Zheng Wang* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.248/)

1. **Syntax-guided Contrastive Learning for Pre-trained Language Model** *Shuai Zhang, Wang Lijie, Xinyan Xiao, Hua Wu* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.191/)

1. **Virtual Augmentation Supported Contrastive Learning of Sentence Representations** *Dejiao Zhang, Wei Xiao, Henghui Zhu, Xiaofei Ma, Andrew Arnold* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.70/)

1. **A Sentence is Worth 128 Pseudo Tokens: A Semantic-Aware Contrastive Learning Framework for Sentence Embeddings** *Haochen Tan, Wei Shao, Han Wu, Ke Yang, Linqi Song* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.22/)

1. **SCD: Self-Contrastive Decorrelation of Sentence Embeddings** *Tassilo Klein, Moin Nabi* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-short.44/)

1. **A Contrastive Framework for Learning Sentence Representations from Pairwise and Triple-wise Perspective in Angular Space** *Yuhao Zhang, Hongji Zhu, Yongliang Wang, Nan Xu, Xiaobo Li, Binqiang Zhao* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.336/)

1. **Debiased Contrastive Learning of Unsupervised Sentence Representations** *Kun Zhou, Beichen Zhang, Xin Zhao, Ji-Rong Wen* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.423/)

1. **UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining** *Jiacheng Li, Jingbo Shang, Julian McAuley* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.426/)

1. **EASE: Entity-Aware Contrastive Learning of Sentence Embedding** *Sosuke Nishikawa, Ryokan Ri, Ikuya Yamada, Yoshimasa Tsuruoka, Isao Echizen* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.04260)

1. **MCSE: Multimodal Contrastive Learning of Sentence Embeddings** *Miaoran Zhang, Marius Mosbach, David Ifeoluwa Adelani, Michael A. Hedderich, Dietrich Klakow* `NAACL 2022` [[pdf]]()

### Information Extraction

1. **ERICA: Improving Entity and Relation Understanding for Pre-trained Language Models via Contrastive Learning** *Yujia Qin, Yankai Lin, Ryuichi Takanobu, Zhiyuan Liu, Peng Li, Heng Ji, Minlie Huang, Maosong Sun, Jie Zhou* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.15022) [[code]](https://github.com/thunlp/ERICA)

1. **CIL: Contrastive Instance Learning Framework for Distantly Supervised Relation Extraction** *Tao Chen, Haizhou Shi, Siliang Tang, Zhigang Chen, Fei Wu, Yueting Zhuang* `ACL 2021` [[pdf]](https://arxiv.org/abs/2106.10855)

1. **CLEVE: Contrastive Pre-training for Event Extraction** *Ziqi Wang, Xiaozhi Wang, Xu Han, Yankai Lin, Lei Hou, Zhiyuan Liu, Peng Li, Juanzi Li, Jie Zhou* `ACL 2021` [[pdf]](https://arxiv.org/abs/2105.14485) [[code]](https://github.com/THU-KEG/CLEVE)

1. **CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning** *Sarkar Snigdha Sarathi Das, Arzoo Katiyar, Rebecca J. Passonneau, Rui Zhang* `ACL 2022` [[pdf]](https://arxiv.org/abs/2109.07589) [[code]](https://github.com/psunlpgroup/CONTaiNER)

1. **TABi: Type-Aware Bi-Encoders for Open-Domain Entity Retrieval** *Megan Leszczynski, Daniel Y. Fu, Mayee F. Chen, Christopher Ré* `Findings of ACL 2022` [[pdf]](https://arxiv.org/abs/2204.08173)

1. **Cross-Lingual Contrastive Learning for Fine-Grained Entity Typing for Low-Resource Languages** *Xu Han, Yuqi Luo, Weize Chen, Zhiyuan Liu, Maosong Sun, Zhou Botong, Hao Fei, Suncong Zheng* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.159/) [[code]](https://github.com/thunlp/crosset)

1. **HiCLRE: A Hierarchical Contrastive Learning Framework for Distantly Supervised Relation Extraction** *Dongyang Li, Taolin Zhang, Nan Hu, Chengyu Wang, Xiaofeng He* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.202/)

1. **HiURE: Hierarchical Exemplar Contrastive Learning for Unsupervised Relation Extraction** *Shuliang Liu, Xuming Hu, Chenwei Zhang, Shu’ang Li, Lijie Wen, Philip S. Yu* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.02225)

1. **Label Refinement via Contrastive Learning for Distantly-Supervised Named Entity Recognition** *Huaiyuan Ying, Shengxuan Luo, Tiantian Dang, Sheng Yu* `Findings of NAACL 2022` [[pdf]]()

### Sequence Labeling
1. **Contrastive Estimation: Training Log-Linear Models on Unlabeled Data** *Noah A. Smith, Jason Eisner* `ACL 2005` [[pdf]](https://aclanthology.org/P05-1044.pdf)

### Machine Translation

1. **Contrastive Learning for Many-to-many Multilingual Neural Machine Translation** *Xiao Pan, Mingxuan Wang, Liwei Wu, Lei Li* `ACL 2021` [[pdf]](https://arxiv.org/abs/2105.09501) [[code]](https://github.com/PANXiao1994/mRASP2)

1. **Contrastive Conditioning for Assessing Disambiguation in MT: A Case Study of Distilled Bia** *Jannis Vamvas, Rico Sennrich* `EMNLP 2021` [[pdf]](https://aclanthology.org/2021.emnlp-main.803.pdf) [[code]](https://github.com/ZurichNLP/contrastive-conditioning)

1. **As Little as Possible, as Much as Necessary: Detecting Over- and Undertranslations with Contrastive Conditioning** *Jannis Vamvas, Rico Sennrich* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-short.53/)

1. **Improving Word Translation via Two-Stage Contrastive Learning** *Yaoyiran Li, Fangyu Liu, Nigel Collier, Anna Korhonen, Ivan Vulić* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.299/)

1. **When do Contrastive Word Alignments Improve Many-to-many Neural Machine Translation?** *Zhuoyuan Mao, Chenhui Chu, Raj Dabre, Haiyue Song, Zhen Wan, Sadao Kurohashi* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2204.12165)

1. **CoCoA-MT: A Dataset and Benchmark for Contrastive Controlled MT with Application to Formality** *Maria Nadejde, Anna Currey, Benjamin Hsu, Xing Niu, Georgiana Dinu, Marcello Federico* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.04022)

### Question Answering 

1. **Dense Passage Retrieval for Open-Domain Question Answering** *Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2004.04906) [[code]](https://github.com/facebookresearch/DPR)

1. **Self-supervised Contrastive Cross-Modality Representation Learning for Spoken Question Answering** *Chenyu You, Nuo Chen, Yuexian Zou* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.03381)

1. **xMoCo: Cross Momentum Contrastive Learning for Open-Domain Question Answering** *Nan Yang, Furu Wei, Binxing Jiao, Daxin Jiang, Linjun Yang* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-long.477.pdf)

1. **Contrastive Domain Adaptation for Question Answering using Limited Text Corpora** *Zhenrui Yue, Bernhard Kratzwald, Stefan Feuerriegel* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2108.13854) [[code]](https://github.com/Yueeeeeeee/CAQA)

1. **To Answer or Not To Answer? Improving Machine Reading Comprehension Model with Span-based Contrastive Learning** *Yunjie Ji, Liangyu Chen, Chenxiao Dou, Baochang Ma, Xiangang Li* `Findings of NAACL 2022` [[pdf]]()

1. **Seeing the wood for the trees: a contrastive regularization method for the low-resource Knowledge Base Question Answering** *Junping Liu, Shijie Mei, Xinrong Hu, Xun Yao, JACK Yang, Yi Guo* `Findings of NAACL 2022` [[pdf]]()

### Summarization

1. **CONFIT: Toward Faithful Dialogue Summarization with Linguistically-Informed Contrastive Fine-tuning** *Xiangru Tang, Arjun Nair, Borui Wang, Bingyao Wang, Jai Amit Desai, Aaron Wade, Haoran Li, Asli Celikyilmaz, Yashar Mehdad, Dragomir Radev* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2112.08713)

1. **CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization** *Shuyang Cao, Lu Wang* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.09209) [[code]](https://shuyangcao.github.io/projects/cliff_summ)

1. **Contrastive Attention Mechanism for Abstractive Sentence Summarization** *Xiangyu Duan, Hongfei Yu, Mingming Yin, Min Zhang, Weihua Luo, Yue Zhang* `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1301.pdf) [[code]](https://github.com/travel-go/Abstractive-Text-Summarization)

1. **SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization** *Yixin Liu, Pengfei Liu* `ACL 2021` [[pdf]](https://arxiv.org/abs/2106.01890) [[code]](https://github.com/yixinL7/SimCLS)

1. **Unsupervised Reference-Free Summary Quality Evaluation via Contrastive Learning** *Hanlu Wu, Tengfei Ma, Lingfei Wu, Tariro Manyumwa, Shouling Ji* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2010.01781) [[code]]https://github.com/whl97/LS-Score)

1. **Contrastive Aligned Joint Learning for Multilingual Summarization** *Danqing Wang, Jiaze Chen, Hao Zhou, Xipeng Qiu, Lei Li* `Findings of ACL 2021` [[pdf]](https://aclanthology.org/2021.findings-acl.242.pdf) [[code]](https://github.com/dqwang122/CALMS)

1. **Topic-Aware Contrastive Learning for Abstractive Dialogue Summarization** *Junpeng Liu, Yanyan Zou, Hainan Zhang, Hongshen Chen, Zhuoye Ding, Caixia Yuan, Xiaojie Wang* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.04994)

1. **Graph Enhanced Contrastive Learning for Radiology Findings Summarization** *Jinpeng Hu, Zhuo Li, Zhihong Chen, Zhen Li, Xiang Wan, Tsung-Hui Chang* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.320/)

### Text Generation

1. **Controllable Natural Language Generation with Contrastive Prefixes** *Jing Qian, Li Dong, Yelong Shen, Furu Wei, Weizhu Chen* `Findings of ACL 2022` [[pdf]](https://arxiv.org/abs/2202.13257) [[code]](https://github.com/yxuansu/SimCTG)

1. **A Contrastive Framework for Neural Text Generation** *Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier* `arXiv` [[pdf]](https://arxiv.org/abs/2202.06417) [[code]](https://github.com/yxuansu/SimCTG)

1. **Counter-Contrastive Learning for Language GANs** *Yekun Chai, Haidong Zhang, Qiyue Yin, Junge Zhang* `Findings of EMNLP 2021` [[pdf]](https://aclanthology.org/2021.findings-emnlp.415.pdf)

1. **Contrastive Learning with Adversarial Perturbations for Conditional Text Generation** *Seanie Lee, Dong Bok Lee, Sung Ju Hwang* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2012.07280) [[code]](https://github.com/seanie12/CLAPS)

1. **Logic-Consistency Text Generation from Semantic Parses** *Chang Shu, Yusen Zhang, Xiangyu Dong, Peng Shi, Tao Yu, Rui Zhang* `Findings of ACL 2021` [[pdf]](https://aclanthology.org/2021.findings-acl.388.pdf) [[code]](https://github.com/Ciaranshu/relogic)

1. **Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation** *Haoran Yang, Wai Lam, Piji Li* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.01484) [[code]](https://github.com/LHRYANG/CRL_EGPG)

1. **Grammatical Error Correction with Contrastive Learning in Low Error Density Domains** *Hannan Cao, Wenmian Yang, Hwee Tou Ng* `Findings of EMNLP 2021` [[pdf]](https://aclanthology.org/2021.findings-emnlp.419/) [[code]](https://github.com/nusnlp/geccl)

1. **Group-wise Contrastive Learning for Neural Dialogue Generation** *Hengyi Cai, Hongshen Chen, Yonghao Song, Zhuoye Ding, Yongjun Bao, Weipeng Yan, Xiaofang Zhao* `Findings of EMNLP 2020` [[pdf]](https://arxiv.org/abs/2009.07543) [[code]](https://github.com/hengyicai/ContrastiveLearning4Dialogue)

1. **Contrastive Attention for Automatic Chest X-ray Report Generation** *Fenglin Liu, Changchang Yin, Xian Wu, Shen Ge, Yuexian Zou, Ping Zhang, Xu Sun* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2106.06965)

1. **Weakly Supervised Contrastive Learning for Chest X-Ray Report Generation** *An Yan, Zexue He, Xing Lu, Jiang Du, Eric Chang, Amilcare Gentili, Julian McAuley, Chun-Nan Hsu* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.12242)

1. **Learning with Contrastive Examples for Data-to-Text Generation** *Yui Uehara, Tatsuya Ishigaki, Kasumi Aoki, Hiroshi Noji, Keiichi Goshima, Ichiro Kobayashi, Hiroya Takamura, Yusuke Miyao* `COLING 2020` [[pdf]](https://aclanthology.org/2020.coling-main.213.pdf) [[code]](https://github.com/aistairc/contrastive_data2text)

1. **A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration** *Shaojie Jiang, Ruqing Zhang, Svitlana Vakulenko, Maarten de Rijke* `arXiv` [[pdf]](https://arxiv.org/abs/2205.02517)[[code]](https://github.com/ShaojieJiang/CT-Loss)

1. **Keywords and Instances: A Hierarchical Contrastive Learning Framework Unifying Hybrid Granularities for Text Generation** *Mingzhe Li, XieXiong Lin, Xiuying Chen, Jinxiong Chang, Qishen Zhang, Feng Wang, Taifeng Wang, Zhongyi Liu, Wei Chu, Dongyan Zhao, Rui Yan* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.304/)

### Data-Efficient Learning

1. **An Explicit-Joint and Supervised-Contrastive Learning Framework for Few-Shot Intent Classification and Slot Filling** *Han Liu, Feng Zhang, Xiaotong Zhang, Siyang Zhao, Xianchao Zhang* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2110.13691)

1. **Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning** *Jianguo Zhang, Trung Bui, Seunghyun Yoon, Xiang Chen, Zhiwei Liu, Congying Xia, Quan Hung Tran, Walter Chang, Philip Yu* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.06349) [[code]](https://github.com/jianguoz/Few-Shot-Intent-Detection)

1. **Bridge to Target Domain by Prototypical Contrastive Learning and Label Confusion: Re-explore Zero-Shot Learning for Slot Filling** *Liwen Wang, Xuefeng Li, Jiachi Liu, Keqing He, Yuanmeng Yan, Weiran Xu* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2110.03572) [[code]](https://github.com/W-lw/PCLC)

1. **Active Learning by Acquiring Contrastive Examples** *Katerina Margatina, Giorgos Vernikos, Loïc Barrault, Nikolaos Aletras* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.03764) [[code]](https://github.com/mourga/contrastive-active-learning)

1. **Bi-Granularity Contrastive Learning for Post-Training in Few-Shot Scene** *Ruikun Luo, Guanhuan Huang, Xiaojun Quan* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2106.02327)

1. **Contrastive Learning for Prompt-based Few-shot Language Learners** *Yiren Jian, Chongyang Gao, Soroush Vosoughi* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.01308)

1. **Zero-Shot Event Detection Based on Ordered Contrastive Learning and Prompt-Based Prediction** *Senhui Zhang, Tao Ji, Wendi Ji, Xiaoling Wang* `Findings of NAACL 2022` [[pdf]]()

1. **RCL: Relation Contrastive Learning for Zero-Shot Relation Extraction** *Shusen Wang, Bosen Zhang, Yajing Xu, Yanan Wu, Bo Xiao* `Findings of NAACL 2022` [[pdf]]()

### Contrastive Pretraining

1. **COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining** *Yu Meng, Chenyan Xiong, Payal Bajaj, Saurabh Tiwary, Paul Bennett, Jiawei Han, Xia Song* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2102.08473) [[code]](https://github.com/microsoft/COCO-LM)

1. **TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning** *Yixuan Su, Fangyu Liu, Zaiqiao Meng, Tian Lan, Lei Shu, Ehsan Shareghi, Nigel Collier* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2111.04198) [[code]](https://github.com/yxuansu/TaCL)

1. **CLEAR: Contrastive Learning for Sentence Representation** *Zhuofeng Wu, Sinong Wang, Jiatao Gu, Madian Khabsa, Fei Sun, Hao Ma* `arXiv` [[pdf]](https://arxiv.org/abs/2012.15466)

1. **Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning** *Beliz Gunel, Jingfei Du, Alexis Conneau, Ves Stoyanov* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2011.01403)

1. **Pre-Training Transformers as Energy-Based Cloze Models** *Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2012.08561) [[code]](https://github.com/google-research/electra)

1. **Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach** *Yue Yu, Simiao Zuo, Haoming Jiang, Wendi Ren, Tuo Zhao, Chao Zhang* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2010.07835) [[code]](https://github.com/yueyu1030/COSINE)

1. **Data-Efficient Pretraining via Contrastive Self-Supervision** *Nils Rethmeier, Isabelle Augenstein* `arXiv` [[pdf]](https://arxiv.org/abs/2010.01061)

1. **Multi-Granularity Contrasting for Cross-Lingual Pre-Training** *Shicheng Li, Pengcheng Yang, Fuli Luo, Jun Xie* `Findings of ACL 2021` [[pdf]](https://aclanthology.org/2021.findings-acl.149.pdf)

1. **InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training** *Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, Ming Zhou* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2007.07834) [[code]](https://aka.ms/infoxlm)

### Interpretability and Explainability

1. **Evaluating Models' Local Decision Boundaries via Contrast Sets** *Matt Gardner, Yoav Artzi, Victoria Basmova, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hanna Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, Ben Zhou* `arXiv` [[pdf]](https://arxiv.org/abs/2004.02709)

1. **ALICE: Active Learning with Contrastive Natural Language Explanations** *Weixin Liang, James Zou, Zhou Yu* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2009.10259)

1. **Explaining NLP Models via Minimal Contrastive Editing (MiCE)** *Alexis Ross, Ana Marasović, Matthew E. Peters* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2012.13985) [[code]](https://github.com/allenai/mice)

1. **KACE: Generating Knowledge Aware Contrastive Explanations for Natural Language Inference** *Qianglong Chen, Feng Ji, Xiangji Zeng, Feng-Lin Li, Ji Zhang, Haiqing Chen, Yin Zhang* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-long.196.pdf)

1. **Contrastive Explanations for Model Interpretability** *Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, Yoav Goldberg* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2103.01378) [[code]](https://github.com/allenai/contrastive-explanations)

1. **Explanation Graph Generation via Pre-trained Language Models: An Empirical Study with Contrastive Learning** *Swarnadeep Saha, Prateek Yadav, Mohit Bansal* `ACL 2022` [[pdf]](https://arxiv.org/abs/2204.04813) [[code]](https://github.com/swarnaHub/ExplagraphGen)

1. **Toward Interpretable Semantic Textual Similarity via Optimal Transport-based Contrastive Sentence Learning** *Seonghyeon Lee, Dongha Lee, Seongbo Jang, Hwanjo Yu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.412/)

### Commonsense Knowledge and Reasoning

1. **Contrastive Self-Supervised Learning for Commonsense Reasoning** *Tassilo Klein, Moin Nabi* `ACL 2020` [[pdf]](https://arxiv.org/abs/2005.00669) [[code]](https://github.com/SAP-samples/acl2020-commonsense/)

1. **Prompting Contrastive Explanations for Commonsense Reasoning Tasks** *Bhargavi Paranjape, Julian Michael, Marjan Ghazvininejad, Luke Zettlemoyer, Hannaneh Hajishirzi* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2106.06823)

1. **KFCNet: Knowledge Filtering and Contrastive Learning Network for Generative Commonsense Reasoning** *Haonan Li, Yeyun Gong, Jian Jiao, Ruofei Zhang, Timothy Baldwin, Nan Duan* `Findings of EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.06704)

1. **Learning from Missing Relations: Contrastive Learning with Commonsense Knowledge Graphs for Commonsense Inference** *Yong-Ho Jung, Jun-Hyung Park, Joon-Young Choi, Mingyu Lee, Junho Kim, Kang-Min Kim, SangKeun Lee* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.119/)

### Vision-and-Language

1. **Counterfactual Contrastive Learning for Weakly-Supervised Vision-Language Grounding** *Zhu Zhang, Zhou Zhao, Zhijie Lin, Jieming Zhu, Xiuqiang He* `NeurIPS 2020` [[pdf]](https://papers.nips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf)

1. **UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning** *Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, Haifeng Wang* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.15409) [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/UNIMO)

1. **SOrT-ing VQA Models : Contrastive Gradient Learning for Improved Consistency** *Sameer Dharur, Purva Tendulkar, Dhruv Batra, Devi Parikh, Ramprasaath R. Selvaraju* `NeurIPS 2020 workshop` [[pdf]](https://arxiv.org/abs/2010.10038) [[code]](https://github.com/sameerdharur/sorting-vqa)

1. **Contrastive Learning for Weakly Supervised Phrase Grounding** *Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2006.09920) [[code]](http://tanmaygupta.info/info-ground/)

1. **Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning** *Wanyun Cui, Guangyu Zheng, Wei Wang* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2010.08200)

1. **VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding** *Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, Christoph Feichtenhofer* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.14084) [[code]](https://github.com/pytorch/fairseq/tree/main/examples/MMPT)

1. **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision** *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig* `ICML 2021` [[pdf]](https://arxiv.org/abs/2102.05918)

1. **UMIC: An Unreferenced Metric for Image Captioning via Contrastive Learning** *Hwanhee Lee, Seunghyun Yoon, Franck Dernoncourt, Trung Bui, Kyomin Jung* `ACL 2021` [[pdf]](https://arxiv.org/abs/2106.14019) [[code]](https://github.com/hwanheelee1993/UMIC)

1. **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** *Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi* `arXiv` [[pdf]](https://arxiv.org/abs/2201.12086) [[code]](https://github.com/salesforce/BLIP)

1. **CyCLIP: Cyclic Contrastive Language-Image Pretraining** *Shashank Goel, Hritik Bansal, Sumit Bhatia, Ryan A. Rossi, Vishwa Vinay, Aditya Grover* `arXiv` [[pdf]](https://arxiv.org/abs/2205.14459) [[code]](https://github.com/goel-shashank/CyCLIP)

1. **Learning Video Representations using Contrastive Bidirectional Transformer** *Chen Sun, Fabien Baradel, Kevin Murphy, Cordelia Schmid* `arXiv` [[pdf]](https://arxiv.org/abs/1906.05743)

### Others

1. **Towards Unsupervised Dense Information Retrieval with Contrastive Learning** *Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave* `arXiv` [[pdf]](https://arxiv.org/abs/2112.09118)

1. **Text and Code Embeddings by Contrastive Pre-Training** *Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, Lilian Weng* `arXiv` [[pdf]](https://arxiv.org/abs/2201.10005) [[code]](https://openai.com/blog/introducing-text-and-code-embeddings/)

1. **Multi-Level Contrastive Learning for Cross-Lingual Alignment** *Beiduo Chen, Wu Guo, Bin Gu, Quan Liu, Yongchao Wang* `ICASSP 2022` [[pdf]](https://arxiv.org/abs/2202.13083) [[code]](https://github.com/salesforce/BLIP)

1. **Understanding Hard Negatives in Noise Contrastive Estimation** *Wenzheng Zhang, Karl Stratos* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2104.06245) [[code]](https://github.com/WenzhengZhang/hard-nce-el)

1. **Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup** *Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan* `RepL4NLP 2021` [[pdf]](https://arxiv.org/abs/2101.06983) [[code]](https://github.com/luyug/GradCache)

1. **Contrastive Distillation on Intermediate Representations for Language Model Compression** *Siqi Sun, Zhe Gan, Yu Cheng, Yuwei Fang, Shuohang Wang, Jingjing Liu* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2009.14167) [[code]](https://github.com/intersun/CoDIR)

1. **FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders** *Pengyu Cheng, Weituo Hao, Siyang Yuan, Shijing Si, Lawrence Carin* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2103.06413)

1. **Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence** *Tal Schuster, Adam Fisch, Regina Barzilay* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2103.08541) [[code]](https://github.com/TalSchuster/VitaminC)

1. **Supporting Clustering with Contrastive Learning** *Dejiao Zhang, Feng Nan, Xiaokai Wei, Shangwen Li, Henghui Zhu, Kathleen McKeown, Ramesh Nallapati, Andrew Arnold, Bing Xiang* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2103.12953) [[code]](https://github.com/amazon-research/sccl)

1. **Modeling Discriminative Representations for Out-of-Domain Detection with Supervised Contrastive Learning** *Zhiyuan Zeng, Keqing He, Yuanmeng Yan, Zijun Liu, Yanan Wu, Hong Xu, Huixing Jiang, Weiran Xu* `ACL 2021` [[pdf]](https://arxiv.org/abs/2105.14289) [[code]](https://github.com/parZival27/supervised-contrastive-learning-for-out-of-domain-detection)

1. **Contrastive Out-of-Distribution Detection for Pretrained Transformers** *Wenxuan Zhou, Fangyu Liu, Muhao Chen* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.08812) [[code]](https://github.com/wzhouad/Contra-OOD)

1. **Contrastive Fine-tuning Improves Robustness for Neural Rankers** *Xiaofei Ma, Cicero Nogueira dos Santos, Andrew O. Arnold* `Findings of ACL 2021` [[pdf]](https://arxiv.org/abs/2105.12932)

1. **Contrastive Code Representation Learning** *Paras Jain, Ajay Jain, Tianjun Zhang, Pieter Abbeel, Joseph E. Gonzalez, Ion Stoica* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2007.04973) [[code]](https://github.com/parasj/contracode)

1. **Knowledge Representation Learning with Contrastive Completion Coding** *Bo Ouyang, Wenbing Huang, Runfa Chen, Zhixing Tan, Yang Liu, Maosong Sun, Jihong Zhu* `Findings of EMNLP 2021` [[pdf]](https://aclanthology.org/2021.findings-emnlp.263.pdf)

1. **Adversarial Training with Contrastive Learning in NLP** *Daniela N. Rim, DongNyeong Heo, Heeyoul Choi* `arXiv` [[pdf]](https://arxiv.org/abs/2109.09075)

1. **Simple Contrastive Representation Adversarial Learning for NLP Tasks** *Deshui Miao, Jiaqi Zhang, Wenbo Xie, Jian Song, Xin Li, Lijuan Jia, Ning Guo* `arXiv` [[pdf]](https://arxiv.org/abs/2111.13301)

1. **Learning To Retrieve Prompts for In-Context Learning** *Ohad Rubin, Jonathan Herzig, Jonathan Berant* `arXiv` [[pdf]](https://arxiv.org/abs/2112.08633)

1. **RELiC: Retrieving Evidence for Literary Claims** *Katherine Thai, Yapei Chang, Kalpesh Krishna, Mohit Iyyer* `ACL 2022` [[pdf]](https://arxiv.org/abs/2203.10053)[[code]](https://relic.cs.umass.edu/)

1. **Multi-Level Contrastive Learning for Cross-Lingual Alignment** *Beiduo Chen, Wu Guo, Bin Gu, Quan Liu, Yongchao Wang* `ICASSP 2022` [[pdf]](https://arxiv.org/abs/2202.13083)

1. **Multi-Scale Self-Contrastive Learning with Hard Negative Mining for Weakly-Supervised Query-based Video Grounding** *Shentong Mo, Daizong Liu, Wei Hu* `arXiv` [[pdf]](https://arxiv.org/abs/2203.03838)

1. **Contrastive Demonstration Tuning for Pre-trained Language Models** *Xiaozhuan Liang, Ningyu Zhang, Siyuan Cheng, Zhen Bi, Zhenru Zhang, Chuanqi Tan, Songfang Huang, Fei Huang, Huajun Chen* `arXiv` [[pdf]](https://arxiv.org/abs/2204.04392)[[code]](https://github.com/zjunlp/PromptKG/tree/main/research/Demo-Tuning)

1. **GL-CLeF: A Global-Local Contrastive Learning Framework for Cross-lingual Spoken Language Understanding** *Libo Qin, Qiguang Chen, Tianbao Xie, Qixin Li, Jian-Guang Lou, Wanxiang Che, Min-Yen Kan* `ACL 2022` [[pdf]](https://arxiv.org/abs/2204.08325)[[code]](https://github.com/LightChen233/GL-CLeF)

1. **Zero-Shot Stance Detection via Contrastive Learning** *Bin Liang, Zixiao Chen, Lin Gui, Yulan He, Min Yang, and Ruifeng Xu* `WWW 2022` [[pdf]](https://dl.acm.org/doi/10.1145/3485447.3511994)[[code]](https://github.com/HITSZ-HLT/PT-HCL)

1. **Multi-level Contrastive Learning for Cross-lingual Spoken Language Understanding** *Shining Liang, Linjun Shou, Jian Pei, Ming Gong, Wanli Zuo, Xianglin Zuo, Daxin Jiang* `arXiv` [[pdf]](https://arxiv.org/abs/2205.03656)

1. **MERIt: Meta-Path Guided Contrastive Learning for Logical Reasoning** *Fangkai Jiao, Yangyang Guo, Xuemeng Song, Liqiang Nie* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.276/)

1. **The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking** *Yinghui Li, Qingyu Zhou, Yangning Li, Zhongli Li, Ruiyang Liu, Rongyi Sun, Zizhen Wang, Chao Li, Yunbo Cao, Hai-Tao Zheng* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.252/)

1. **Mitigating Contradictions in Dialogue Based on Contrastive Learning** *Weizhao Li, Junsheng Kong, Ben Liao, Yi Cai* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.219/)

1. **Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems** *Zhongli Li, Wenxuan Zhang, Chao Yan, Qingyu Zhou, Chao Li, Hongzhi Liu, Yunbo Cao* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.195/)

1. **Mitigating the Inconsistency Between Word Saliency and Model Confidence with Pathological Contrastive Training** *Pengwei Zhan, Yang Wu, Shaolei Zhou, Yunjian Zhang, Liming Wang* `Findings of ACL 2022` [[pdf]](https://aclanthology.org/2022.findings-acl.175/)

1. **Disentangled Knowledge Transfer for OOD Intent Discovery with Unified Contrastive Learning** *Yutao Mou, Keqing He, Yanan Wu, Zhiyuan Zeng, Hong Xu, Huixing Jiang, Wei Wu, Weiran Xu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-short.6/)

1. **JointCL: A Joint Contrastive Learning Framework for Zero-Shot Stance Detection** *Bin Liang, Qinglin Zhu, Xiang Li, Min Yang, Lin Gui, Yulan He, Ruifeng Xu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.7/)

1. **New Intent Discovery with Pre-training and Contrastive Learning** *Yuwei Zhang, Haode Zhang, Li-Ming Zhan, Xiao-Ming Wu, Albert Lam* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.21/)

1. **RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining** *Hui Su, Weiwei Shi, Xiaoyu Shen, Zhou Xiao, Tuo Ji, Jiarui Fang, Jie Zhou* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.65/)

1. **Sentence-aware Contrastive Learning for Open-Domain Passage Retrieval** *Wu Hong, Zhuosheng Zhang, Jinyuan Wang, Hai Zhao* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.76/)

1. **Improving Event Representation via Simultaneous Weakly Supervised Contrastive Learning and Clustering** *Jun Gao, Wei Wang, Changlong Yu, Huan Zhao, Wilfred Ng, Ruifeng Xu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.216/)

1. **Contrastive Visual Semantic Pretraining Magnifies the Semantics of Natural Language Representations** *Robert Wolfe, Aylin Caliskan* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.217/)

1. **Multilingual Molecular Representation Learning via Contrastive Pre-training** *Zhihui Guo, Pramod Sharma, Andy Martinez, Liang Du, Robin Abraham* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.242/)

1. **SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models** *Liang Wang, Wei Zhao, Zhuoyu Wei, Jingming Liu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.295/)

1. **Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models** *Zaiqiao Meng, Fangyu Liu, Ehsan Shareghi, Yixuan Su, Charlotte Collins, Nigel Collier* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.329/)

1. **KNN-Contrastive Learning for Out-of-Domain Intent Classification** *Yunhua Zhou, Peiju Liu, Xipeng Qiu* `ACL 2022` [[pdf]](https://aclanthology.org/2022.acl-long.352/)

1. **Cross-modal Contrastive Learning for Speech Translation** *Rong Ye, Mingxuan Wang, Lei Li* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.02444)

1. **Revisit Overconfidence for OOD Detection: Reassigned Contrastive Learning with Adaptive Class-dependent Threshold** *Yanan Wu, Keqing He, Yuanmeng Yan, QiXiang Gao, Zhiyuan Zeng, Fujia Zheng, Lulu Zhao, Huixing Jiang, Wei Wu, Weiran Xu* `NAACL 2022` [[pdf]]()

1. **Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities** *Benjamin Hsu, Graham Horwood* `NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.11438)

1. **Domain Confused Contrastive Learning for Unsupervised Domain Adaptation** *Quanyu Long, Tianze Luo, Wenya Wang, Sinno Pan* `NAACL 2022` [[pdf]]()

1. **Intent Detection and Discovery from User Logs via Deep Semi-Supervised Contrastive Clustering** *Rajat Kumar, Mayur Patidar, VAIBHAV VARSHNEY, Lovekesh Vig, Gautam Shroff
* `NAACL 2022` [[pdf]]()

1. **Detect Rumors in Microblog Posts for Low-Resource Domains via Adversarial Contrastive Learning** *Hongzhan Lin, Jing Ma, Liangliang Chen, Zhiwei Yang, Mingfei Cheng, Guang Chen* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2204.08143)

1. **CLMLF:A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection** *Zhen Li, Bing Xu, Conghui Zhu, Tiejun Zhao* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2204.05515)

1. **Prompt Augmented Generative Replay via Supervised Contrastive Learning for Lifelong Intent Detection** *VAIBHAV VARSHNEY, Mayur Patidar, Rajat Kumar, Lovekesh Vig, Gautam Shroff* `Findings of NAACL 2022` [[pdf]]()

1. **CODE-MVP: Learning to Represent Source Code from Multiple Views with Contrastive Pre-Training** *Xin Wang, Yasheng Wang, Yao Wan, Jiawei Wang, Pingyi Zhou, Li Li, Hao Wu, Jin Liu* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2205.02029)

1. **Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks** *Zhao Meng, Yihan Dong, Mrinmaya Sachan, Roger Wattenhofer* `Findings of NAACL 2022` [[pdf]](https://arxiv.org/abs/2107.07610)

## Contributor
Please contact [Rui Zhang](https://ryanzhumich.github.io/) if you want to add any references!