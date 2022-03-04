<p align="center">
<h2 align="center"> Contrastive Learning for Natural Language Processing Papers</h2>
</p>

## Tutorial and Survey

* **Contrastive Data and Learning for Natural Language Processing** *Rui Zhang, Yangfeng Ji, Yue Zhang, Rebecca J. Passonneau* `NAACL 2022 Tutotrial` [[website]](https://contrastive-nlp-tutorial.github.io/)

* **A Primer on Contrastive Pretraining in Language Processing: Methods, Lessons Learned and Perspectives** *Nils Rethmeier, Isabelle Augenstein* [[pdf]](https://arxiv.org/abs/2102.12982)

## Presentations and Blogs

* **Contrastive Representation Learning in Text** *Danqi Chen* [[slide]](https://cds.nyu.edu/wp-content/uploads/2021/11/TaD-Slides-Danqi-Chen-compressed.pdf)

* **Contrastive pairs are better than independent samples, for both learning and evaluation** *Matt Gardner* [[video]](https://drive.google.com/file/d/1DWMDeUzy9m0Z5a1gzQm4I78ZEQp8gyhm/view)

* **Contrastive Representation Learning** *Lilian Weng* [[blog]](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

* **Phrase Retrieval and Beyond** *Jinhyuk Lee* [[blog]](https://princeton-nlp.github.io/phrase-retrieval-and-beyond/)

## Foundations of Contrastive Learning

### Contrastive Learning Objectives
1. **Learning a similarity metric discriminatively, with application to face verification** *Sumit Chopra, Raia Hadsell, Yann LeCun* `CVPR 2005` [[pdf]](https://ieeexplore.ieee.org/abstract/document/1467314)

1. **Facenet: A unified embedding for face recognition and clustering** *Florian Schroff, Dmitry Kalenichenko, and James Philbin* `CVPR 2015` [[pdf]](https://arxiv.org/abs/1503.03832)

1. **Deep metric learning via lifted structured feature embedding** *Hyun Oh Song, Yu Xiang, Stefanie Jegelka, Silvio Savarese* `CVPR 2016` [[pdf]](https://arxiv.org/abs/1511.06452)

1. **Improved deep metric learning with multi-class n-pair loss objective** *Kihyuk Sohn* `NeurIPS 2016` [[pdf]](https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf)

1. **Noise-contrastive estimation: A new estimation principle for unnormalized statistical models** *Michael Gutmann and Aapo Hyvärinen* `AISTATS 2010` [[pdf]](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

1. **Representation learning with contrastive predictive coding** *Aaron van den Oord, Yazhe Li, Oriol Vinyals* `arXiv` [[pdf]](https://arxiv.org/abs/1807.03748)

1. **Learning a nonlinear embedding by preserving class neighbourhood structure** *Ruslan Salakhutdinov, Geoff Hinton* `AISTATS 2007` [[pdf]](http://proceedings.mlr.press/v2/salakhutdinov07a/salakhutdinov07a.pdf)

1. **Analyzing and improving representations with the soft nearest neighbor loss** *Nicholas Frosst, Nicolas Papernot, Geoffrey Hinton* `ICML 2019` [[pdf]](http://proceedings.mlr.press/v97/frosst19a/frosst19a.pdf)

### Sampling Strategy for Contrastive Learning 
1. **Debiased Contrastive Learning** *Ching-Yao Chuang, Joshua Robinson, Lin Yen-Chen, Antonio Torralba, Stefanie Jegelka* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2007.00224)

1. **Contrastive Learning with Hard Negative Samples** *Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, Stefanie Jegelka* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2010.04592)

1. **Supervised Contrastive Learning** *Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2004.11362)

1. **Adversarial Self-Supervised Contrastive Learning** *Minseon Kim, Jihoon Tack, Sung Ju Hwang* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2006.07589) [[code]](https://github.com/Kim-Minseon/RoCL)

### Most Notable Applications of Contrastive Learning 
1. **Efficient Estimation of Word Representations in Vector Space** *Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean* `arXiv` [[pdf]](https://arxiv.org/abs/1301.3781)

1. **A Simple Framework for Contrastive Learning of Visual Representations** *Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton* `ICML 2020` [[pdf]](https://arxiv.org/abs/2002.05709) [[code]](https://github.com/google-research/simclr)

1. **Learning Transferable Visual Models From Natural Language Supervision** *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever* `arXiv` [[pdf]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP)

### Analysis of Contrastive Learning
1. **What Makes for Good Views for Contrastive Learning?** *Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, Phillip Isola* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2005.10243) [[code]](https://hobbitlong.github.io/InfoMin/)

1. **Demystifying Contrastive Self-Supervised Learning: Invariances, Augmentations and Dataset Biases** *Senthil Purushwalkam, Abhinav Gupta* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2007.13916)

1. **What Should Not Be Contrastive in Contrastive Learning** *Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2008.05659)

1. **Poisoning and Backdooring Contrastive Learning** *Nicholas Carlini, Andreas Terzis* `ICLR 2022` [[pdf]](https://arxiv.org/abs/2106.09667)

## Contrastive Learning for NLP

### Contrastive Data Augmentation for NLP

1. **Learning the Difference that Makes a Difference with Counterfactually-Augmented Data** *Divyansh Kaushik, Eduard Hovy, Zachary C. Lipton* `ICLR 2020` [[pdf]](https://arxiv.org/abs/1909.12434) [[code]](https://github.com/acmi-lab/counterfactually-augmented-data)

1. **NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation** *Kaustubh D. Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Srivastava, Samson Tan, Tongshuang Wu, Jascha Sohl-Dickstein, Jinho D. Choi, Eduard Hovy, Ondrej Dusek, Sebastian Ruder, Sajant Anand, Nagender Aneja, Rabin Banjade, Lisa Barthe, Hanna Behnke, Ian Berlot-Attwell, Connor Boyle, Caroline Brun, Marco Antonio Sobrevilla Cabezudo, Samuel Cahyawijaya, Emile Chapuis, Wanxiang Che, Mukund Choudhary, Christian Clauss, Pierre Colombo, Filip Cornell, Gautier Dagan, Mayukh Das, Tanay Dixit, Thomas Dopierre, Paul-Alexis Dray, Suchitra Dubey, Tatiana Ekeinhor, Marco Di Giovanni, Rishabh Gupta, Rishabh Gupta, Louanes Hamla, Sang Han, Fabrice Harel-Canada, Antoine Honore, Ishan Jindal, Przemyslaw K. Joniak, Denis Kleyko, Venelin Kovatchev, Kalpesh Krishna, Ashutosh Kumar, Stefan Langer, Seungjae Ryan Lee, Corey James Levinson, Hualou Liang, Kaizhao Liang, Zhexiong Liu, Andrey Lukyanenko, Vukosi Marivate, Gerard de Melo, Simon Meoni, Maxime Meyer, Afnan Mir, Nafise Sadat Moosavi, Niklas Muennighoff, Timothy Sum Hon Mun, Kenton Murray, Marcin Namysl, Maria Obedkova, Priti Oli, Nivranshu Pasricha, Jan Pfister, Richard Plant, Vinay Prabhu, Vasile Pais, Libo Qin, Shahab Raji, Pawan Kumar Rajpoot, Vikas Raunak, Roy Rinberg, Nicolas Roberts, Juan Diego Rodriguez, Claude Roux, Vasconcellos P. H. S., Ananya B. Sai, Robin M. Schmidt, Thomas Scialom, Tshephisho Sefara, Saqib N. Shamsi, Xudong Shen, Haoyue Shi, Yiwen Shi, Anna Shvets, Nick Siegel, Damien Sileo, Jamie Simon, Chandan Singh, Roman Sitelew, Priyank Soni , Taylor Sorensen, William Soto, Aman Srivastava, KV Aditya Srivatsa, Tony Sun, Mukund Varma T, A Tabassum, Fiona Anting Tan, Ryan Teehan, Mo Tiwari, Marie Tolkiehn, Athena Wang, Zijian Wang, Gloria Wang, Zijie J. Wang, Fuxuan Wei, Bryan Wilie, Genta Indra Winata, Xinyi Wu, Witold Wydmański, Tianbao Xie, Usama Yaseen, M. Yee, Jing Zhang, Yue Zhang* `arXiv` [[pdf]](https://arxiv.org/abs/2112.02721) [[code]](https://github.com/GEM-benchmark/NL-Augmenter)

### Text Classification

### Sentence Embeddings and Phrase Embeddings

1. **Towards Universal Paraphrastic Sentence Embeddings** *John Wieting, Mohit Bansal, Kevin Gimpel, Karen Livescu* `ICLR 2016` [[pdf]](https://arxiv.org/abs/1511.08198) [[code]](https://github.com/jwieting/iclr2016)

1. **SimCSE: Simple Contrastive Learning of Sentence Embeddings** *Tianyu Gao, Xingcheng Yao, Danqi Chen* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.08821) [[code]](https://github.com/princeton-nlp/simcse)

1. **Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders** *Fangyu Liu, Ivan Vulić, Anna Korhonen, Nigel Collier* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2104.08027) [[code]](https://github.com/cambridgeltl/mirror-bert)

1. **Learning Dense Representations of Phrases at Scale** *Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, Danqi Chen* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.12624) [[code]](https://github.com/princeton-nlp/DensePhrases)

1. **Phrase Retrieval Learns Passage Retrieval, Too** *Jinhyuk Lee, Alexander Wettig, Danqi Chen* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.08133) [[code]](https://github.com/princeton-nlp/DensePhrases)

### Information Extraction

### Sequence Labeling
1. **Contrastive Estimation: Training Log-Linear Models on Unlabeled Data** *Noah A. Smith, Jason Eisner* `ACL 2005` [[pdf]](https://aclanthology.org/P05-1044.pdf)

### Machine Translation

### Question Answering 

### Summarization

1. **CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization** *Shuyang Cao, Lu Wang* `EMNLP 2021` [[pdf]](https://arxiv.org/abs/2109.09209) [[code]](https://shuyangcao.github.io/projects/cliff_summ)

### Text Generation

1. **A Contrastive Framework for Neural Text Generation** *Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier* `arXiv` [[pdf]](https://arxiv.org/abs/2202.06417) [[code]](https://github.com/yxuansu/SimCTG)

### Few-shot Learning

### Contrastive Pretraining

1. **COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining** *Yu Meng, Chenyan Xiong, Payal Bajaj, Saurabh Tiwary, Paul Bennett, Jiawei Han, Xia Song* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2102.08473) [[code]](https://github.com/microsoft/COCO-LM)

1. **TaCL: Improving BERT Pre-training with Token-aware Contrastive Learning** *Yixuan Su, Fangyu Liu, Zaiqiao Meng, Tian Lan, Lei Shu, Ehsan Shareghi, Nigel Collier* `arXiv` [[pdf]](https://arxiv.org/abs/2111.04198) [[code]](https://github.com/yxuansu/TaCL)

### Interpretability and Explainability

### Commonsense Knowledge and Reasoning

### Vision-and-Language

1. **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** *Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi* `arXiv` [[pdf]](https://arxiv.org/abs/2201.12086) [[code]](https://github.com/salesforce/BLIP)

### Others

1. **Towards Unsupervised Dense Information Retrieval with Contrastive Learning** *Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave* `arXiv` [[pdf]](https://arxiv.org/abs/2112.09118)

1. **Text and Code Embeddings by Contrastive Pre-Training** *Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, Lilian Weng* `arXiv` [[pdf]](https://arxiv.org/abs/2201.10005) [[code]](https://openai.com/blog/introducing-text-and-code-embeddings/)

## Contributor
[Rui Zhang](https://ryanzhumich.github.io/)