---
layout:     post
title:      "PairwiseRanking的介绍"
subtitle:   " \"This is a tool in L2R\""
date:       2017-06-11 12:00:00
author:     "S&W"
header-img: "img/post-pwintro-2017.jpg"
catalog: true
tags:
    - Learn to Rank，PairwiseRanking的专题。
---

> "每当我想批评别人的时候，我父亲总对我说，要记住，这世上并不是所有人，都有我拥有的那些优势。"

# 什么是Pairwise Ranking？

Pairwise Ranking是Learning to Rank(L2R)的一种方式。其中Ranking是一种在很多应用中需要解决的问题，比如，搜索文件，推荐系统，机器翻译等。那么在下面章节中，我们将首先描述一下Rank 和Learning to Rank方法；然后介绍其中比较著名的方法，Pairwise Ranking；最后介绍一下我在搜狐图文匹配大赛中使用Tensorflow编写的PairwiseRanking的方法和效果。

## Learning To Rank

Learning To Rank的定义：machine learning technology for ranking creation and ranking aggregation：

1. ranking creation: 基于一个query和一系列的候选集，对于基于这个query的相关度，对这个候选集合排序。其中，有全局排序模型以及局部排序模型。一般而言，使用局部排序模型居多，监督学习居多。
2. ranking aggregation: 基于多个已经排好序的偏序关系生成一个全序。一般使用全局排序模型，监督学习和无监督学习模型都存在。

### 技巧
1. 方法：pointwise，pairwise，listwise的方法。
2. 理论：Generalization，Consistency。
3. 应用：Search，Collaborative Filtering，Key Phrase Extraction。

### 主要研究问题
1. 数据标记：利用相关程度做标记；利用排序序列做标记；利用整个序列的顺序做标记（极少数）。
2. 特征工程：深度提取，一些标准提取。
3. 评价方法：NDCG (Normalized Discounted Cumulative Gain)，MAP (Mean Average Precision)，MRR (Mean Reciprocal Rank)，WTA (Winners Take All)，Kendall’s Tau。
4. 学习算法：SVM-based, Boosting-based, Neural Network-based, Others.

## Pairwise Ranking
1. 将ranking转成Pairwise 分类：给予任意一个配对(x1，x2)，输出
f(x1) - f(x2)。学习器就是学习函数f(x)。
2. 优势：相比于pointwise直接学习一个函数f(x)属于一个监督学习，其中，如果没有人为对每个x标注具体值，那么f(x)有可能出现样本不均衡状况，也就是负例太多的问题。而当使用pairwise的方法以后可以解决这类问题。
3. 劣势：并没有利用到query-document对的结构关系。
4. 关于这三种方法的一些结论：1）Pairwise以及listwise的方法比pointwise效果好，2）LabmdaMART现有阶段是公认比较好的方法（这是以前的一个共识，但是自从DNN被引入以后，类似于百度的Simnet和IRGAN都在某些数据集上拥有更好的表现），3）Pairwise和listwise没有特别的效果区别。

## Simnet模型
### 简单介绍
   这是我在百度NLP部实习学到的主要模型，这个模型被广泛利用在大搜，度秘以及广告等业务上。本来并不准备展现出来，但是百度NLP组已经将其技术发布在机器之心上,[百度NLP|神经网络语义匹配技术](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650727938&idx=3&sn=c0f0479d3edc4ca07cd2cec16be830bd&chksm=871b227cb06cab6af4f2f3c1aa405a12733d95d090c55d10753240facdbdea2d8767e08962eb&mpshare=1&scene=1&srcid=0615Z8RrrVK6msstXZcPhjVY&key=227c90f337be412a6384bdda58f606ec845987b98855fde3fde65e6fabaeeefbd3f741bbb8ac5f46bf20ab9d1c959e9f05024c1defb01296591b65b9323ef2da51a436780b22edb04bb31fc2023369d0&ascene=0&uin=ODcyOTEwNjYw&devicetype=iMac+MacBookPro13%2C1+OSX+OSX+10.12+build(16A2323a)&version=12020810&nettype=WIFI&fontScale=100&pass_ticket=tRRZU9r72M2vJDTRV%2BtpkBra3haXN0qOIhcDTxttcyQzhULNlEBgidR48k7bTgUV)，我在此利用自己的理解对其进行一定的补充。
   首先Simnet是一个End-to-End模型，所谓End-to-End就是它本质上是由多个层次组成，但是每个层次是有关联的，所以我们利用深度学习的方式不仅学习了每一层的参数，关键是学习到的是一个整体，是一个关系。换句话说，如果这些层次单独抽出来，一定没有它们在系统中表现的好。比较有特点的就是word-embedding这个层次，这是simnet在不同任务下都会训练出来的层次，但对于同样的文本不同任务的word-embeeding的结果并不相同。
   其次，Simnet是一个Pairwise Ranking的模型，在抽取出x1，x2的feature层以后，学习f(x),输出是f(x1) - f(x2)。
   第三，Simnet是一个基于SVM的模型，因为它的loss是Hinge Loss。
   最后，Simnet是一个融合模型，它对词和字粒度的模型进行融合，对表示方法可以融合比如LSTM，RNN，CNN以及BOW等。
   
### 模型
  该模型是由表示模块，评估模块和损失模块等组成。那么，下面图片是一个Simnet模型的具化表现形式，
![Simnet的一个表现形式](https://swtheing.github.io/swtheking.github.io/img/in-post/simnet_2017_06.png)

  其中输入层是一个典型的Pairwise的三元组(Pos, Query, Neg)，在Measurement Layer之前，都是表示模块，其中这里的表示模块就是简单的BOW。在进入Measurement的之前，所有Pos，Query，Neg都被影射到一个256维的空间里，在这个空间中我们希望Pos向量能和Query向量夹角越小越好，而Query和Neg的夹角越小越好。因此我们使用了cos做度量函数，HingeLoss做Loss函数。

### 应用和变形
1. 应用方面，在对话系统中，Query就是提问，Pos就是回答，Neg是随机选择的一句话。在搜索中，Query就是搜索问题，Pos就是被点击的文档，Neg就是未被点击的文档。
2. 变形方面，对于表示层的改变，其中有利用LSTM，RNN，CNN对其表示形式改变，对于评估模块，使用海明距会提高整个计算速度和降低计算复杂度。

## PairwiseRanking的具体应用
1. 搜狐比赛中使用Tensorflow写的PairwiseRanking。
2. 知乎比赛中使用Keras写的PairwiseRanking。
                        


