---
layout:     post
title:      "Deep Q Learning in Pong Game"
subtitle:   " \"This is a swift and cool tool in Deep Learning\""
date:       2017-01-20 12:00:00
author:     "S&W"
header-img: "img/post-deepq-2017.jpg"
catalog: true
tags:
    - Deep Reinforcement Learning
---

> "用二进制的眼光看世界，似乎有点深邃，似乎又有点简单。"

# 什么是Deep Q Learning ？

Deep Q Learning 是一个数学公式很复杂，但道理太简单的学习。
在看本篇内容前，请先阅读 Google 的关于DQN的ppt，[Deep Reinforcement Learning. David Silver, Google DeepMind](http://www.iclr.cc/lib/exe/fetch.php?media=iclr2015:silver-iclr2015.pdf)，数学部分不需要特别看懂。
## 走别人走过的路
首先我们解释下什么是，Q Learning：

$$
Q_{i+1}(s,a) = E[r + \theta \max_{a'}Q_i(s',a')|s,a]
$$

这个公式其实很容易理解，它就是说，第i轮的value，$$Q(s,a)$$和第i+1轮的$$Q(s',a')$$的关系，也就是当前利益$$r$$加上未来最大利益$$\max_{a'}Q_i(s',a')$$乘以一个折扣$$\theta$$(为什么用折扣$$\theta$$,大家可以看看经济学书，因为我们总是对未来利益打折计算的)。其实这个过程就是第i个人帮你走过一段路，他总结了这段路上所得所失，你作为第i+1个人就会避免他走过的错路，就会期望走他走的对的路。所以每一个i+1都是站在i的人的肩膀上走路。
## 有点探索精神，Exploiting or Exploring
走别人走过的路很简单，但是似乎世界需要探索。如果你不设置探索，很多时候未知的路就走不出来，未知的美就探索不到。其实用数学的语言就是，你会陷入一个局部最优，而不是全局最优。所以我们经常会讨论我们该花多少概率探索未知，花多少时间挖掘已知。
## Deep：为了找到“似曾相识”的路
其实，一直以来，Q Learning唯一可以炫耀的其实是数学上收敛这个性质，应该是$$\theta < 1$$的原因吧。其实Q Learning就是个暴力搜索。

那么为什么在引入Deep以后火起来了呢，因为Deep可以记住很像的图像。Deep的本质是感知，就是能看出来所谓很像的东西，感觉人的语气，感情，这就是Deep的好处。而且Deep处理的越好，收敛速度越快，因为“举一反三”。
# 代码解读
* 深度网络, 输入是上篇文章的prepro函数处理过的图片，当然如果换成cnn应该训练速度会增加，输出是每一个action的预估Q值。

{% highlight ruby %}
states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
network = tl.layers.InputLayer(states_batch_pl, name='input_layer')
network = tl.layers.DenseLayer(network, n_units=H,
                                        act = tf.nn.relu, name='relu1')
network = tl.layers.DenseLayer(network, n_units=3,
                            act = tf.identity, name='output_layer')
Q_s = network.outputs
{% endhighlight %}

* 探索或者挖掘，我们不能停止探索，但也不能只有漫无目的地探索未知。因此$$episilon$$是一个trade-off，它调节我们探索与挖掘的比例，在初期，我们没有很多经验的时候，我们倾向高概率地探索，但在后期，我们倾向给予更高的概率去挖掘以前的知识，所以$$episilon$$设置为$$0.1$$。

{% highlight ruby %}
  if np.random.random() <= episilon:
            action = np.random.choice([1,2,3])
        else:
            action = np.argmax(prob) + 1
{% endhighlight %}

* Memory and Skip, 我们用了点trick，先把所有的尝试的路存储在Memory中，然后随机出一些路训练，为了是打破多个图片的联系性，提高速度。Skip是在$$n$$个图片之间我们不做action的变化，这样可以加速整个训练过程。

{% highlight ruby %}

        for i in range(0, Skip):
            if done:
                break;
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
        cur_x = prepro(observation)
        cur_x = cur_x.reshape(1, D)
        M.append((prev_x,action-1,reward,cur_x,done))
        
{% endhighlight %}

* Q值计算：


{% highlight ruby %}

	if terminal:
     	targets[i] = reward_t
	else:
     	targets[i] = reward_t + gamma * np.max(Q_sa)

{% endhighlight %}         
                        


