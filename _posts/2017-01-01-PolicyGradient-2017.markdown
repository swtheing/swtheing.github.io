---
layout:     post
title:      "Policy Gradient in Pong Game"
subtitle:   " \"This is a swift and cool tool in Deep Learning\""
date:       2017-01-02 12:00:00
author:     "S&W"
header-img: "img/post-policygradient-2017.jpg"
catalog: true
tags:
    - Deep Reinforcement Learning
---

> “我听过两个关于罗曼蒂克的故事，一个发生在上海，另一个也发生在上海。”

## Deep Reinforcement Learning
Deep Reinforce Learning可以从下面几个资料中获得，
* [Reinforcement Learning: An Introduction. Richard S. Sutton and Andrew G. Barto](https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html)，这本书可以说是整个Reinforcement Learning最权威的书籍，尤其第二版加入了很多关于Deep的东西。
* [Deep Reinforcement Learning. David Silver, Google DeepMind](http://www.iclr.cc/lib/exe/fetch.php?media=iclr2015:silver-iclr2015.pdf)，这是DeepMind一个ppt主要讲的是Deep Q-Network。
* [UCL的一个关于Deep Reinforcement Learning的课程](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)。
* [Policy Gradient Tutoiral](http://karpathy.github.io/2016/05/31/rl/)，本文的代码以及解释来源于此。

## Policy Gradient
### what is policy gradient
在笔者看来，Policy Gradient是对于一个Policy的求导，也就是对于一个映射函数（state -> action）的求导。
### why policy gradient
Policy Gradient的出现是因为Deep Learning的出现，否则，怎么会有人想到为一个Policy函数求Gradient呢。相比于Deep Q-learning，Policy Gradient更容易实现，也更容易做BP，如果想了解细节，请看[Policy Gradient Tutoiral](http://karpathy.github.io/2016/05/31/rl/)。
### Loss of Policy Gradient
其实Policy Gradient的Loss是Log Loss，也就是Cross Entropy，但是有一个trick的地方是，Cross Entropy计算出的gradient的只是一个方向，它的值需要再乘以discount reward，具体数学可以看[Policy Gradient Tutoiral](http://karpathy.github.io/2016/05/31/rl/)，代码部分解释可以看下面。

## [代码解读](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
* observation 代码，`prepro(I)`函数中，做了一些剔除背景的操作。

```javascript
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()
```
* 计算discount_reward代码，其中对于`[0,0,0,1]`这样的得分数组，如果设`gamma = 0.9`，那么得到打折后的得分是`[0.729,0.81,0.9,1]`这样的打折分数：

```javascript
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r
```
* `Policy Forward` 和 `Policy Backward`算法，就是普通的foward 和 BP算法。例如backward中的`dW2 = np.dot(eph.T, epdlogp).ravel()`就是使用了链式法则去求Gradient。
* 下面一段是Open AI Gym的一些配置，只要知道`render`是gym中是否显示图像的开关，如果在服务器上训练，请关掉。
* 下面代码是整个文章的核心，首先声明一点，在原文章代码中它只考虑action= 2 or 3的两种情况（上或者下），也就是没有考虑1，也就是停止的情况。因此在计算Gradient的时候，你会发现它使用的Gradient方向是`y-aprob`，为什么如此，因为我们可以看一下整个aprob计算，它是通过sigmoid函数计算出概率值aprob。因此，我们就是计算 ylog(aprob) + (1-y)log(1-aprob)的倒数值，又 aprob = σ(x)，所以结果是`y-aprob`，[具体可以看一下logloss的gradient推倒方法](http://cs231n.github.io/neural-networks-2/#losses)

```javascript
while True:
  if render: env.render()
  
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) 
```
* 最后其实就是如何存储Gradient以及如何计算batch Gradient的方法，大家可以依据很多深度学习资料比对一下，至于running_reward的公式的原理，大家可以看看Deep Reinforcement Learning的书籍。

```javascript
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
```

## Policy Gradient in Pong Game with Tensorflow
这段代码是我根据网上一个人的代码改写的，他写的代码虽然说是Policy Gradient，但实际只用了L2的loss，其实是错误的。

```javascript
import numpy as np
import gym
import tensorflow as tf

# hyperparameters
n_obs = 80 * 80           # dimensionality of observations
h = 200                   # number of hidden layer neurons
n_actions = 3             # number of available actions
learning_rate = 1e-4
gamma = 0.99               # discount factor for reward
decay = 0.99              # decay rate for RMSProp gradients
batch_size = 10
save_path='models_batch/pong.ckpt'

# gamespace 
env = gym.make("Pong-v0") # environment info
observation = env.reset()
prev_x = None
xs,rs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
# initialize model
tf_model = {}
with tf.variable_scope('layer_one',reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two',reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)

# tf operations
def discount_rewards(r):
  discounted_r = np.zeros_like(r,dtype=np.float32)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def tf_policy_forward(x): #x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    return logp

# downsampling
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="tf_x")
tf_y = tf.placeholder(dtype=tf.int32, shape=[None],name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None], name="tf_epr")

# tf optimizer op
tf_aprob = tf_policy_forward(tf_x)
p = tf.nn.softmax(tf_aprob)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(tf_aprob, tf_y)
loss = tf.reduce_sum(tf.mul(cross_entropy,tf_epr))
train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(loss)

# tf graph initialization
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# try load saved model
saver = tf.train.Saver(tf.all_variables())
load_was_success = True 
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    episode_number = int(load_path.split('-')[-1])

#episode_number = 0
# training loop
while True:
#     if True: env.render()
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    x = x.reshape(1,n_obs)
    #print(x)
    prev_x = cur_x

    # stochastically sample a policy from the network
    feed = {tf_x: x}
    aprob = sess.run(p,feed)
    #aprob = tf.nn.softmax(t_aprob)
    #print(aprob)
    aprob = aprob[0,:]
    #print(aprob)
    action = np.random.choice([0,1,2], p=aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action+1)
    reward_sum += reward
    
    # record game history
    xs.append(x) ; ys.append(action) ; rs.append(reward)
    if done:
        episode_number += 1
        if(episode_number % batch_size == 0):
            # update running reward
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            dis = discount_rewards(np.asarray(rs))
            #dis2 = discount_rewards(np.asarray(rs))
            #print dis
            #print dis2
            dis -= np.mean(dis)
            dis /= np.std(dis)
            #print(dis)
        # parameter update
            feed = {tf_x: np.vstack(xs), tf_epr: dis, tf_y: np.asarray(ys)}
            _ , loss_val= sess.run([train_op,loss],feed)
            print(loss_val)


        # print progress console
            if episode_number % 50 == 0:
                print 'ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            else:
                print '\tep {}: reward: {}'.format(episode_number, reward_sum)
            # bookkeeping
            if episode_number % 50 == 0:
                saver.save(sess, save_path, global_step=episode_number)
                print "SAVED MODEL #{}".format(episode_number)
            xs,rs,ys = [],[],[] # reset game history
        observation = env.reset() # reset env
        reward_sum = 0
        prev_x = None
```
