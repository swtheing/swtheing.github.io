#! /usr/bin/python
# -*- coding: utf8 -*-


"""
To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. Before we start, we highly recommend
you to go through a famous blog called “Deep Reinforcement Learning: Pong from
Pixels” which is a minimalistic implementation of deep reinforcement learning by
using python-numpy and OpenAI gym environment.

The code here is the reimplementation of Karpathy's Blog by using TensorLayer.

Compare with Karpathy's code, we store observation for a batch, he store
observation for a episode only, they store gradients instead. (so we will use
more memory if the observation is very large.)

Link
-----
http://karpathy.github.io/2016/05/31/rl/

"""

import tensorflow as tf
import tensorlayer as tl
import gym
import numpy as np
import time
import random
from collections import deque


# hyperparameters
image_size = 80
D = image_size * image_size
H = 200
BATCH = 32
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = False      # display the game environment
resume = False      # load existing policy network
model_file_name = "model_pong"
REPLAY_MEMORY = 50000
np.set_printoptions(threshold=np.nan)
Initial_epi = 1.0
Observe = 3200
Skip = 0
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
model_file_name = "model_pong_no_skip"


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []
M = deque()
# observation for training and inference
states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
network = tl.layers.InputLayer(states_batch_pl, name='input_layer')
network = tl.layers.DenseLayer(network, n_units=H,
                                        act = tf.nn.relu, name='relu1')
network = tl.layers.DenseLayer(network, n_units=3,
                            act = tf.identity, name='output_layer')
Q_s = network.outputs
a = tf.placeholder(tf.float32,shape=[None,3])
Q = tf.placeholder(tf.float32,shape=[None])
action_q_values = tf.reduce_sum(tf.mul(Q_s, a), reduction_indices=1)
loss = tf.reduce_mean(tf.square(Q - action_q_values))
train_op = tf.train.AdamOptimizer(learning_rate, decay_rate).minimize(loss)

r_episilon = 0
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    if resume:
        load_params = tl.files.load_npz(name=model_file_name+'.npz')
        tl.files.assign_params(sess, load_params, network)
    network.print_params()
    network.print_layers()
    episilon = Initial_epi
    ave_Q_max = 0.0
    while True:
        Q_sa = 0
        if render: env.render()
        if prev_x is None:
            prev_x = np.zeros(D)
        #print(prev_x)
        prev_x = prev_x.reshape(1,D)
        prob = sess.run(
            Q_s,
            feed_dict={states_batch_pl: prev_x}
        )
        # action. 1: STOP  2: UP  3: DOWN
        if np.random.random() <= episilon:
            action = np.random.choice([1,2,3])
        else:
            action = np.argmax(prob) + 1
        ave_Q_max += np.max(prob)
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        for i in range(0, Skip):
            if done:
                break;
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
        cur_x = prepro(observation)
        cur_x = cur_x.reshape(1, D)
        M.append((prev_x,action-1,reward,cur_x,done))
        prev_x = cur_x
        if episilon !=0 and episilon > FINAL_EPSILON and episode_number > Observe:
            episilon -= (Initial_epi - FINAL_EPSILON) / EXPLORE
        if len(M) > REPLAY_MEMORY:
            M.popleft()
        if done:
            prev_x = None
            observation = env.reset()
            if episode_number % 50 == 0:
                r_episilon = episilon
                episilon = 0
                tl.files.save_npz(network.all_params, name=model_file_name + '.npz')
                print("reward = " + str(reward_sum))
                print("Q = " + str(ave_Q_max))
                ave_Q_max = 0
            else:
                episilon = r_episilon
            if episode_number % 50 == 1:
                print("episilon" + str(episilon))
            episode_number += 1
            reward_sum = 0
            if episode_number == Observe:
                print("start training:")
            if episode_number > Observe:
                input = np.zeros((BATCH,D))
                action = np.zeros((BATCH,3))
                targets = np.zeros(BATCH)
                minibatch = random.sample(M, BATCH)
                for i in range(0,len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    input[i] = state_t
                #print(reward_t)
                    Q_sa = sess.run(
                           Q_s,
                           feed_dict={states_batch_pl: state_t1}
                            ) # Hitting each buttom probability
                    action[i,action_t] = 1
                    if terminal:
                        targets[i] = reward_t
                    else:
                        targets[i] = reward_t + gamma * np.max(Q_sa)
                _ = sess.run([train_op],
                        feed_dict={
                        states_batch_pl: input,
                        a: action,
                        Q: targets}
                    )




