'''Solves Pong with Policy Gradients in Tensorflow.'''
# written December 2016 by Wei Shen
# inspired by karpathy's gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
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
  """ take 1D float array of rewards and compute discounted reward """
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
#print(tf_grads)

# tf graph initialization
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# try load saved model
saver = tf.train.Saver(tf.all_variables())
load_was_success = True # yes, I'm being optimistic
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
