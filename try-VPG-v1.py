# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:05:31 2019

@author: Reza
"""

#%%
import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

if "../" not in sys.path:
  sys.path.append("../") 
from cliff_walking import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')

#%%
env = CliffWalkingEnv()
print('Action space: ', env.action_space)
print('Observation space: ', env.observation_space)


# Actor
class PolicyEstimator():
    def __init__(self, env, learning_rate=0.01, scope='Policy_Estimator'):
        tf.reset_default_graph()
        self.state_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='State_PH')
        self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='Action_PH')
        self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='Target_PH')
        
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        print('self.n_actions: ', self.n_actions)
        self.state_oh = tf.one_hot(self.state_ph, depth=self.n_states)
        self.action_oh = tf.one_hot(self.action_ph, depth=self.n_actions)
        
        self.model = self._build_model(scope)
        print('self.model: ', self.model)
        
        self.action_probs = tf.squeeze(tf.nn.softmax(self.model))
        self.picked_action_prob = tf.gather(self.action_probs, self.action_ph)
        print('self.picked_action_prob: ', self.picked_action_prob)
        self.picked_action_prob2 = tf.reduce_sum(tf.multiply(self.action_probs, self.action_oh), axis=1)
        print('self.picked_action_prob2: ', self.picked_action_prob2)
        
        self.loss = -tf.log(self.picked_action_prob) * self.target_ph
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
                
    def _build_model(self, scope):
        with tf.variable_scope(scope):
            dense1 = tf.layers.dense(self.state_oh, units=100, activation=tf.nn.relu)
            model = tf.layers.dense(dense1, units=self.n_actions, activation=None)
        return model
    
    def get_action(self, state, sess=None):
        sess = sess or tf.get_default_session()
#        print('state shape in get_action', np.shape([state]))
        action_probs = sess.run(self.action_probs, feed_dict={self.state_ph: [state]})
        try:
            action = np.random.choice(np.array(self.n_actions), p=action_probs)
        except:
            print("action_probs", action_probs)   
        return action
    
    def update(self, state, action, target, sess=None):
        sess = sess or tf.get_default_session()
        feed = {self.state_ph: [state], self.action_ph:[action], self.target_ph:[target]}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


#%% Critic
class ValuesEstimator():
    def __init__(self, learning_rate=0.01, scope='Values_Estimator'):
        self.n_states = env.observation_space.n
        
        self.stateV_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='StateV_PH')
        self.targetV_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='TargetV_PH')
        
        self.stateV_oh = tf.one_hot(self.stateV_ph, depth=self.n_states)
        
        self.modelV = self._build_model(scope)
        print('self.modelV: ', self.modelV)   
        
        self.values_estimate = tf.squeeze(tf.nn.softmax(self.modelV))
        
        self.lossV = tf.squared_difference(self.values_estimate, self.targetV_ph)
        
        self.optimizerV = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.trainV_op = self.optimizerV.minimize(self.lossV)
      
    def _build_model(self, scope):
        with tf.variable_scope(scope):
            dense1 = tf.layers.dense(self.stateV_oh, units=100, activation=tf.nn.relu)
            modelV = tf.layers.dense(dense1, units=1, activation=None)
        return modelV
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.values_estimate, feed_dict={self.stateV_ph: state})
    
    def get_advantage(self, state, total_return, sess=None):
        sess = sess or tf.get_default_session()
        baseline_value = sess.run(self.values_estimate, feed_dict={self.stateV_ph: [state]})
        advantage = total_return - baseline_value
        return advantage
    
    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed = {self.stateV_ph: [state], self.targetV_ph:[target]}
        _, lossV = sess.run([self.trainV_op, self.lossV], feed_dict=feed)
        return lossV
    
#%% reinforce
def generate_one_episode(env, policy):
    Transition = collections.namedtuple("Transition", ["state", "action", 
                                    "reward", "next_state", "done"])
    
    episode = []
    episode_reward = 0
    state = env.reset()
    for t in itertools.count():
#            print('state in generate_episodes', state)
        action = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        episode.append(Transition(state=state, action=action, 
                    reward=reward, next_state=next_state, done=done))
        print('episode in generate_one_episode: ', episode)
        
        episode_reward +=  reward
        state = next_state
        
        if done:
            break
        
    episode_length = t
        
    return episode, episode_length, episode_reward
        
        
def reinforce(evn, estimator_policy, estimator_value, n_episodes, gamma=1.0):
    n_episodes = 300
    gamma = 0.99
    
    for e in range(n_episodes):
        episode, episode_length, episode_reward = \
                    generate_one_episode(env, estimator_policy)
        print('episode: \n ', episode)
        print('episode_length: \n ', episode_length)
        print('episode_reward: \n ', episode_reward)
        
        for t, transition in enumerate(episode):
            print('transition: \n', transition)
            print('transition: \n', transition)
            print('transition.reward: \n', transition.reward)
            reward_to_go = sum(gamma**i * trans.reward for i, trans in enumerate(episode[t:]))
            print('reward_to_go: \n', reward_to_go)
            
            estimator_value.update(transition.state, reward_to_go)
            
            advantage = estimator_value.get_advantage(transition.state, reward_to_go)
            print('advantage: \n', advantage)
            estimator_policy.update(transition.state, transition.action, advantage)
    
#    return stats

    
#%% training
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(env)
value_estimator = ValuesEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = reinforce(env, policy_estimator, value_estimator, 2000, gamma=1.0)
   
#%% plot
