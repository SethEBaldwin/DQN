import gym
import tensorflow as tf
import numpy as np
import time
from collections import deque
from PIL import Image

sess = tf.Session()

#hyperparameters
ENV_NAME = "PongNoFrameskip-v4"
EPSILON_START = 1 #starting probability of random action
EPSILON_MIN = .05 #.1
EPSILON_DEC = .95/1e5 #.9/1e6 #amount to decrease epsilon each training step until arriving at EPSILON_MIN
FRAME_SKIP = 4 #make observation once every FRAME_SKIP frames (frames = FRAME_SKIP*steps)
N_ACTIONS = 3 #pong only needs actions 1,2,3 (have to add one when env.step)
LOAD = False #load model?
LOAD_PATH = "/dqn_saves/pong.ckpt" #loads from here if LOAD
MEMORY_CAPACITY = 10000 #10**6
TRAIN = True #train model?
SAVE = True #save model? only saved if TRAIN is also True
SAVE_AFTER = 10000 #saves after this many steps
SAVE_PATH = "/dqn_saves/pong.ckpt" #saves to here if TRAIN
TENSORBOARD = True #send data to tensorboard? only sends data if TRAIN is also True
TENSORBOARD_AFTER = 10000 #sends data to tensorboard after this many steps
TENSORBOARD_DIR = "/pong"
START_TRAIN = 9000 #50000 #start training after this many steps. Must be >= 5 for indexing reasons.
UPDATE_TARGET = 1000 #10000 #number of steps in between each target network update
DISCOUNT_RATE = .99
LEARNING_RATE = 1e-4 #1e-5
MB_SIZE = 32 #mini batch size
CROP_TOP = 16
CROP_BOTTOM = 34 #crop range is chosen for pong/breakout
N_SCORES = 100 #number of scores to save

#tensorboard
reward_tb = tf.placeholder(tf.float32, shape = None, name = "reward_tb")
reward_summary = tf.summary.scalar("Reward", reward_tb)
score_tb = tf.placeholder(tf.float32, shape = None, name = "score_tb")
score_summary = tf.summary.scalar("Score", score_tb)
Q_value_tb = tf.placeholder(tf.float32, shape = None, name = "Q_value_tb")
Q_value_summary = tf.summary.scalar("Q value", Q_value_tb)
loss_tb = tf.placeholder(tf.float32, shape = None, name = "loss_tb")
loss_summary = tf.summary.scalar("Loss", loss_tb)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)

def get_target(Q_value, reward, done):
	if done: return reward
	else: return reward + DISCOUNT_RATE*Q_value

class DeepQNetwork:

	def __init__(self,name, learning_rate, n_actions):
		self.name = name
		self.n_actions = n_actions
		with tf.variable_scope(self.name):
			self.input = tf.placeholder(tf.float32, shape =[None,80,80,4])
			self.conv = tf.layers.conv2d(self.input, filters=32, kernel_size=8, strides=(4,4), 
				activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.conv2 = tf.layers.conv2d(self.conv, 64, 4, (2,2), activation = tf.nn.relu, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.flat = tf.layers.flatten(self.conv2)
			self.dense = tf.layers.dense(self.flat, 512, activation = tf.nn.relu, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.output = tf.layers.dense(self.dense, self.n_actions, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.targets = tf.placeholder(tf.float32, shape =[None])
			self.actions = tf.placeholder(tf.float32, shape =[None])
			self.loss = tf.losses.huber_loss(self.targets,
				tf.reduce_sum(self.output*tf.one_hot(tf.cast(self.actions,tf.int32), self.n_actions),axis=1))
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
		sess.run(tf.global_variables_initializer())
		
	def greedy_action(self, input):
		output = sess.run(self.output,feed_dict={self.input: input})[0]
		return (np.argmax(output),max(output)) #action, Q_value

	def get_action(self, input, epsilon):
		greedy = np.random.random() > epsilon
		if greedy: 
			action, Q_value = self.greedy_action(input)
		else: 
			action = np.random.randint(0, self.n_actions)
			Q_value = 0 #will be ignored since not greedy
		return (action, Q_value, greedy)
		
	def feed_forward(self,input):
		return sess.run(self.output, feed_dict={self.input: input})
		
	def get_targets(self,x_batch2,rewards_batch,dones_batch):
		output = self.feed_forward(x_batch2)
		targets = [get_target(max(Q),reward,done) for (Q,reward,done) in zip(output, rewards_batch, dones_batch)]
		return targets
		
	def train_on_batch(self, input_batch, targets, actions_batch):
		loss, _ = sess.run([self.loss, self.optimizer],
			feed_dict={self.input: input_batch, self.targets:targets, self.actions: actions_batch})	
		return loss
		
dqn = DeepQNetwork("dqn", LEARNING_RATE, N_ACTIONS)
dqn_target = DeepQNetwork("dqn_target", LEARNING_RATE, N_ACTIONS)
saver = tf.train.Saver()

def copy_weights(from_name, to_name):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_name)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_name)
	op_holder = []
	for from_var,to_var in zip(from_vars, to_vars):
		op_holder.append(to_var.assign(from_var))
	sess.run(op_holder)

def compress(observation):
	img = Image.fromarray(observation[CROP_BOTTOM : -CROP_TOP, : , : ])
	img = img.convert("L")
	img = img.resize((80,80), Image.NEAREST)
	return np.reshape(np.array(img),[1, 80, 80, 1]) 
	#don't divide by 255 yet as this changes from uint8 to float32 which takes up 4 times as much memory
	
#returns the last m elements of a deque. if m < length returns the whole deque.
def tail(xs, m): 
	length = len(xs)
	n = min(length, m)
	return [ xs[i] for i in range(length-n, length) ]
	
class GameState:

	def __init__(self, env_name):
		self.env = gym.make(env_name)
		observation = compress(self.env.reset())
		self.observations_4 = np.concatenate([observation]*4,-1)
		self.action = 0
		self.reward = 0
		self.done = False
		self.steps = 0
		self.score = 0
	
	def reset(self):
		observation = compress(self.env.reset())
		self.observations_4 = np.concatenate([observation]*4,-1)
		self.action = 0
		self.reward = 0
		self.done = False
		self.steps = 0
		self.score = 0
		self.env.render()
		
	#perform self.action n_frames times, update self.reward, self.score, self.done, and self.observations_4
	def perform_action(self, n_frames):
		reward = 0
		for _ in range(n_frames):
			raw_observation, new_reward, self.done, _ = self.env.step(self.action+1)
			reward += new_reward
			if self.done: break
		self.reward = reward
		self.score += reward
		self.observations_4 = np.concatenate([self.observations_4[:,:,:,1:4],compress(raw_observation)],-1)
		self.env.render()

state = GameState(ENV_NAME)

class ReplayMemory:

	def __init__(self, capacity):
		self.observations = deque(maxlen = capacity)
		self.actions = deque(maxlen = capacity)
		self.rewards = deque(maxlen = capacity)
		self.dones = deque(maxlen = capacity)
		
	def append(self, observation, action, reward, done):
		self.observations.append(observation)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
		
	def stack_observations(self, n, m):
		observations = [self.observations[i] for i in range(n,m)]
		return np.concatenate(observations,-1)
		
	def get_batch(self, batch_size):
		batch_idxs = [ np.random.randint(3,len(self.actions)-1) for _ in range(batch_size) ]
		actions_batch = [ self.actions[idx] for idx in batch_idxs ]
		rewards_batch = [ self.rewards[idx] for idx in batch_idxs ]
		dones_batch = [ self.dones[idx] for idx in batch_idxs ]
		x_batch = np.concatenate([ self.stack_observations(idx-3,idx+1) for idx in batch_idxs ],0)
		x_batch2 = np.concatenate([ self.stack_observations(idx-2,idx+2) for idx in batch_idxs ],0)
		return (actions_batch, rewards_batch, dones_batch, x_batch, x_batch2)

memory = ReplayMemory(MEMORY_CAPACITY)

class EvaluationData:

	def __init__(self):
		self.Q_values = deque(maxlen = TENSORBOARD_AFTER)
		self.losses = deque(maxlen = TENSORBOARD_AFTER)
		self.scores = deque(maxlen = N_SCORES)
		self.max_score = -21.0
		self.steps_total = 0
		self.games_complete = 0
		self.time_start = time.clock()
		self.epsilon = EPSILON_START

	def game_complete(self, steps, score):
		self.games_complete += 1
		self.scores.append(score)
		if score > self.max_score: self.max_score = score
		print("Episode {} ended after {} frames".format(self.games_complete, steps*FRAME_SKIP))
		print("Score: {}".format(score))
		
	def print_data(self,recent_rewards):
		print("Best score: {}".format(self.max_score))
		print("Reward average over last {} rewards: {}".format(len(recent_rewards), round(np.mean(recent_rewards), 3)))
		print("Score average over last {} games: {}".format(len(self.scores),round(np.mean(self.scores),1)))
		if len(self.Q_values) > 0:
			print("Predicted max Q average over last {} greedy steps: {:.2f}".format(len(self.Q_values), np.mean(self.Q_values)))
		if len(self.losses) > 0:
			print("Loss average over last {} training steps: {:.5f}".format(len(self.losses), np.mean(self.losses)))
		print("Probability of random action: {}".format(round(self.epsilon,2)))
		print("Total training steps: {}".format(self.steps_total))
		time_elapsed = time.clock() - self.time_start
		print("Total training hours: {}".format(round(time_elapsed/(60*60), 1)))
		print("Average training steps per hour: {}".format(int(round(60*60*self.steps_total/time_elapsed))))
		
data = EvaluationData()

#start program
if LOAD:
	saver.restore(sess, LOAD_PATH)
	print("\nModel restored")
while not (len(data.scores) >= N_SCORES and np.mean(data.scores) >= 18):
	#initialize episode
	print("\nStarting episode {}...".format(data.games_complete+1))
	state.reset()
	#game loop
	while not state.done:
		#get action
		state.action, Q_value, greedy = dqn.get_action(state.observations_4/255, data.epsilon)
		if greedy: data.Q_values.append(Q_value)
		#get reward, done, new observation
		state.perform_action(FRAME_SKIP) #performs state.action, updates state.observations_4,state.reward,state.done
		memory.append(state.observations_4[:,:,:,2:3], state.action, state.reward, state.done) #old observation
		#train
		if TRAIN and data.steps_total >= START_TRAIN and data.steps_total >= 5:
			actions_batch, rewards_batch, dones_batch, x_batch, x_batch2 = memory.get_batch(MB_SIZE)
			targets = dqn_target.get_targets(x_batch2/255, rewards_batch, dones_batch)
			loss = dqn.train_on_batch(x_batch/255, targets, actions_batch)
			data.losses.append(loss)
		#step complete
		state.steps += 1
		data.steps_total += 1
		if data.epsilon > EPSILON_MIN: data.epsilon -= EPSILON_DEC
		#update target network
		if TRAIN and data.steps_total % UPDATE_TARGET == 0: 
			copy_weights(dqn.name, dqn_target.name)
			print("Target network updated")
		#save
		if TRAIN and SAVE and data.steps_total % SAVE_AFTER == 0:
			saver.save(sess, SAVE_PATH)
			print("Model saved in path: %s" % SAVE_PATH)
		#tensorboard
		if TRAIN and TENSORBOARD and data.steps_total % TENSORBOARD_AFTER == 0:
			#losses will be empty if TENSORBOARD_AFTER <= START_TRAIN
			#scores may be empty if TENSORBOARD_AFTER is smaller than the number of steps in the first episode
			recent_rewards = tail(memory.rewards,TENSORBOARD_AFTER)
			summary = sess.run(merged, feed_dict={
				reward_tb: np.mean(recent_rewards), score_tb: np.mean(data.scores), 
				Q_value_tb: np.mean(data.Q_values), loss_tb: np.mean(data.losses)})
			train_writer.add_summary(summary, data.steps_total)
			print("Training data sent to tensorboard")
	#game complete
	data.game_complete(state.steps, state.score) #updates data and prints results of game
	recent_rewards = tail(memory.rewards, TENSORBOARD_AFTER)
	data.print_data(recent_rewards)
#stop condition met
if TRAIN and SAVE: 
	print("\nTraining complete")
	saver.save(sess, SAVE_PATH)
	print("Model saved in path: %s" % SAVE_PATH)
print("\nProgram complete")
state.env.close()
