import gym
import tensorflow as tf
import numpy as np
import time
from collections import deque
from PIL import Image

env = gym.make('PongNoFrameskip-v4')
sess = tf.Session()

EPSILON = 1 #probability of random action
EPSILON_MIN = .05 #.1
EPSILON_DEC = .95/1e5 #.9/1e6 #amount to decrease EPSILON each training step until arriving at EPSILON_MIN
FRAME_SKIP = 4 #make observation once every FRAME_SKIP frames (frames = FRAME_SKIP*steps)
N_ACTIONS = 3 #pong only needs actions 1,2,3 (have to add one when env.step)
LOAD = False #load model?
LOAD_PATH = "/dqn_saves/pong.ckpt" #loads from here if LOAD
MEMORY_CAPACITY = 10000 #10**6
TRAIN = True #train model?
SAVE = True #save model? only saved if TRAIN is also True
SAVE_AFTER = 10000 #saves after this many steps
SAVE_PATH = "/dqn_saves/pong.ckpt" #saves to here if TRAIN
TENSORBOARD = True #send data to tensorboard?  only sends data if TRAIN is also True
TENSORBOARD_AFTER = 10000 #sends data to tensorboard after this many steps
TENSORBOARD_DIR = '/pong'
START_TRAIN = 10000 #50000 #start training after this many steps. Must be >= 5 for indexing reasons.
UPDATE_TARGET = 1000 #10000 #number of steps in between each target network update
DISCOUNT_RATE = .99
LEARNING_RATE = 1e-4 #1e-5
MB_SIZE = 32 #mini batch size

#tensorboard
REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
SCORE_PH = tf.placeholder(tf.float32, shape=None, name='score_summary')
SCORE_SUMMARY = tf.summary.scalar('score', SCORE_PH)
Q_PH = tf.placeholder(tf.float32, shape=None, name='Q_summary')
Q_SUMMARY = tf.summary.scalar('Q', Q_PH)
LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)

class deep_q_network:

	def __init__(self,name):
		with tf.variable_scope(name):
			self.input = tf.placeholder(tf.float32, shape =[None,80,80,4])
			self.conv = tf.layers.conv2d(self.input, filters=32, kernel_size=8, strides=(4,4), 
				activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.conv2 = tf.layers.conv2d(self.conv, 64, 4, (2,2), activation = tf.nn.relu, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.flat = tf.layers.flatten(self.conv2)
			self.dense = tf.layers.dense(self.flat, 512, activation = tf.nn.relu, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.output = tf.layers.dense(self.dense, N_ACTIONS, 
				kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
			self.targets = tf.placeholder(tf.float32, shape =[None])
			self.actions = tf.placeholder(tf.float32, shape =[None])
			self.loss = tf.losses.huber_loss(self.targets,
				tf.reduce_sum(self.output*tf.one_hot(tf.cast(self.actions,tf.int32), N_ACTIONS),axis=1))
			self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)
		sess.run(tf.global_variables_initializer())
		
	def greedy_action(self,x_feed):
		output = sess.run(self.output,feed_dict={self.input: x_feed})[0]
		return (np.argmax(output),max(output)) #action, Q_value

	def feed_forward(self,x_batch):
		return sess.run(self.output, feed_dict={self.input: x_batch})
		
	def train_on_batch(self,x_batch,targets,actions_batch):
		loss, _ = sess.run([self.loss,self.optimizer],
			feed_dict={self.input: x_batch, self.targets:targets, self.actions: actions_batch})	
		return loss
		
DQN = deep_q_network(name="DQN")
DQN_target = deep_q_network(name="DQN_target")
saver = tf.train.Saver()

def update_target_network():
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN_target")
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

def compress(observation):
	img = Image.fromarray(observation[34:-16,:,:]) #crop range is chosen for pong
	img = img.convert('L')
	img = img.resize((80,80),Image.NEAREST)
	return np.reshape(np.array(img),[1,80,80,1]) 
	#don't divide by 255 yet as this changes from uint8 to float32 which takes up 4 times as much memory
	
def get_target(Q_value,reward,done):
	if done: return reward
	else: return reward + DISCOUNT_RATE*Q_value
	
def tail(xs,m): #returns the last m elements of a deque.  if m < length returns the whole deque.
	length = len(xs)
	n = min(length,m)
	return [ xs[i] for i in range(length-n,length) ]

class replay_memory:

	def __init__(self,capacity):
		self.observations = deque(maxlen=capacity)
		self.actions = deque(maxlen=capacity)
		self.rewards = deque(maxlen=capacity)
		self.dones = deque(maxlen=capacity)
		self.capacity = capacity
		
	def append(self,observation,action,reward,done):
		self.observations.append(observation)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
		
	def get_observations_4(self,index):
		obv3 = self.observations[index-3]
		obv2 = self.observations[index-2]
		obv1 = self.observations[index-1]
		obv0 = self.observations[index]
		return np.concatenate([obv3,obv2,obv1,obv0],-1)

memory = replay_memory(MEMORY_CAPACITY)
		
class game_state:

	def __init__(self):
		raw_observation = env.reset()
		self.observation = compress(raw_observation)
		self.action = 0
		self.reward = 0
		self.done = False
		self.steps = 0
		self.score = 0
	
	def reset(self):
		raw_observation = env.reset()
		self.observation = compress(raw_observation)
		self.action = 0
		self.reward = 0
		self.done = False
		self.steps = 0
		self.score = 0

state = game_state()

Q_values = deque(maxlen=TENSORBOARD_AFTER)
losses = deque(maxlen=TENSORBOARD_AFTER)
scores = deque(maxlen=100)
max_score = -21.0
steps_total = 0
games_complete = 0
x_feed = np.concatenate([state.observation]*4,-1)

#start program
if LOAD:
	saver.restore(sess, LOAD_PATH)
	print("\nModel restored")
time_start = time.clock()
while not (len(scores) >= 100 and np.mean(scores) >= 18):
	#initialize episode
	print("\nStarting episode {}...".format(games_complete+1))
	state.reset()
	env.reset()
	env.render()
	#game loop
	while not state.done:
		#get action
		x_feed = np.concatenate([x_feed[:,:,:,1:4],state.observation],-1)
		if np.random.random() < EPSILON: 
			state.action = np.random.randint(0,N_ACTIONS)
		else: 
			state.action, Q_value = DQN.greedy_action(x_feed/255)
			Q_values.append(Q_value)
		#get reward, done, new observation
		state.reward = 0
		for _ in range(FRAME_SKIP):
			raw_observation, new_reward, state.done, _ = env.step(state.action+1)
			state.reward += new_reward
			if state.done: break
		env.render()
		memory.append(state.observation,state.action,state.reward,state.done) #old observation
		state.observation = compress(raw_observation) #new observation
		state.score += state.reward
		#train
		if TRAIN and steps_total >= START_TRAIN:
			batch_idxs = [ np.random.randint(3,len(memory.actions)-1) for _ in range(MB_SIZE) ]
			actions_batch = [ memory.actions[idx] for idx in batch_idxs ]
			rewards_batch = [ memory.rewards[idx] for idx in batch_idxs ]
			dones_batch = [ memory.dones[idx] for idx in batch_idxs ]
			x_batch = np.concatenate([ memory.get_observations_4(idx) for idx in batch_idxs ],0)
			x_batch2 = np.concatenate([ memory.get_observations_4(idx+1) for idx in batch_idxs ],0)
			next_Qs = DQN_target.feed_forward(x_batch2/255)
			targets = [ get_target(max(next_Qs[i]),rewards_batch[i],dones_batch[i]) for i in range(MB_SIZE) ]
			loss = DQN.train_on_batch(x_batch/255,targets,actions_batch)
			losses.append(loss)
		#step complete
		if EPSILON > EPSILON_MIN: EPSILON -= EPSILON_DEC
		state.steps += 1
		steps_total += 1
		#update target network
		if TRAIN and steps_total % UPDATE_TARGET == 0: 
			print("Checkpoint: total training steps: {}".format(steps_total))
			sess.run(update_target_network())
			print("Target network updated")
		#save
		if TRAIN and SAVE and steps_total % SAVE_AFTER == 0:
			saver.save(sess, SAVE_PATH)
			print("Model saved in path: %s" % SAVE_PATH)
		#tensorboard
		if TRAIN and TENSORBOARD and steps_total % TENSORBOARD_AFTER == 0:
			recent_rewards = tail(memory.rewards,TENSORBOARD_AFTER)
			#losses will be empty if TENSORBOARD_AFTER <= START_TRAIN
			#scores may be empty if TENSORBOARD_AFTER is small enough that a game isn't complete
			summary = sess.run(merged,feed_dict={
				REWARD_PH: np.mean(recent_rewards), SCORE_PH: np.mean(scores), 
				Q_PH: np.mean(Q_values), LOSS_PH: np.mean(losses)})
			train_writer.add_summary(summary, steps_total)
			print("Training data sent to tensorboard")
	#game complete
	games_complete += 1
	scores.append(state.score)
	if state.score > max_score: max_score = state.score
	#print data
	print("Episode {} ended after {} frames".format(games_complete,state.steps*FRAME_SKIP))
	print("Score: {}".format(state.score))
	print("Best score: {}".format(max_score))
	recent_rewards = tail(memory.rewards,TENSORBOARD_AFTER)
	print("Reward average over last {} rewards: {}".format(len(recent_rewards),round(np.mean(recent_rewards),3)))
	print("Score average over last {} games: {}".format(len(scores),round(np.mean(scores),1)))
	if len(Q_values) > 0:
		print("Predicted max Q average over last {} greedy steps: {:.2f}".format(len(Q_values),np.mean(Q_values)))
	if len(losses) > 0:
		print("Loss average over last {} training steps: {:.5f}".format(len(losses),np.mean(losses)))
	time_elapsed = time.clock() - time_start
	print("Probability of random action: {}".format(round(EPSILON,2)))
	print("Total training steps: {}".format(steps_total))
	print("Total training hours: {}".format(round(time_elapsed/(60*60),1)))
	print("Average training steps per hour: {}".format(int(round(60*60*steps_total/time_elapsed))))
#stop condition met
if TRAIN and SAVE: 
	print("\nTraining complete")
	saver.save(sess, SAVE_PATH)
	print("Model saved in path: %s" % SAVE_PATH)
print("\nProgram complete")
env.close()
