import time

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from mlagents_envs.base_env import ActionTuple
import collections

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.
			Parameters:
				policy_class - the policy neural network to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - hyperparameters containing all extra arguments passed into PPO that should be hyperparameters.
			Returns:
				None
		"""
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		
		# Extract environment information
		env.reset()
		self.env = env
		self.behavior_name = list(env.behavior_specs)[0]
		env_specs = env.behavior_specs[self.behavior_name]
		# Calculate total observations
		total_observations=0
		for index, obs in enumerate(env_specs.observation_specs):
			total_obs=1
			# Enumerate for matrices
			for index, shape in enumerate(obs.shape):
				total_obs = total_obs * shape
			total_observations = total_observations + total_obs
		self.discrete_branches = env_specs.action_spec.discrete_branches
		# Here we specify dimensions input and output dimensions of our neural network
		self.total_branches= env_specs.action_spec.discrete_size
		self.discrete_size =0
		for action, branch_size in enumerate(self.discrete_branches):
			self.discrete_size += branch_size
		self.continuous_size = env_specs.action_spec.continuous_size
		self.obs_dim = total_observations
		self.act_dim = self.continuous_size +  self.discrete_size

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		# You can learn more about pytorch optimizers here: https://pytorch.org/docs/stable/optim.html
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr= self.actor_lr)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr= self.critic_lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': [],     # losses of critic network in current iteration
		}

	def load_networks(self, act_path, cri_path):
		self.actor.load_state_dict(torch.load(act_path))
		self.critic.load_state_dict(torch.load(cri_path))
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr= self.actor_lr)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr= self.critic_lr)


	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = self.rollout()                     # ALG STEP 3
			

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)
			# Increment the number of iterations
			i_so_far += 1
			self.logger['batch_rews'] = batch_rews
			self.logger['batch_lens'] = batch_lens
			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far	
			
			V, _, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			#A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				
				# Shuffle the batch (aka randomize data)
				batch_obs, batch_acts, batch_log_probs, batch_rtgs, A_k = self.shuffle_batch(batch_obs, batch_acts, batch_log_probs, batch_rtgs, A_k)
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
				
				# Calculate the ratio pi(a_t | s_t) / pi_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				
				# Calculate gradients and perform backward propagation for actor and critic networks
				# The backward propagation is already built-in to the nn from torch library, so we don't need to worry about that

				actor_loss = -torch.min(surr1, surr2) 
				critic_loss = nn.MSELoss()(V, batch_rtgs)
				#----------------------------------Task-----------------------------------#
				# Write total loss function and make backward propagation on the actor.
				total_loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy



				self.actor_optim.zero_grad()
				total_loss.mean().backward(retain_graph=True)
				self.actor_optim.step()


				# Here we apply backward propagation on the critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['critic_losses'].append(critic_loss.detach())
				# Print a summary of our training so far
				


				# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				print("saving model...")
				torch.save(self.actor.state_dict(), f'results/ppo_actor.pth')
				torch.save(self.critic.state_dict(), f'results/ppo_critic.pth')
				print("model saved")
			
			self._log_summary()

	
	def just_roll(self):
		"""
			To just keep running the agents in the environment.
			Parameters:
				None
			Return:
				None
		"""
		self.env.reset()
		decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
		total_agents = len(decision_steps)
		while True:
			
			
			
			# Calculate action for each agent.
			continouos_actions=np.empty((len(decision_steps), self.continuous_size))
			discrete_actions=np.empty((len(decision_steps), self.total_branches))
			#------------------Task---------------------#
			# finish agent inference mode
			for agent_id in decision_steps:
				
				observations=[]
				for observation in decision_steps[agent_id].obs:
					observations.append(observation.flatten())
				observations = np.concatenate(observations)

				observations = torch.tensor(observations, dtype=torch.float)
				_, c_actions, d_actions, _ = self.get_actions(observations)
				continouos_actions[agent_id] = c_actions
				discrete_actions[agent_id] = d_actions




			# Take the actions in the environement
			unity_actions = ActionTuple(continouos_actions, discrete_actions)
			self.env.set_actions(self.behavior_name, unity_actions)
			# Step environment
			self.env.step()
			# If all the agents are terminated, reset
			if len(terminal_steps) == total_agents:
				self.env.reset()
			decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)






	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.
			Parameters:
				None
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: ( number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: ( number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data for each agent. For more details, check function header.

		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rtgs = []
		batch_lens = []
		batch_rews = []
		
		self.env.reset()
		decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
		total_agents=len(decision_steps)		
		t = 0# Keeps track of how many timesteps we've run so far this batch
		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			# Episodic data. Keeps track of obs, actions and rewards per episode, will get cleared
			# upon each new episode
			temp_batch_obs = []
			temp_batch_acts =[]
			temp_batch_log_probs =[]
			temp_batch_lens = [0] *total_agents
			ep_rews = [] # rewards collected per episode
			done = [False]*total_agents # to track agent termination better
			# To track whether observation was made before reading the reward from the environment
			observed = [False]*total_agents
			# nesting 2d arrays for each agent
			for i in range(total_agents):
				ep_rews.append([])
				temp_batch_obs.append([])
				temp_batch_acts.append([])
				temp_batch_log_probs.append([])
			
			
			self.env.reset()
			# Reset the environment. sNote that obs is short for observation. 
			decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			while any(ep_t < self.max_timesteps_per_episode for ep_t in temp_batch_lens) and t < self.timesteps_per_batch:
				
				
				for agent_id in decision_steps:# Track observations for each agent
					done[agent_id] = False
					t +=1
					temp_batch_lens[agent_id] += 1
					observations=[]
					# Flatten all observations into a 1D array
					for observation in decision_steps[agent_id].obs:
						observations.append(observation.flatten())
					observations = np.concatenate(observations)
					temp_batch_obs[agent_id].append(observations)
					observed[agent_id]= True


				# Calculate action for each agent.
				continouos_actions=np.empty((len(decision_steps.agent_id), self.continuous_size))
				discrete_actions=np.empty((len(decision_steps.agent_id), self.total_branches))
				for agent_id in decision_steps:
					# Convert observations into a tensor
					observations = torch.tensor(temp_batch_obs[agent_id][-1], dtype=torch.float)
					actions, c_actions, d_actions, log_prob = self.get_actions(observations)
					continouos_actions[agent_id] = c_actions
					discrete_actions[agent_id] = d_actions
					# Track recent actions, and action log probabilities
					temp_batch_log_probs[agent_id].append(log_prob)
					temp_batch_acts[agent_id].append(actions)

				
				
				# Take the actions in the environement
				unity_actions = ActionTuple(continouos_actions, discrete_actions)

				self.env.set_actions(self.behavior_name, unity_actions)
				# Step environment
				self.env.step()
				# Get new observations and rewards
				decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
				for agent_id in decision_steps:
					
					# Note that rew is short for reward.
					# Track recent reward
					rew = decision_steps[agent_id].reward
					if observed[agent_id]:
						ep_rews[agent_id].append(rew)
						observed[agent_id] = False
					if temp_batch_lens[agent_id] > self.max_timesteps_per_episode:	
						batch_lens.append(temp_batch_lens[agent_id])
						batch_rews.append(ep_rews[agent_id])
						batch_obs.extend(temp_batch_obs[agent_id])
						batch_log_probs.extend(temp_batch_log_probs[agent_id])
						batch_acts.extend(temp_batch_acts[agent_id])
						temp_batch_lens[agent_id] = 0
						temp_batch_obs[agent_id] = []
						temp_batch_acts[agent_id] = []
						temp_batch_log_probs[agent_id] = []
						ep_rews[agent_id] = []
						done[agent_id] = True
					

				# Track episodic lenghts and rewards
				for agent_id in terminal_steps:
					if observed[agent_id]:
						rew = terminal_steps[agent_id].reward
						ep_rews[agent_id].append(rew)
						observed[agent_id] = False
					
					
					batch_lens.append(temp_batch_lens[agent_id])
					batch_rews.append(ep_rews[agent_id])
					batch_obs.extend(temp_batch_obs[agent_id])
					batch_log_probs.extend(temp_batch_log_probs[agent_id])
					batch_acts.extend(temp_batch_acts[agent_id])
					temp_batch_lens[agent_id] = 0
					temp_batch_obs[agent_id] = []
					temp_batch_acts[agent_id] = []
					temp_batch_log_probs[agent_id] = []
					ep_rews[agent_id] = []
					done[agent_id] = True				
				# If all the agents are terminated, break
				if len(terminal_steps) >= total_agents:
					break
			# Track episodic lenghts and rewards
			for agent_id in range(total_agents):
				# if got observations and took actions, but received no reward
				if observed[agent_id]:
					t -= 1
					temp_batch_obs[agent_id].pop()
					temp_batch_acts[agent_id].pop()
					temp_batch_log_probs[agent_id].pop()
				if not done[agent_id]:
					batch_lens.append(temp_batch_lens[agent_id])
					batch_rews.append(ep_rews[agent_id])
					batch_obs.extend(temp_batch_obs[agent_id])
					batch_log_probs.extend(temp_batch_log_probs[agent_id])
					batch_acts.extend(temp_batch_acts[agent_id])
					temp_batch_obs[agent_id] = []
					temp_batch_acts[agent_id] = []
					temp_batch_log_probs[agent_id] = []
					ep_rews[agent_id] = []
					done[agent_id] = True
		
			

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		# compute reward to go, for each agent   
		batch_rtgs = self.compute_rtgs(batch_rews)
		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs =[]
		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

				# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
				# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)
			# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		return batch_rtgs

	def shuffle_batch(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, A):
		"""
			Shuffles batches
			Parameters:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: ( number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				A - Advantage of each observation. Shape: (number of timesteps)
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: ( number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				A - Advantage of each observation. Shape: (number of timesteps)
		
		"""
		
		idx = torch.randperm(batch_obs.shape[0])
		batch_obs = batch_obs[idx].view(batch_obs.size())
		batch_acts = batch_acts[idx].view(batch_acts.size())
		batch_log_probs = batch_log_probs[idx].view(batch_log_probs.size())
		batch_rtgs = batch_rtgs[idx].view(batch_rtgs.size())
		A = A[idx].view(A.size())

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, A









	def get_actions(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.
			Parameters:
				obs - the observation at the current timestep
			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for actions
		actions = self.actor(obs)
		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		
		dist = MultivariateNormal(actions, self.cov_mat, validate_args=True)
		# Sample distribution and get the actions
		sampled_actions = dist.sample()
		discrete_actions =[]
		branch_enumerator=0
		# Choose discrete actions from the distribution
		for _, branch_size in enumerate(self.discrete_branches):
			
			current_pos =self.continuous_size+branch_enumerator
			d_action =  torch.argmax(torch.sigmoid(sampled_actions[current_pos:current_pos+branch_size]))
			discrete_actions.append(d_action)
			branch_enumerator += branch_size
			


		
		# Calculate the log probability for actions
		log_probs = dist.log_prob(sampled_actions)
		
		c_actions = sampled_actions[:self.continuous_size]
		# Return the sampled actions and the log probabilities of those actions in our distributions
		return sampled_actions.detach().numpy(), c_actions.detach().numpy(), np.array(discrete_actions).flatten(), log_probs.detach() 

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
				entropy - the entropy of the distribution
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		actions = self.actor(batch_obs)
		dist = MultivariateNormal(actions, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)
		
		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs, dist.entropy()

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters
			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.
			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters						
		self.timesteps_per_batch = 2048               # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 256            # Max number of timesteps per episode
		self.n_updates_per_iteration = 3                # Number of times to update actor/critic per iteration
		self.actor_lr = 0.0003                          # Learning rate of optimizers
		self.critic_lr= 0.0003
		self.gamma = 0.995                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.critic_discount = 0.5
		self.entropy_beta = 0.005
		# Miscellaneous parameters
		self.save_freq = 25                            # How often we save in number of iterations
		self.seed = None
		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.
			Parameters:
				None
			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])
		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))
		avg_critic_loss = str(round(avg_critic_loss, 5))


		# Write logging statements to file
		with open("logs.txt", 'a') as logs:		
			print(flush=True, file=logs)
			print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True, file=logs)
			print(f"Average Episodic Length: {avg_ep_lens}", flush=True, file=logs)
			print(f"Average Episodic Return: {avg_ep_rews}", flush=True, file=logs)
			print(f"Average Actor Loss: {avg_actor_loss}", flush=True, file=logs)
			print(f"Average Critic Loss: {avg_critic_loss}", flush=True, file=logs)
			print(f"Timesteps So Far: {t_so_far}", flush=True, file=logs)
			print(f"Iteration took: {delta_t} secs", flush=True, file=logs)
			print(f"------------------------------------------------------", flush=True, file=logs)
			print(flush=True, file=logs)


		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
		print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []