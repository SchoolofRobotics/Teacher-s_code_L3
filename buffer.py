import torch
import numpy as np
# This class will store our agent experiences
class Buffer:
    def __init__(self, gamma, batch_size):
        self.batch_size=batch_size
        self.gamma = gamma
        if torch.cuda.is_available():
            self.dev="cuda:0"
        else:
            self.dev="cpu"
        # buffer data for each agent.
        # We don't store the done signal, because we use 2d array to store rewards
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rtgs = []
        self.batch_lens = []
        self.batch_rews = []


    def append(self, obs, acts, log_probs, lens, rews):
        self.batch_obs.extend(obs)
        self.batch_acts.extend(acts)
        self.batch_log_probs.extend(log_probs)
        self.batch_lens.append(lens)
        self.batch_rews.append(rews)

    def toTensor(self):
        """
        Reshapes data to tensors
        Parameters:
            None
        Returns:
            None
        """
        self.batch_obs = torch.tensor(self.batch_obs, dtype=torch.float).to(torch.device(self.dev)) 
        self.batch_acts = torch.tensor(self.batch_acts, dtype=torch.float).to(torch.device(self.dev)) 
        self.batch_log_probs = torch.tensor(self.batch_log_probs, dtype=torch.float).to(torch.device(self.dev)) 



    def reset(self):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rtgs = []
        self.batch_lens = []
        self.batch_rews = []
    
    def length(self):
        return np.sum(self.batch_lens)


    def shuffle_batch(self, A_k):
        """
			Shuffles batches
			Parameters:
				A_k - Advantage at each timestep. Shape: (number of timesteps), Advantage is calculated afterwards we collect our buffer data, outside of buffer.
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: ( number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				A_k - Advantage at each timestep. Shape: (number of timesteps)
		
		"""
        idx = torch.randperm(self.batch_obs.shape[0])
        batch_obs = self.batch_obs[idx].view(self.batch_obs.size())
        batch_acts = self.batch_acts[idx].view(self.batch_acts.size())
        batch_log_probs = self.batch_log_probs[idx].view(self.batch_log_probs.size())
        batch_rtgs = self.batch_rtgs[idx].view(self.batch_rtgs.size())
        A_k = A_k[idx].view(A_k.size())
        return batch_obs[:self.batch_size], batch_acts[:self.batch_size], batch_log_probs[:self.batch_size], batch_rtgs[:self.batch_size], A_k[:self.batch_size]


    def compute_rtgs(self):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.
			Parameters:
				self - We take buffer's rewards to calculate Reward-To_Go for each episode.
			Return:
				None
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
        self.batch_rtgs =[]
		# Iterate through each episode
        for ep_rews in reversed(self.batch_rews):

            discounted_reward = 0 # The discounted reward so far

				# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
				# discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                self.batch_rtgs.insert(0, discounted_reward)
			# Convert the rewards-to-go into a tensor
        self.batch_rtgs = torch.tensor(self.batch_rtgs, dtype=torch.float).to(torch.device(self.dev)) 

