import torch
from torch import nn # making use of 
import torch.nn.functional as F # importing commonly used neural network functions (activation functions).
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""

		# NOTE Most of the neural network routines (like backward propogation with gradient descent)
		# will be handled by the pytorch library
		# So we don't need to worry about that 
		super(FeedForwardNN, self).__init__()
		
		self.input_layer = nn.Linear(in_dim, 256)
		
		#---------------TASK-------------------#
		# To make this into a deep neural network we would need to add atleast 2 more layers
		# Don't forget to fix the forward propagation function
		self.layer1 = nn.Linear(256, 256)
		self.layer2 = nn.Linear(256, 256)
 		
		


		# keep the input dimension of the layer at 256  
		self.output_layer = nn.Linear(256, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
        # tensor is a multi-dimensional matrix containing elements of a single data type
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)
        
		# Using relu activation function in between the layers 
		activation_input = F.relu(self.input_layer(obs))        
		
		
		#-------------------TASK----------------#
		# Implement deep layers in feed forward process
		# Use relu for all the layers
		activation1 = F.relu(self.layer1(activation_input))
		activation2 = F.relu(self.layer2(activation1))
		
		output = self.output_layer(activation2)
		
		return output