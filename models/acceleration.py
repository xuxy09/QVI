# This file implements acceleration/velocity calculation

import torch
import torch.nn as nn


class AcFusionLayer(nn.Module):
	"""docstring for AcFusionLayer"""
	def __init__(self, ):
		super(AcFusionLayer, self).__init__()
	
	def forward(self, flo10, flo12, flo21, flo23, t=0.5):
		"""
			-- input: four flows
			-- output: center shift
		"""

		return 0.5 * ((t + t**2)*flo12 - (t - t**2)*flo10), 0.5 * (((1 - t) + (1 - t)**2)*flo21 - ((1 - t) - (1 - t)**2)*flo23)
		
