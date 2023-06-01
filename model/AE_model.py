# -*- coding: utf-8 -*-
"""
Created on Thu Ocr  6 10:17:15 2022

@author: chunman zuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import collections
import time
import math
import random

from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.modules.module import Module
from collections import OrderedDict
from torch.distributions import Normal, kl_divergence as kl


class AE(Module):
	def __init__( self, layer_e, layer_d, hidden1, args, droprate = 0.1, type = "NB" ):
		super(AE, self).__init__()
		
		### function definition
		self.encoder     = build_multi_layers( layer_e )

		if type == "NB":
			self.decoder = Decoder_logNorm_NB( layer_d, hidden1, layer_e[0], droprate = droprate )

		else: #Gaussian
			self.decoder = Decoder( layer_d, hidden1, layer_e[0], Type = type, droprate = droprate)

		self.args      = args
		self.type      = type
	
	def inference(self, X = None, scale_factor = 1.0):
		
		latent = self.encoder( X )
		
		### decoder
		if self.type == "NB":
			output        =  self.decoder( latent, scale_factor )
			norm_x        =  output["normalized"]
			disper_x      =  output["disperation"]
			recon_x       =  output["scale_x"]

		else:
			recons_x      =  self.decoder( latent )
			recon_x       =  recons_x
			norm_x        =  None
			disper_x      =  None

		return dict( norm_x   = norm_x, disper_x   = disper_x, 
					 recon_x  = recon_x, latent_z  = latent )


	def return_loss(self, X = None, X_raw = None, scale_factor = 1.0 ):

		output           =  self.inference( X, scale_factor )
		recon_x          =  output["recon_x"]
		disper_x         =  output["disper_x"]

		if self.type == "NB":
			loss         =  log_nb_positive( X_raw, recon_x, disper_x )

		else:
			loss = mse_loss( X, recon_x )

		return loss

		
	def forward( self, X = None, scale_factor = 1.0 ):

		output =  self.inference( X, scale_factor )

		return output


	def predict(self, dataloader, out='z' ):
		
		output = []

		for batch_idx, ( X, size_factor ) in enumerate(dataloader):

			if self.args.use_cuda:
				X, size_factor = X.cuda(), size_factor.cuda()

			X           = Variable( X )
			size_factor = Variable(size_factor)

			result      = self.inference( X, size_factor)

			if out == 'z': 
				output.append( result["latent_z"].detach().cpu() )

			elif out == 'recon_x':
				output.append( result["recon_x"].detach().cpu().data )

			else:
				output.append( result["norm_x"].detach().cpu().data )

		output = torch.cat(output).numpy()
		return output


	def fit( self, train_loader, test_loader ):

		params    = filter(lambda p: p.requires_grad, self.parameters())
		optimizer = optim.Adam( params, lr = self.args.lr_AET, weight_decay = self.args.weight_decay, eps = self.args.eps )

		train_loss_list   = []
		reco_epoch_test   = 0
		test_like_max     = 1000000000
		flag_break        = 0

		patience_epoch         = 0
		self.args.anneal_epoch = 10

		start = time.time()

		for epoch in range( 1, self.args.max_epoch_T + 1 ):

			self.train()
			optimizer.zero_grad()

			patience_epoch += 1

			kl_weight      =  min( 1, epoch / self.args.anneal_epoch )
			epoch_lr       =  adjust_learning_rate( self.args.lr_AET, optimizer, epoch, self.args.lr_AET_F, 10 )

			for batch_idx, ( X, X_raw, size_factor ) in enumerate(train_loader):

				if self.args.use_cuda:
					X, X_raw, size_factor = X.cuda(), X_raw.cuda(), size_factor.cuda()
				
				X, X_raw, size_factor     = Variable( X ), Variable( X_raw ), Variable( size_factor )
				loss1  = self.return_loss( X, X_raw, size_factor )
				loss   = torch.mean( loss1  )

				loss.backward()
				optimizer.step()

			if epoch % self.args.epoch_per_test == 0 and epoch > 0: 
				self.eval()

				with torch.no_grad():

					for batch_idx, ( X, X_raw, size_factor ) in enumerate(test_loader): 

						if self.args.use_cuda:
							X, X_raw, size_factor = X.cuda(), X_raw.cuda(), size_factor.cuda()

						X, X_raw, size_factor     = Variable( X ), Variable( X_raw ), Variable( size_factor )

						loss      = self.return_loss( X, X_raw, size_factor )
						test_loss = torch.mean( loss )

						train_loss_list.append( test_loss.item() )

						print( test_loss.item() )

						if math.isnan(test_loss.item()):
							flag_break = 1
							break

						if test_like_max >  test_loss.item():
							test_like_max   = test_loss.item()
							reco_epoch_test = epoch
							patience_epoch  = 0        

			if flag_break == 1:
				print("containin NA")
				print(epoch)
				break

			if patience_epoch >= 30 :
				print("patient with 50")
				print(epoch)
				break
			
			if len(train_loss_list) >= 2 :
				if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4 :

					print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )
					print( "converged!!!" )
					print( epoch )
					break

		duration = time.time() - start

		print('Finish training, total time is: ' + str(duration) + 's' )
		self.eval()
		print(self.training)

		print( 'train likelihood is :  '+ str(test_like_max) + ' epoch: ' + str(reco_epoch_test) )


def build_multi_layers(layers, use_batch_norm=True, dropout_rate = 0.1 ):
	"""Build multilayer linear perceptron"""

	if dropout_rate > 0:
		fc_layers = nn.Sequential(
			collections.OrderedDict(
				[
					(
						"Layer {}".format(i),
						nn.Sequential(
							nn.Linear(n_in, n_out),
							nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
							nn.ReLU(),
							nn.Dropout(p=dropout_rate),
						),
					)

					for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
				]
			)
		)

	else:
		fc_layers = nn.Sequential(
			collections.OrderedDict(
				[
					(
						"Layer {}".format(i),
						nn.Sequential(
							nn.Linear(n_in, n_out),
							nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
							nn.ReLU(),
						),
					)

					for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
				]
			)
		)
		
		
	
	return fc_layers


class Encoder(Module):
	
	## for one modulity
	def __init__(self, layer, hidden, Z_DIMS, droprate = 0.1 ):
		super(Encoder, self).__init__()
		
		if len(layer) > 1:
			self.fc1   =  build_multi_layers( layers = layer, dropout_rate = droprate )
			
		self.layer = layer
		self.fc_means   =  nn.Linear(hidden, Z_DIMS)
		self.fc_logvar  =  nn.Linear(hidden, Z_DIMS)
		
	def reparametrize(self, means, logvar):

		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(means)

		else:
		  return means

	def return_all_params(self, x):

		if len(self.layer) > 1:
			h = self.fc1(x)
		else:
			h = x

		mean_x   = self.fc_means(h)
		logvar_x = self.fc_logvar(h)
		latent   = self.reparametrize(mean_x, logvar_x)
		
		return mean_x, logvar_x, latent, h

		
	def forward(self, x):

		_, _, latent = self.return_all_params( x )
		
		return latent


class Decoder(Module):
	### for scATAC-seq
	def __init__(self, layer, hidden, input_size, Type = "Bernoulli" , droprate = 0.1 ):
		super(Decoder, self).__init__()
		
		if len(layer) >1 :
			self.decoder   =  build_multi_layers( layer, dropout_rate = droprate )
		
		self.decoder_x = nn.Linear( hidden, input_size )
		self.Type      = Type
		self.layer     = layer

	def forward(self, z):
		
		if len(self.layer) >1 :
			latent  = self.decoder( z )
		else:
			latent = z
			
		recon_x = self.decoder_x( latent )
		
		if self.Type == "Bernoulli":
			Final_x = torch.sigmoid(recon_x)
			
		elif self.Type == "Gaussian":
			Final_x = F.relu(recon_x)

		else:
			Final_x = recon_x
		
		return Final_x



class Decoder_logNorm_NB(Module):
	
	### for scRNA-seq
	
	def __init__(self, layer, hidden, input_size, droprate = 0.1  ):
		
		super(Decoder_logNorm_NB, self).__init__()

		self.decoder =  build_multi_layers( layers = layer, dropout_rate = droprate  )
		
		self.decoder_scale = nn.Linear(hidden, input_size)
		self.decoder_r = nn.Linear(hidden, input_size)

	def forward(self, z, scale_factor = torch.tensor(1.0)):
		
		latent = self.decoder(z)
		
		normalized_x = F.softmax( self.decoder_scale( latent ), dim = 1 )  ## mean gamma

		batch_size   = normalized_x.size(0)
		scale_factor.resize_(batch_size,1)
		scale_factor.repeat(1, normalized_x.size(1))

		scale_x      =  torch.exp(scale_factor) * normalized_x
		
		disper_x     =  torch.exp( self.decoder_r( latent ) ) ### theta
		
		return dict( normalized      =  normalized_x,
					 disperation     =  disper_x,
					 scale_x         =  scale_x,
				   )


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

	lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr   

def log_nb_positive(x, mu, theta, eps=1e-8):
	
	x = x.float()
	
	if theta.ndimension() == 1:
		theta = theta.view(
			1, theta.size(0)
		)  # In this case, we reshape theta for broadcasting

	log_theta_mu_eps = torch.log(theta + mu + eps)

	res = (
		theta * (torch.log(theta + eps) - log_theta_mu_eps)
		+ x * (torch.log(mu + eps) - log_theta_mu_eps)
		+ torch.lgamma(x + theta)
		- torch.lgamma(theta)
		- torch.lgamma(x + 1)
	)

	#print(res.size())

	return - torch.sum( res, dim = 1 )

def mse_loss(y_true, y_pred):

	y_pred = y_pred.float()
	y_true = y_true.float()

	ret = torch.pow( (y_pred - y_true) , 2)

	return torch.sum( ret, dim = 1 )

