# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import collections
import time
import math
import random

import torch.utils.data as data_utils
import scipy.sparse as sp
import torch.nn.modules.loss

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.distributions import Normal, kl_divergence as kl
from torch import optim

from torch.distributions import Normal, kl_divergence as kl

from Layers import Hierarchical_encoder, Semantic_encoder, Contrast, LRP_attention, build_multi_layers, Contrast_single, Decoder_logNorm_NB, Decoder
from utilities import adjust_learning_rate


class Cell_module(nn.Module):
	def __init__(self, hidden_dim, feats_dim_list, feat_drop, 
				 attn_drop, P, sample_rate, nei_num, tau, lam):
		super(Cell_module, self).__init__()
		self.hidden_dim = hidden_dim
		self.fc_list    = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
										for feats_dim in feats_dim_list])

		for fc in self.fc_list:
			nn.init.xavier_normal_(fc.weight, gain=1.414)

		if feat_drop > 0:
			self.feat_drop = nn.Dropout(feat_drop)
		else:
			self.feat_drop = lambda x: x
		self.mp       = Semantic_encoder(P, hidden_dim, attn_drop)
		self.sc       = Hierarchical_encoder(hidden_dim, nei_num, attn_drop, sample_rate)      
		self.contrast = Contrast(hidden_dim, tau, lam)

	def forward(self, feats, pos, mps, nei_index):  # p a s
		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
		z_mp     = self.mp(h_all[0], mps)
		z_sc     = self.sc(h_all, nei_index)
		lori_mp, lori_sc, lori_mp2mp, lori_sc2sc = self.contrast(z_mp, z_sc, pos)
		return lori_mp, lori_sc, lori_mp2mp, lori_sc2sc

	def get_mp_embeds(self, feats, mps):

		z_mp     = F.elu(self.feat_drop(self.fc_list[0](feats[0])))
		z_mp     = self.mp(z_mp, mps)

		return z_mp.detach()

	def get_sc_embeds(self, feats, nei_index):
		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		z_sc     = self.sc(h_all, nei_index)

		return z_sc.detach()

	def get_attention_mp(self, feats, mps):

		z_mp         = F.elu(self.feat_drop(self.fc_list[0](feats[0])))
		atten_mul_V  = self.mp.return_mul_view_attention(z_mp, mps)

		return atten_mul_V.detach(), None, None, None, None

	def get_attention_sc(self, feats, nei_index):

		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		atten_var   = self.sc.return_var_attention(h_all, nei_index)
		atten_inter = self.sc.return_inter_attention(h_all, nei_index)

		return atten_var.detach(), atten_inter.detach()


class CCI_model(nn.Module):

	def __init__(self, cci_pairs, hidden_dim, attn_drop, layers, tau):
		super(CCI_model, self).__init__()

		self.LRP_attention = LRP_attention(cci_pairs, hidden_dim, attn_drop)
		self.enco_latent   = nn.Linear(layers[0], layers[1], bias=False)
		self.contrast      = Contrast_single(hidden_dim, tau)

	def forward(self, nei_adj, ligand_exp, receptor_exp, pos):

		embeds_LRs   = self.LRP_attention(nei_adj, ligand_exp, receptor_exp)
		latent       = self.enco_latent(embeds_LRs)
		lori         = self.contrast(latent, pos)
		return lori

	def return_LRP_strength(self, nei_adj, ligand_exp, receptor_exp):

		embeds_LRs = self.LRP_attention(nei_adj, ligand_exp, receptor_exp)

		return embeds_LRs

	def return_LR_atten(self, nei_adj, ligand_exp, receptor_exp):

		atten_list = self.LRP_attention.return_LR_atten_spot(nei_adj, ligand_exp, receptor_exp)

		return atten_list


class Gene_module(nn.Module):
	def __init__(self, hidden_dim, feats_dim_list, feat_drop, 
				 attn_drop, sample_rate, nei_num, tau):
		super(Gene_module, self).__init__()

		self.hidden_dim = hidden_dim
		self.fc_list    = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
										for feats_dim in feats_dim_list])

		for fc in self.fc_list:
			nn.init.xavier_normal_(fc.weight, gain=1.414)

		if feat_drop > 0:
			self.feat_drop = nn.Dropout(feat_drop)
		else:
			self.feat_drop = lambda x: x

		self.sc       = Hierarchical_encoder(hidden_dim, nei_num, attn_drop, sample_rate)
		self.contrast = Contrast_single(hidden_dim, tau)

	def forward(self, feats, pos, nei_index=None): 
		h_all    = []
		for i in range(len(feats)):
			bb   = F.elu(self.feat_drop(self.fc_list[i](feats[i])))
			h_all.append(bb)

		z    = self.sc(h_all, nei_index)
		lori = self.contrast(z, pos)

		return lori

	def get_embeds(self, feats, nei_index):
		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		z_sc  = self.sc(h_all, nei_index)

		return z_sc.detach()

	def get_attention(self, feats, nei_index):

		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		atten_var   = self.sc.return_var_attention(h_all, nei_index)
		atten_inter = self.sc.return_inter_attention(h_all, nei_index)

		return atten_var.detach(), atten_inter.detach()


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

