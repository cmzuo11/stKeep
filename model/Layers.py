# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""

import torch
import torch.nn as nn
import numpy as np
import collections
import torch.nn.functional as F
import os

from typing import Optional
from torch.nn.modules.module import Module
from torch.autograd import Variable
from collections import OrderedDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

CUDA_LAUNCH_BLOCKING = 1


class Contrast(nn.Module):
	def __init__(self, hidden_dim, tau, lam):
		super(Contrast, self).__init__()
		self.proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ELU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		self.tau = tau
		self.lam = lam
		for model in self.proj:
			if isinstance(model, nn.Linear):
				nn.init.xavier_normal_(model.weight, gain=1.414)

	def sim(self, z1, z2):
		z1_norm = torch.norm(z1, dim=-1, keepdim=True)
		z2_norm = torch.norm(z2, dim=-1, keepdim=True)
		dot_numerator = torch.mm(z1, z2.t())
		dot_denominator = torch.mm(z1_norm, z2_norm.t())
		sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
		return sim_matrix

	def forward(self, z_mp, z_sc, pos):
		z_proj_mp    = self.proj(z_mp)
		z_proj_sc    = self.proj(z_sc)
		matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
		matrix_sc2mp = matrix_mp2sc.t()

		matrix_mp2mp = self.sim(z_proj_mp, z_proj_mp)
		matrix_sc2sc = self.sim(z_proj_sc, z_proj_sc)
		
		matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
		lori_mp      = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

		matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
		lori_sc      = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()

		matrix_mp2mp = matrix_mp2mp / (torch.sum(matrix_mp2mp, dim=1).view(-1, 1) + 1e-8)
		lori_mp2mp   = -torch.log(matrix_mp2mp.mul(pos).sum(dim=-1)).mean()

		matrix_sc2sc = matrix_sc2sc / (torch.sum(matrix_sc2sc, dim=1).view(-1, 1) + 1e-8)
		lori_sc2sc   = -torch.log(matrix_sc2sc.mul(pos).sum(dim=-1)).mean()

		#total        = self.lam * lori_mp + (1 - self.lam) * lori_sc
		#print("mp: " + str(lori_mp.data.cpu())+"  sc: "+ str(lori_sc.data.cpu())+"  total: "+ str(total.data.cpu()))

		return lori_mp, lori_sc, lori_mp2mp, lori_sc2sc

class Contrast_single(nn.Module):
	def __init__(self, hidden_dim, tau):
		super(Contrast_single, self).__init__()
		
		self.tau = tau

	def sim(self, z):
		z_norm = torch.norm(z, dim=-1, keepdim=True)
		dot_numerator   = torch.mm(z, z.t())
		dot_denominator = torch.mm(z_norm, z_norm.t())
		sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
		return sim_matrix

	def forward(self, z, pos):
		matrix  = self.sim(z)
		matrix  = matrix/(torch.sum(matrix, dim=1).view(-1, 1) + 1e-8)
		lori    = -torch.log(matrix.mul(pos).sum(dim=-1)).mean()

		return lori


class GraphAttentionLayer(Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""
	def __init__(self, in_features, out_features, dropout, alpha = 0.2):
		super(GraphAttentionLayer, self).__init__()
		self.dropout      = dropout
		self.in_features  = in_features
		self.out_features = out_features
		self.alpha        = alpha

		self.W         = nn.Parameter(torch.empty(size=(in_features, out_features)))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)

		self.a         = nn.Parameter(torch.empty(size=(2*out_features, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)

		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):
		Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
		e  = self._prepare_attentional_mechanism_input(Wh)

		zero_vec  = -9e15*torch.ones_like(e)
		attention = torch.where(adj.to_dense() > 0, e, zero_vec)
		attention = F.softmax(attention, dim=1)
		attention = F.dropout(attention, self.dropout, training=self.training)

		h_prime   = torch.matmul(attention, Wh)

		return h_prime

	def return_attention(self, h, adj):
		Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
		e  = self._prepare_attentional_mechanism_input(Wh)

		zero_vec  = -9e15*torch.ones_like(e)
		attention = torch.where(adj.to_dense() > 0, e, zero_vec)

		return attention

	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (N, 1)
		# e.shape (N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e   = Wh1 + Wh2.T
		return self.leakyrelu(e)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
	def __init__(self, in_ft, out_ft, dropout = 0.0, alpha = 0.2, nheads =2):
		"""Dense version of GAT."""
		super(GAT, self).__init__()

		self.out_att = GraphAttentionLayer(in_ft, out_ft, dropout=dropout, alpha=alpha)
		self.dropout = dropout
   
	def forward(self, x, adj):
		x = F.dropout(x, self.dropout)
		x = F.elu(self.out_att(x, adj))
		
		return x

	def return_attention_weight(self, x, adj):
		# 4727*4727
		x          = F.dropout(x, self.dropout)
		atten_node = self.out_att.return_attention(x, adj)
		
		return atten_node

class GCN(nn.Module):
	def __init__(self, in_ft, out_ft, bias=True):
		super(GCN, self).__init__()
		self.fc  = nn.Linear(in_ft, out_ft, bias=False)
		self.act = nn.PReLU()

		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_ft))
			self.bias.data.fill_(0.0)
		else:
			self.register_parameter('bias', None)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight, gain=1.414)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, seq, adj):
		seq_fts = self.fc(seq)
		out = torch.spmm(adj, seq_fts)
		if self.bias is not None:
			out += self.bias
		return self.act(out)

class Attention(nn.Module):
	def __init__(self, hidden_dim, attn_drop):
		super(Attention, self).__init__()
		self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.xavier_normal_(self.fc.weight, gain=1.414)

		self.tanh = nn.Tanh()
		self.att  = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
		nn.init.xavier_normal_(self.att.data, gain=1.414)

		self.softmax = nn.Softmax()
		if attn_drop:
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			self.attn_drop = lambda x: x

	def forward(self, embeds):
		beta = []
		attn_curr = self.attn_drop(self.att)
		for embed in embeds:
			sp = self.tanh(self.fc(embed)).mean(dim=0)
			beta.append(attn_curr.matmul(sp.t()))
		beta = torch.cat(beta, dim=-1).view(-1)
		beta = self.softmax(beta)
		#print(beta.size())
		z_mp = 0
		for i in range(len(embeds)):
			z_mp += embeds[i]*beta[i]
		return z_mp

	def return_attention(self, embeds):
		# 30 , 5, 2
		beta = []
		attn_curr = self.attn_drop(self.att)
		for embed in embeds:
			sp = self.tanh(self.fc(embed)).mean(dim=0)
			beta.append(attn_curr.matmul(sp.t()))
		beta = torch.cat(beta, dim=-1).view(-1)
		beta = self.softmax(beta)

		return beta


class inter_att(nn.Module):
	def __init__(self, hidden_dim, attn_drop):
		super(inter_att, self).__init__()
		self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.xavier_normal_(self.fc.weight, gain=1.414)

		self.tanh = nn.Tanh()
		self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
		nn.init.xavier_normal_(self.att.data, gain=1.414)

		self.softmax = nn.Softmax()
		if attn_drop:
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			self.attn_drop = lambda x: x

	def forward(self, embeds):
		beta = []
		attn_curr = self.attn_drop(self.att)
		for embed in embeds:
			sp = self.tanh(self.fc(embed)).mean(dim=0)
			beta.append(attn_curr.matmul(sp.t()))
		beta = torch.cat(beta, dim=-1).view(-1)
		beta = self.softmax(beta)
		#print("sc ", beta.data.cpu().numpy())  # type-level attention
		z_mc = 0
		for i in range(len(embeds)):
			z_mc += embeds[i] * beta[i]

		return z_mc

	def return_attention(self, embeds):
		# 30, 2
		beta = []
		attn_curr = self.attn_drop(self.att)
		for embed in embeds:
			sp = self.tanh(self.fc(embed)).mean(dim=0)
			beta.append(attn_curr.matmul(sp.t()))
		beta = torch.cat(beta, dim=-1).view(-1)
		beta = self.softmax(beta)

		return beta


class intra_att(nn.Module):
	def __init__(self, hidden_dim, attn_drop):
		super(intra_att, self).__init__()
		self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
		nn.init.xavier_normal_(self.att.data, gain=1.414)
		if attn_drop:
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			self.attn_drop = lambda x: x
			
		self.softmax   = nn.Softmax(dim=1)
		self.leakyrelu = nn.LeakyReLU()

	def forward(self, nei, h, h_refer):
		nei_emb   = F.embedding(nei, h)
		h_refer   = torch.unsqueeze(h_refer, 1)
		h_refer   = h_refer.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		nei_emb   = (att*nei_emb).sum(dim=1)
		return nei_emb

	def return_attention(self, nei, h, h_refer):
		#[4727,5,1], [4727,30,1]
		nei_emb   = F.embedding(nei, h)
		h_refer   = torch.unsqueeze(h_refer, 1)
		h_refer   = h_refer.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		att       = torch.squeeze(att, dim=-1)

		return att


class intra_att_LR(nn.Module):
	def __init__(self, hidden_dim, attn_drop):
		super(intra_att_LR, self).__init__()
		self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
		nn.init.xavier_normal_(self.att.data, gain=1.414)
		if attn_drop:
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			self.attn_drop = lambda x: x
			
		self.softmax   = nn.Softmax(dim=1)
		self.leakyrelu = nn.LeakyReLU()

		self.map_l     = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.uniform_(self.map_l.weight, a=0, b=1)
		nn.init.uniform_(self.map_l.bias, a=0, b=0.01)

		self.map_r     = nn.Linear(hidden_dim, hidden_dim, bias=True)
		nn.init.uniform_(self.map_r.weight, a=0, b=1)
		nn.init.uniform_(self.map_r.bias, a=0, b=0.01)

	def forward(self, nei, h, h_refer):
		h         = F.relu(self.map_l(h))
		h_refer   = F.relu(self.map_r(h_refer))

		nei_emb   = F.embedding(nei, h)
		h_refer_n = torch.unsqueeze(h_refer, 1)
		h_refer_n = h_refer_n.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer_n, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		nei_emb   = (att*nei_emb).sum(dim=1)
		nei_emb   = F.relu(nei_emb*h_refer)
		return nei_emb

	def return_attention(self, nei, h, h_refer):
		h         = F.relu(self.map_l(h))
		h_refer   = F.relu(self.map_r(h_refer))
		nei_emb   = F.embedding(nei, h)
		h_refer   = torch.unsqueeze(h_refer, 1)
		h_refer   = h_refer.expand_as(nei_emb)
		all_emb   = torch.cat([h_refer, nei_emb], dim=-1)
		attn_curr = self.attn_drop(self.att)
		att       = self.leakyrelu(all_emb.matmul(attn_curr.t()))
		att       = self.softmax(att)
		att       = torch.squeeze(att, dim=-1)

		return att

class LRP_attention(nn.Module):
	def __init__(self, cci_pairs, hidden_dim, attn_drop):
		super(LRP_attention, self).__init__()
		
		self.intra_cci  = nn.ModuleList([intra_att_LR(hidden_dim, attn_drop) for _ in range(cci_pairs)])
		self.cci_pairs  = cci_pairs

	def forward(self, sele_nei, ligand_exp, receptor_exp):
		LR_embeds     = []
		for z in range(self.cci_pairs):
			temp_emb  = self.intra_cci[z](sele_nei, ligand_exp[:,z].view(-1,1), receptor_exp[:,z].view(-1,1))
			LR_embeds.append( temp_emb.view(1,-1) )

		LR_embeds  = torch.cat(LR_embeds, dim=0)
		LR_embeds  = LR_embeds.t().cuda()

		return LR_embeds

	def return_LR_atten_spot(self, sele_nei, ligand_exp, receptor_exp):

		atten_list = []
		for z in range(self.cci_pairs):
			temp_atten = self.intra_cci[z].return_attention(sele_nei, ligand_exp[:,z].view(-1,1), receptor_exp[:,z].view(-1,1))
			atten_list.append(temp_atten)

		atten_list = torch.cat(atten_list, dim=1)

		return atten_list


class Hierarchical_encoder(nn.Module):
	def __init__(self, hidden_dim, nei_num, attn_drop, sample_rate):
		super(Hierarchical_encoder, self).__init__()
		self.intra       = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
		self.inter       = inter_att(hidden_dim, attn_drop)
		self.nei_num     = nei_num
		self.sample_rate = sample_rate

	def forward(self, nei_h, nei_index):
		embeds = []
		for i in range(self.nei_num):
			sele_nei = []
			sample_num = self.sample_rate[i]
			for per_node_nei in nei_index[i]:
				per_node_nei_n  = per_node_nei[per_node_nei>(-1)]
				if len(per_node_nei_n) >= sample_num:
					select_one = torch.tensor(np.array(per_node_nei_n[:sample_num]))[np.newaxis]
					#select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
					#                                           replace=False))[np.newaxis]
				else:
					select_one = torch.tensor(np.random.choice(per_node_nei_n, sample_num, replace=True))[np.newaxis]
				sele_nei.append(select_one)
		   
			sele_nei     = torch.cat(sele_nei, dim=0)
			sele_nei     = sele_nei.cuda()
			one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
			embeds.append(one_type_emb)
		z_mc = self.inter(embeds)
		return z_mc

	def return_var_attention(self, nei_h, nei_index):

		sele_nei   = []
		sample_num = self.sample_rate[0]

		for per_node_nei in nei_index[0]:
			per_node_nei_n  = per_node_nei[per_node_nei>(-1)]

			if len(per_node_nei_n) >= sample_num:
				select_one = torch.tensor(np.array(per_node_nei_n[:sample_num]))[np.newaxis]

			else:
				select_one = torch.tensor(np.random.choice(per_node_nei_n, sample_num, replace=True))[np.newaxis]
			sele_nei.append(select_one)
		   
		sele_nei   = torch.cat(sele_nei, dim=0)
		sele_nei   = sele_nei.cuda()
		atten_var  = self.intra[0].return_attention(sele_nei, nei_h[1], nei_h[0])

		return atten_var

	def return_inter_attention(self, nei_h, nei_index):
		
		embeds = []

		for i in range(self.nei_num):
			sele_nei   = []
			sample_num = self.sample_rate[i]

			for per_node_nei in nei_index[i]:
				per_node_nei_n  = per_node_nei[per_node_nei>(-1)]

				if len(per_node_nei_n) >= sample_num:
					select_one = torch.tensor(np.array(per_node_nei_n[:sample_num]))[np.newaxis]
				else:
					select_one = torch.tensor(np.random.choice(per_node_nei_n, sample_num, replace=True))[np.newaxis]

				sele_nei.append(select_one)
		   
			sele_nei     = torch.cat(sele_nei, dim=0)
			sele_nei     = sele_nei.cuda()
			one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
			embeds.append(one_type_emb)

		atten_inter = self.inter.return_attention(embeds)

		return atten_inter


class Semantic_encoder(nn.Module):
	def __init__(self, P, hidden_dim, attn_drop):
		super(Semantic_encoder, self).__init__()
		
		self.node_level     = nn.ModuleList([GAT(hidden_dim, hidden_dim) for _ in range(P)])
		self.att            = Attention(hidden_dim, attn_drop)
		self.P              = P

	def forward(self, h, mps):
		embeds = []
		for i in range(self.P):
			embeds.append(self.node_level[i](h, mps[i]))

		z_mp = self.att(embeds)
		return z_mp

	def return_mul_view_attention(self, h, mps):
		embeds = []
		for i in range(self.P):
			embeds.append(self.node_level[i](h, mps[i]))

		atten_mul_V = self.att.return_attention(embeds)

		return atten_mul_V


def build_multi_layers(layers, use_batch_norm=True, dropout_rate = 0.0 ):
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