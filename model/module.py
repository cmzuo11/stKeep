# -*- coding: utf-8 -*-
"""
Created on Sun Oct 2 10:40:25 2022

@author: chunman zuo
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from Layer import Mp_encoder, Sc_encoder, Contrast, MaskedLinear
from Layer import CCI_attention, build_multi_layers, Contrast_single


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
		self.mp       = Mp_encoder(P, hidden_dim, attn_drop)
		self.sc       = Sc_encoder(hidden_dim, nei_num, attn_drop, sample_rate)      
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


class CCC(nn.Module):

	def __init__(self, cci_pairs, hidden_dim, attn_drop, layers, tau):
		super(CCC, self).__init__()

		self.CCI_attention = CCI_attention(cci_pairs, hidden_dim, attn_drop)
		self.enco_latent   = nn.Linear(layers[0], layers[1], bias=False)
		self.contrast      = Contrast_single(hidden_dim, tau)

	def forward(self, nei_adj, ligand_exp, receptor_exp, pos):

		embeds_LRs   = self.CCI_attention(nei_adj, ligand_exp, receptor_exp)
		latent       = self.enco_latent(embeds_LRs)
		lori         = self.contrast(latent, pos)

		return lori

	def return_LR_value(self, nei_adj, ligand_exp, receptor_exp):

		embeds_LRs = self.CCI_attention(nei_adj, ligand_exp, receptor_exp)

		return embeds_LRs

	def return_LR_atten(self, nei_adj, ligand_exp, receptor_exp):

		atten_list = self.CCI_attention.return_LR_atten_spot(nei_adj, ligand_exp, receptor_exp)

		return atten_list


class Mp_encoder_HIN(nn.Module):
	def __init__(self, hidden_dim, feats_dim_list, feat_drop, 
				 attn_drop, P, tau):
		super(Mp_encoder_HIN, self).__init__()

		self.hidden_dim = hidden_dim
		self.fc_list    = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
										for feats_dim in feats_dim_list])

		for fc in self.fc_list:
			nn.init.xavier_normal_(fc.weight, gain=1.414)

		if feat_drop > 0:
			self.feat_drop = nn.Dropout(feat_drop)
		else:
			self.feat_drop = lambda x: x

		self.mp       = Mp_encoder(P, hidden_dim, attn_drop)
		self.contrast = Contrast_single(hidden_dim, tau)

	def forward(self, feats, pos, mps, nei_index=None): 
		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		z    = self.mp(h_all[0], mps)
		lori = self.contrast(z, pos)

		return lori

	def get_mp_embeds(self, feats, mps):
		z_mp     = F.elu(self.feat_drop(self.fc_list[0](feats[0])))
		z_mp     = self.mp(z_mp, mps)

		return z_mp.detach()

	def get_attention_mp(self, feats, mps):

		z_mp         = F.elu(self.feat_drop(self.fc_list[0](feats[0])))
		atten_mul_V  = self.mp.return_mul_view_attention(z_mp, mps)

		return atten_mul_V.detach(), None, None, None, None

class Sc_encoder_HIN(nn.Module):
	def __init__(self, hidden_dim, feats_dim_list, feat_drop, 
				 attn_drop, sample_rate, nei_num, tau):
		super(Sc_encoder_HIN, self).__init__()

		self.hidden_dim = hidden_dim
		self.fc_list    = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
										for feats_dim in feats_dim_list])

		for fc in self.fc_list:
			nn.init.xavier_normal_(fc.weight, gain=1.414)

		if feat_drop > 0:
			self.feat_drop = nn.Dropout(feat_drop)
		else:
			self.feat_drop = lambda x: x

		self.sc       = Sc_encoder(hidden_dim, nei_num, attn_drop, sample_rate)
		self.contrast = Contrast_single(hidden_dim, tau)

	def forward(self, feats, pos, mps=None, nei_index=None): 
		h_all    = []
		for i in range(len(feats)):
			bb   = F.elu(self.feat_drop(self.fc_list[i](feats[i])))
			h_all.append(bb)

		z    = self.sc(h_all, nei_index)
		lori = self.contrast(z, pos)

		return lori

	def get_sc_embeds(self, feats, nei_index):
		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		z_sc  = self.sc(h_all, nei_index)

		return z_sc.detach()

	def get_attention_sc(self, feats, nei_index):

		h_all    = []
		for i in range(len(feats)):
			h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

		atten_var   = self.sc.return_var_attention(h_all, nei_index)
		atten_inter = self.sc.return_inter_attention(h_all, nei_index)

		return atten_var.detach(), atten_inter.detach()


