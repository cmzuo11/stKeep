# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:16:25 2022

@author: chunman zuo
"""
import numpy
import torch
import warnings
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path

from utilities import parameter_setting
from model_training_1 import Trian_hin_model


def train_with_argas( args ):

	basePath         = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/NG_BRCA/CID44971/'
	args.use_cuda    = args.use_cuda and torch.cuda.is_available()
	## random seed ##
	numpy.random.seed( args.seed )
	random.seed( args.seed )
	torch.manual_seed( args.seed )
	torch.cuda.manual_seed( args.seed )

	args.inputPath       = basePath
	args.outPath         = basePath + 'HVG_3000/'
	args.spotGene        = basePath + 'Spot_gene_neighbors_new.txt'
	args.spotGroup       = basePath + 'Spot_groups_new.txt'
	args.spotLatent      = basePath + 'HVGs_3000_AE_50.txt'
	#args.GeneLatent      = basePath + 'HVG_3000_embed_1/Spot_rep_SC_0.002_40_0.4_0.4_0.0_0.1_20_499.csv'
	args.GeneLatent      = None
	args.visualFeature   = basePath + 'Spot_image_similar_simCLR.txt'
	args.spatialLocation = basePath + 'Spot_location_order.txt'
	args.pos_pair        = basePath + 'Spot_pos_new_self.txt'

	args.spotGene_exp    = basePath + 'Spot_gene_Spot_adjancy.txt'
	args.spot_Group      = basePath + 'Spot_group_Spot_adjancy.txt'
	args.cci_files       = basePath + 'CCI_spot_adjace.txt'

	args.Node_list    = [1162, 3000, 16]
	args.knn          = 7

	args.lr            = lr_list[z]
	args.patience      = patience_list[zz]
	args.tau           = tau_list[zzz]
	args.lam           = lam_list[zm]
	args.feat_drop     = feat_drop_list[zzzzz]
	args.sample_rate   = [sample_rate_list[zi],1]
	args.attn_drop     = attn_drop_list[zzzz]
								
	loss, final_epo = Trian_hin_model(args, pattern = None)
	
if __name__ == "__main__":
	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)