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
from model_training_1 import Trian_hin_model, Trian_single_HIN_model


def train_with_argas( args ):

	basePath         = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/NG_BRCA/CID44971/'
	args.use_cuda    = args.use_cuda and torch.cuda.is_available()
	## random seed ##
	numpy.random.seed( args.seed )
	random.seed( args.seed )
	torch.manual_seed( args.seed )
	torch.cuda.manual_seed( args.seed )
	
	args.inputPath       = basePath
	args.outPath         = basePath + 'HVG_3000_embed/'
	args.spotLatent      = basePath + 'HVGs_3000_AE_50.txt'
	args.geneSpot        = basePath + 'Gene_spot_adjancy_enco_clu.txt'
	args.geneGroup       = basePath + 'Gene_groups_enco_clu.txt'
	args.pos_pair_gene   = basePath + 'Gene_pos_pairs_enco_new_clu.txt'

	args.Node_list     = [5000, 1162,  22]
	args.knn           = 7

	args.lr            = lr_list[z]
	args.patience      = patience_list[zz]
	args.tau           = tau_list[zzz]
	args.lam           = lam_list[zzzz]
	args.feat_drop     = feat_drop_list[zzzzz]
	args.sample_rate   = [sample_rate_list[zi],1]
	args.attn_drop     = attn_drop_list[zm]
								
	Trian_single_HIN_model(args, "RNA", "SC")
								
								
if __name__ == "__main__":
	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)