# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:16:25 2022

@author: chunman zuo
"""
import numpy
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
from model_training import Trian_hin_model, Trian_single_HIN_model, Trian_CCI_classify, Trian_CCI_contrast


def train_with_argas( args ):

	basePath         = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/NG_BRCA/CID44971/HVG_3000/'
	args.use_cuda    = args.use_cuda and torch.cuda.is_available()
	## random seed ##
	numpy.random.seed( args.seed )
	random.seed( args.seed )
	torch.manual_seed( args.seed )
	torch.cuda.manual_seed( args.seed )

	args.inputPath       = basePath
	args.outPath         = basePath + 'HVG_3000_LR_1/'
	args.spatialLocation = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/NG_BRCA/CID44971/Spot_location_order.txt'

	args.Ligands_exp     = basePath + 'ligands_expression_0.002_30_0.3_0.2_0.1_0.1_30_872.txt'
	args.Receptors_exp   = basePath + 'receptors_expression_0.002_30_0.3_0.2_0.1_0.1_30_872.txt'
	args.Cluster_file    = basePath + 'Spots_HIN_clustering_0.002_30_0.3_0.2_0.1_0.1_30_872.txt'

	args.pos_pair        = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_check/NG_BRCA/CID44971/Spot_pos_new_self.txt'
	#args.pos_pair        = basePath + 'Spot_pos_new_self_stKPI.txt'

	args.Node_list    = [1162, 3000, 16]
	args.knn          = 7
	args.cluster_pre  = 22
	attn_drop_list    = [ 0.0 ]
	tau_list          = [ 0.05, 0.1, 0.2, 0.3 ]
	lr_list           = [ 0.0005, 0.001, 0.002  ]
	args.patience     = 15
	args.lr_cci       = lr_list[z]
	args.attn_drop    = attn_drop_list[zz]
	args.tau          = tau_list[zzz]

	Trian_CCI_contrast(args)

if __name__ == "__main__":
	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)