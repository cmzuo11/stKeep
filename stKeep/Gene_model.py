# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""
import numpy
import torch
import warnings
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path

from utilities import parameter_setting
from model_training import Trian_gene_model


def train_with_argas( args ):

	## random seed 
	numpy.random.seed( args.seed )
	random.seed( args.seed )
	torch.manual_seed( args.seed )
	torch.cuda.manual_seed( args.seed )

	start = time.time()

	args.outPath         = args.inputPath + 'stKeep/'
	args.use_cuda        = args.use_cuda and torch.cuda.is_available()

	args.geneGroup       = args.outPath + args.geneGroup
	args.spotLatent      = args.outPath + args.spotLatent
	args.geneSpot        = args.outPath + args.geneSpot
	args.pos_pair_gene   = args.outPath + args.pos_pair_gene 
	args.geneName        = args.outPath + args.geneName 

	args.lr              = 0.05
	args.patience        = 20
	args.tau             = 0.4
	args.lam             = 0.4
	args.feat_drop       = 0.0
	args.sample_rate     = [30, 1]
	args.attn_drop       = 0.2
	
	Trian_gene_model( args )


	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )
								
if __name__ == "__main__":
	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)
