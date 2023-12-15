# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""

from PIL.Image import NONE
import cv2
import json

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import numpy as np
import pandas as pd
import os
import glob2
import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from anndata import AnnData
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib.image import imread

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.modules.module import Module

class CustomDataset(Dataset):

	def __init__(self, imgs_path = None, sampling_index = None, sub_path = None, 
				 file_code = None, transform = None):

		file_list = None

		if file_code is not None:
			temp_file_list = []

			for z in list(range(len(file_code))):
				temp_files    =  glob2.glob( sub_path + file_code[z] + "/tmp/*.jpeg" )
				temp_file_list.extend(temp_files)
			file_list = temp_file_list

		else:
			file_list = glob2.glob( str(imgs_path) + "/*.jpeg" )

		self.data      = []
		self.barcode   = []

		if file_code is not None:
			if sampling_index is not None:
				for index in sampling_index:
					self.data.append( file_list[index] )
					#self.barcode.append( file_list[index].rpartition("/")[-1].rpartition("40.jpeg")[0] )
					temp_code1 = file_list[index].rpartition("/")[0].rpartition("/")[0].rpartition("/")[2]
					temp_code2 = file_list[index].rpartition("/")[-1].rpartition("40.jpeg")[0]
					self.barcode.append( temp_code1 + "_" + temp_code2 )
			else:
				for file in file_list:
					self.data.append( file )
					temp_code1 = file.rpartition("/")[0].rpartition("/")[0].rpartition("/")[2]
					temp_code2 = file.rpartition("/")[-1].rpartition("40.jpeg")[0]
					self.barcode.append( temp_code1 + "_" + temp_code2 )
		else:
			for file in file_list:
				self.data.append( file )
				self.barcode.append( file.rpartition("/")[-1].rpartition("40.jpeg")[0] )

		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path   = self.data[idx]
		img        = Image.open( img_path )

		image_code = self.barcode[idx]

		if self.transform is not None:
			pos_1 = self.transform(img)
			pos_2 = self.transform(img)

		return pos_1, pos_2, image_code


train_transform_64 = transforms.Compose([
	transforms.RandomResizedCrop(64),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
	transforms.RandomGrayscale(p=0.8),
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_32 = transforms.Compose([
	transforms.RandomResizedCrop(32),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
	transforms.RandomGrayscale(p=0.8),
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def tiling(
	adata: AnnData,
	out_path: str = None,
	library_id: str = None,
	crop_size: int = 40,
	target_size: int = 32,
	verbose: bool = False,
	copy: bool = False,
) -> Optional[AnnData]:
	"""
	adopted from stLearn package
	Tiling H&E images to small tiles based on spot spatial location
	"""

	if library_id is None:
		library_id = list(adata.uns["spatial"].keys())[0]

	# Check the exist of out_path
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	image = adata.uns["spatial"][library_id]["images"][
		adata.uns["spatial"][library_id]["use_quality"]
	]
	if image.dtype == np.float32 or image.dtype == np.float64:
		image = (image * 255).astype(np.uint8)
	img_pillow = Image.fromarray(image)
	tile_names = []

	with tqdm(
		total=len(adata),
		desc="Tiling image",
		bar_format="{l_bar}{bar} [ time left: {remaining} ]",
	) as pbar:
		for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
			imagerow_down  = imagerow - crop_size / 2
			imagerow_up    = imagerow + crop_size / 2
			imagecol_left  = imagecol - crop_size / 2
			imagecol_right = imagecol + crop_size / 2
			tile           = img_pillow.crop( (imagecol_left, imagerow_down,
											   imagecol_right, imagerow_up)
											)
			tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
			tile.resize((target_size, target_size))

			tile_name = str(barcode) + str(crop_size)
			out_tile  = Path(out_path) / (tile_name + ".jpeg")

			tile_names.append(str(out_tile))

			if verbose:
				print(
					"generate tile at location ({}, {})".format(
						str(imagecol), str(imagerow)
					)
				)
			tile.save(out_tile, "JPEG")

			pbar.update(1)

	adata.obs["tile_path"] = tile_names
	return adata if copy else None


class simCLR_model(Module):
	def __init__(self, feature_dim=128):
		super(simCLR_model, self).__init__()

		self.f = []

		#load resnet50 structure
		for name, module in models.resnet50().named_children():
			if name == 'conv1':
				module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)
		# projection head
		self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
							   nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		out     = self.g(feature)

		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, args):

	net.train().to("cuda:0")

	total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

	for pos_1, pos_2, _ in train_bar:

		pos_1, pos_2  = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		_, out_1      = net(pos_1)
		_, out_2      = net(pos_2)
		
		out           = torch.cat([out_1, out_2], dim=0) # [2*B, D]
		sim_matrix    = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature) # [2*B, 2*B]
		mask          = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size_I, 
						 device=sim_matrix.device)).bool()
		
		sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size_I, -1) # [2*B, 2*B-1]

		# compute loss
		pos_sim    = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
		
		pos_sim    = torch.cat([pos_sim, pos_sim], dim=0) # [2*B]
		loss       = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

		train_optimizer.zero_grad()
		loss.backward()
		train_optimizer.step()

		total_num  += args.batch_size_I
		total_loss += loss.item() * args.batch_size_I
		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(args.current_epoch_I, args.max_epoch_I, total_loss / total_num))

	return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image

def test(net, test_data_loader, args):

	net.eval().to("cuda:1")

	total_loss, total_num, test_bar = 0.0, 0, tqdm(test_data_loader)

	with torch.no_grad():

		for pos_1, pos_2, _ in test_bar:

			pos_1, pos_2 = pos_1.to("cuda:1"), pos_2.to("cuda:1")

			_, out_1   = net(pos_1)
			_, out_2   = net(pos_2)

			out        = torch.cat([out_1, out_2], dim=0) # [2*B, D]
			sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature) # [2*B, 2*B]
			mask       = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size_I, 
						  device=sim_matrix.device)).bool()
			sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size_I, -1) # [2*B, 2*B-1]

			# compute loss
			pos_sim    = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
			pos_sim    = torch.cat([pos_sim, pos_sim], dim=0) # [2*B]
			loss       = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

			total_num  += args.batch_size_I
			total_loss += loss.item() * args.batch_size_I
			test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f}'.format(args.current_epoch_I, args.max_epoch_I, total_loss / total_num))

	return total_loss / total_num


def train_simCLR_sImage( args, sub_path = None, file_code = None):

	latent_I, k            =  args.latent_I, args.k
	batch_size, epochs     =  args.batch_size_I, args.max_epoch_I
	test_prop              =  args.test_prop
	file_list              =  None

	if file_code is not None:
		temp_file_list = []

		for z in list(range(len(file_code))):
			temp_files    =  glob2.glob( sub_path + file_code[z] + "/tmp/*.jpeg" )
			temp_file_list.extend( temp_files )

		file_list = temp_file_list

	else:
		file_list         = glob2.glob( str(args.tillingPath) + "/*.jpeg" )

	train_idx, test_idx   = train_test_split(np.arange(len(file_list)),
											 test_size    = 0.15, random_state = 200)
	# data prepare
	print('step1:  ')
	
	if args.image_size == 64:
		train_data    = CustomDataset(imgs_path = args.tillingPath, sampling_index = train_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_64 )

		test_data     = CustomDataset(imgs_path = args.tillingPath, sampling_index = test_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_64)

	else:
		train_data    = CustomDataset(imgs_path = args.tillingPath, sampling_index = train_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_32 )

		test_data     = CustomDataset(imgs_path = args.tillingPath, sampling_index = test_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_32)
	
	train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
							   pin_memory=True, drop_last=True)

	test_loader   = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
							   pin_memory=True, drop_last=True)

	# model setup and optimizer config
	print('step2:  ')

	model      = simCLR_model(latent_I).cuda()
	optimizer  = optim.Adam(model.parameters(), lr=args.lr_I, weight_decay=1e-6)

	# training loop
	minimum_loss  = 10000

	for epoch in range(1, epochs + 1):
		print('epoch:  '+ str(epoch))
		args.current_epoch_I = epoch
		train_loss           = train(model, train_loader, optimizer, args)
		test_loss            = test(model, test_loader, args)

		if test_loss < minimum_loss:
			minimum_loss = test_loss
			torch.save(model.state_dict(), args.outPath + args.visual_model)


def extract_representation_simCLR_model( args, adata, sub_path = None, file_code = None):

	model  =  simCLR_model(args.latent_I).cuda()
	model.load_state_dict( torch.load( args.outPath + args.visual_model ) )
	model.eval()

	latent_I, k              = args.latent_I, args.k
	batch_size, epochs       = args.batch_size_I, args.max_epoch_I
	test_prop                = args.test_prop

	# data prepare
	print('step1:  ')
	total_data    = CustomDataset(imgs_path = args.tillingPath,
								  sub_path  = sub_path, file_code = file_code, 
								  transform = test_transform )

	total_loader  = DataLoader(total_data, batch_size=args.batch_size_I, shuffle=False, 
							   pin_memory=True, drop_last=False)

	print('step2:  ')

	total_bar     = tqdm(total_loader)
	feature_dim   = []
	barcode       = []

	for image, _, image_code in total_bar:

		image     = image.cuda(non_blocking=True)
		feature,_ = model(image)

		feature_dim.append( feature.data.cpu().numpy() )
		barcode.append( image_code )

	feature_dim = np.concatenate(feature_dim)
	barcode     = np.concatenate(barcode)

	data_frame  = pd.DataFrame(data=feature_dim, index=barcode, columns =  list(range(1, 2049)) )
	data_frame.index  = data_frame.index.map(str)
	data_frame_1      = data_frame.loc[adata.obs_names]
	data_frame_1.to_csv( args.outPath + args.visualFeature , sep='\t')
	

class resnet50_model(Module):
	def __init__(self):
		super(resnet50_model, self).__init__()

		### load pretrained resnet50 model
		resnet50 = models.resnet50(pretrained=True)

		for param in resnet50.parameters():
			param.requires_grad = False

		self.f = []

		for name, module in resnet50.named_children():
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
	   
		return F.normalize(feature, dim=-1)


def extract_representation_resnet50_model( args, adata, sub_path = None, file_code = None ):
	## extract 2048 representation from pretrained resnet50
	batch_size    = args.batch_size_I

	# data prepare
	print('step1:  ')
	total_data    = CustomDataset(imgs_path = args.tillingPath,
								  sub_path  = sub_path, file_code = file_code, 
								  transform = test_transform )

	total_loader  = DataLoader(total_data, batch_size=args.batch_size_I, shuffle=False, 
							   pin_memory=True, drop_last=False)

	print('step2:  ')
	model       = resnet50_model().eval().cuda()
	total_bar   = tqdm(total_loader)
	feature_dim = []
	barcode     = []

	for image, _, image_code in total_bar:

		image   = image.cuda(non_blocking=True)
		feature = model(image)

		feature_dim.append( feature.data.cpu().numpy() )
		barcode.append( image_code )

	feature_dim = np.concatenate(feature_dim)
	barcode     = np.concatenate(barcode)

	data_frame        = pd.DataFrame(data=feature_dim, index=barcode, columns =  list(range(1, 2049)) )
	data_frame.index  = data_frame.index.map(str)
	data_frame_1      = data_frame.loc[adata.obs_names]
	data_frame_1.to_csv( args.outPath + args.visualFeature , sep='\t')
	
