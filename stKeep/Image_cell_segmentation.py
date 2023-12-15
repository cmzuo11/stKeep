# -*- coding: utf-8 -*-
"""

@author: chunman zuo
"""

import numpy as np
import pandas as pd
import json
import cv2
from pathlib import Path
import os
import time

from utilities import parameter_setting

print('Start processing cell segmentation')

start           =  time.time()
parser          =  parameter_setting()
args            =  parser.parse_args()

imageSeg_dir    = args.inputPath + 'image_segmentation/'
imageSeg_d      = Path( imageSeg_dir )
imageSeg_d.mkdir(parents=True, exist_ok=True)

image_loc_in    = pd.read_csv(args.inputPath + 'spatial/Spot_location.csv', header = 0, index_col = 0)

args.jsonFile   = 'tissue_hires_image.json'
image_width     = 2000
image_height    = 2000
crop_size       = 40
shape           = (image_width, image_height)
sh_fname        = os.path.join(imageSeg_dir, args.jsonFile)

with open(sh_fname, 'r') as f:
	sh_json = json.load(f)

shape           = (image_width, image_height)
check_or_not    = [False] * len(image_loc_in.index)
cell_type_dict  = {}	
region_pro      = 0.5

for sh in sh_json['shapes']:
	polys  = []
	geom   = np.squeeze(np.array(sh['points']))
	pts    = geom.astype(int)
	polys.append(pts)
	mask   = np.zeros(shape)
	ss     = cv2.fillPoly(mask, polys, 1)
	mask   = mask.astype(int)
	temp_celltype = sh['label']
	print(temp_celltype)
	count_status  = -1
	for barcode, imagerow, imagecol in zip(image_loc_in.index, image_loc_in["imagerow"], image_loc_in["imagecol"]):
		count_status   = count_status + 1
		imagerow_down  = imagerow - crop_size / 2
		imagerow_up    = imagerow + crop_size / 2
		imagecol_left  = imagecol - crop_size / 2
		imagecol_right = imagecol + crop_size / 2
		spots = []
		spot  = np.array([[imagecol_left, imagerow_up],[imagecol_left, imagerow_down], 
						 [imagecol_right,imagerow_down], [imagecol_right,imagerow_up]])
		pts   = spot.astype(int)
		spots.append(pts)
		mask1 = np.zeros(shape)
		ss1   = cv2.fillPoly(mask1, spots, 1)
		mask1 = mask1.astype(int)
		mask2 = mask + mask1

		if np.sum(mask2>1)/np.sum(mask1>0) > region_pro:
			if check_or_not[count_status] is False:
				if cell_type_dict.__contains__(temp_celltype):
					temp_sopt_list = cell_type_dict[temp_celltype]
					temp_sopt_list.append(barcode)
				else:
					temp_spo_l = []
					temp_spo_l.append( barcode )
					cell_type_dict[temp_celltype] = temp_spo_l
				check_or_not[count_status] = True

cell_types = []
cell_names = []
for cell_type in cell_type_dict.keys():
	temp_barcodes = cell_type_dict[cell_type]
	cell_names.extend( temp_barcodes )
	cell_types.extend( [cell_type] * len(temp_barcodes) )

remain_cells = list(set(image_loc_in.index.tolist()) - set(cell_names))
cell_names.extend(remain_cells)
cell_types.extend( ['cluster'+str((len(cell_type_dict)+1))] * len(remain_cells) )
df           = {'Cell_type': cell_types }
data_frame   = pd.DataFrame(data=df, index = cell_names ).to_csv( imageSeg_dir + 'Image_cell_segmentation_0.5.csv' ) 

duration     = time.time() - start
print('Finish training, total time is: ' + str(duration) + 's' )

