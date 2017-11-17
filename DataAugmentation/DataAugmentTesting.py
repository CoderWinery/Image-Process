from DataAugmentation import DataAugment
import os
import sys
import glob
import random
import cv2
import numpy as np
import shutil
import xml.dom.minidom
from xml.etree import ElementTree as ET
import math
from matplotlib import pyplot as plt


# set data augmentation method 
# if methos = 1 =========> use
# lasting updating
DataAugmentMethod = {
	'_img_zoom' : 1,
	'_avg_blur' : 0,
	'_gaussain_blur' : 1,
	'_gaussain_noise' : 1,
	'_img_rotation' : 1,
	'_img_flip' : 1,
	'_img_contrast' : 1
}


# generate data

_generate_quantity = 2

# set the database relative path

database = './data'
xmlbase  = './xml'

# change dir to database path

os.chdir(database)

# get all of the '.jpg' file in the database path 

images = os.listdir('.')
images = glob.glob('*.jpg')

# get quantity of '.jpg' file

size = len(images)
print (size)

# check workspace

os.chdir('../')
print(os.getcwd())

# parameter for data augment functions

_max_filiter_size  = 5 		# for avg_blur and gaussain_blur
_sigma             = 0 		# for gaussain_blur

_mean              = 0 		# for gaussain_noise
_var               = 0.1		# for gaussain_noise

_x_min_shift_piexl = -20 	# for img_shift
_x_max_shift_piexl = 20 	# for img_shift
_y_min_shift_piexl = -20 	# for img_shift
_y_max_shift_piexl = 20		# for img_shift
_fill_pixel        = 255	# for img_shift and img_rotation

_min_angel         = -10	# for img_rotation
_max_angel         = 10		# for img_rotation
_min_scale         = 0.9	# for img_rotation
_max_scale         = 1.1	# for img_rotation

_min_zoom_scale    = 1		# for img_zoom
_max_zoom_scale    = 1.5	# for img_zoom

_min_s             = -5		# for img_contrast
_max_s             = 5		# for img_contrast
_min_v             = -5		# for img_contrast
_max_v             = 5		# for img_contrast

_min_h             = -10	# for img_color
_max_h             = 10		# for img_color

DataAugmentBase = './Augment/'

for i in images:
	generate_quantity = _generate_quantity
	print(i + "  done")
	while generate_quantity > 0 :
		flag = 1
		global_angle = 0
		global_scale = 1
		Matrix = [(0,0,0),(0,0,0)]
		flip_factor_var = 3
		img_dir = database + '/' + i
		img = cv2.imread(img_dir)
		h, w, c = img.shape
		print("w %d h %d c %d" % (w, h, c))
		#print (generate_quantity)
		if DataAugmentMethod['_img_zoom'] == 1:
			if random.randint(0, 1) == 1 :
				img, global_scale = DataAugment.img_zoom(img, _min_zoom_scale, _max_zoom_scale)

		if DataAugmentMethod['_avg_blur'] == 1 :
			if random.randint(0, 1) == 1 :
				img = DataAugment.avg_blur(img, _max_filiter_size)

		if DataAugmentMethod['_gaussain_blur'] == 1 :
			if random.randint(0, 1) == 1 :
				img = DataAugment.gaussain_blur(img, _max_filiter_size, _sigma)

		if DataAugmentMethod['_gaussain_noise'] == 1 :
			if random.randint(0, 1) == 1 :
				img = DataAugment.gaussain_noise(img, _mean, _var)

		if DataAugmentMethod['_img_rotation'] == 1 :

			img, global_angle, Matrix = DataAugment.img_rotation(img, _min_angel, _max_angel, _min_scale, _max_scale, _fill_pixel)

		if DataAugmentMethod['_img_flip'] == 1:
			if random.randint(0, 1) == 1 :
				img, flip_factor_var = DataAugment.img_flip(img)

		if DataAugmentMethod['_img_contrast'] == 1:
			if random.randint(0, 1) == 1 :
				img = DataAugment.img_contrast(img, _min_s, _max_s, _min_v, _max_v)


		save_dir = ('_%02d_') % (generate_quantity)

		# parse xml and save xml of augment img
		xml_name_dir = xmlbase + "/" + i.split(".")[0] + ".xml"
		augment_xml_name_dir = xmlbase + "/" + save_dir + i.split(".")[0] + ".xml"
		shutil.copyfile(xml_name_dir, augment_xml_name_dir)

		# read and modify and parse for new xml file
		with open(augment_xml_name_dir, "r") as xml_file:
			lines = xml_file.readlines()
			xml_file.seek(0)
			DOMTree = xml.dom.minidom.parse(xml_file)
			pos_xmin = DOMTree.getElementsByTagName("xmin")
			pos_xmax = DOMTree.getElementsByTagName("xmax")
			pos_ymin = DOMTree.getElementsByTagName("ymin")
			pos_ymax = DOMTree.getElementsByTagName("ymax")
			# initial threshold
			crop_min_x = 10000
			crop_min_y = 10000
			crop_max_x = 0
			crop_max_y = 0

			xml_count = 0
			# new height, width
			new_h, new_w = img.shape[:2]
			new_cx = new_w / 2
			new_cy = new_h / 2

			for num_0 in range(len(pos_xmax)):
				augment_xmin = pos_xmin[num_0].firstChild.data
				augment_ymin = pos_ymin[num_0].firstChild.data
				augment_xmax = pos_xmax[num_0].firstChild.data
				augment_ymax = pos_ymax[num_0].firstChild.data

				if crop_min_x > int(augment_xmin):
					crop_min_x = int(augment_xmin)

				if crop_max_x < int(augment_xmax):
					crop_max_x = int(augment_xmax)

				if crop_min_y > int(augment_ymin):
					crop_min_y = int(augment_ymin)

				if crop_max_y < int(augment_ymax):
					crop_max_y = int(augment_ymax)

				if num_0 == len(pos_ymax) - 1:
					crop_min_y = crop_min_y / 2
					crop_max_y = crop_max_y + (new_h - crop_max_y) / 2
					crop_min_x = crop_min_x / 2
					crop_max_x = crop_max_x + (new_w - crop_max_x) / 2
					img = img[int(crop_min_y):int(crop_max_y), int(crop_min_x):int(crop_max_x)]

			for num in range(len(pos_xmax)):
				# img_rotation
				augment_xmin = pos_xmin[num].firstChild.data
				augment_ymin = pos_ymin[num].firstChild.data
				augment_xmax = pos_xmax[num].firstChild.data
				augment_ymax = pos_ymax[num].firstChild.data

				# img_zoom
				augment_xmin = global_scale * float(augment_xmin)
				augment_ymin = global_scale * float(augment_ymin)
				augment_xmax = global_scale * float(augment_xmax)
				augment_ymax = global_scale * float(augment_ymax)

				v_min = [(float(augment_xmin),float(augment_xmax)),(float(augment_ymin),float(augment_ymax)),(1,1)]
				calculated = np.dot(Matrix, v_min)
				augment_xmin,augment_xmax = calculated[0]
				augment_ymin,augment_ymax = calculated[1]

				# img_flip
				if flip_factor_var == 1 :
					augment_xmin = new_w - augment_xmin
					augment_xmax = new_w - augment_xmax
				elif flip_factor_var == 0 :
					augment_ymin = new_h - augment_ymin
					augment_ymax = new_h - augment_ymax
				elif flip_factor_var == -1 :
					augment_xmin = new_w - augment_xmin
					augment_xmax = new_w - augment_xmax
					augment_ymin = new_h - augment_ymin
					augment_ymax = new_h - augment_ymax

				augment_xmin = math.floor(augment_xmin)
				augment_ymin = math.floor(augment_ymin)
				augment_xmax = math.floor(augment_xmax)
				augment_ymax = math.floor(augment_ymax)

				if augment_xmin < 0:
					augment_xmin = 0

				if augment_ymin < 0:
					augment_ymin = 0

				if augment_xmax < 0:
					augment_xmax = 0

				if augment_ymax < 0:
					augment_ymax = 0

				print(augment_xml_name_dir)

				augment_xmin = augment_xmin - crop_min_x
				augment_ymin = augment_ymin - crop_min_y
				augment_xmax = augment_xmax - crop_min_x
				augment_ymax = augment_ymax - crop_min_y

				print("augment xmin : %d ymin : %d xmax : %d ymax : %d" % (augment_xmin, augment_ymin, augment_xmax, augment_ymax))

				xml_line_num = len(lines) - 1
				tmp = 1
				while xml_count < xml_line_num:

					xml_count += 1
					if "<part" in lines[xml_count]:
						flag = 0
					if "</part" in lines[xml_count]:
						flag = 1

					if "<xmin>" in lines[xml_count]:
						if flag == 1:
							print("------------------------------------------>>>>")
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t<xmin>%d</xmin>\n" % augment_xmin)
						else:
							print("------------------------------------------>>>>")
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t\t<xmin>%d</xmin>\n" % augment_xmin)
					elif "<ymin>" in lines[xml_count]:
						if flag == 1:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t<ymin>%d</ymin>\n" % augment_ymin)
						else:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t\t<ymin>%d</ymin>\n" % augment_ymin)
					elif "<xmax>" in lines[xml_count]:
						if flag == 1:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t<xmax>%d</xmax>\n" % augment_xmax)
						else:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t\t<xmax>%d</xmax>\n" % augment_xmax)
					elif "<ymax>" in lines[xml_count]:
						if flag == 1:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t<ymax>%d</ymax>\n" % augment_ymax)
						else:
							lines[xml_count] = lines[xml_count].replace(lines[xml_count], "\t\t\t\t<ymax>%d</ymax>\n" % augment_ymax)
						print("===================================================>>>>")
						break

				open(augment_xml_name_dir, 'w').writelines(lines)

				cv2.rectangle(img,(int(augment_xmin),int(augment_ymin)),(int(augment_xmax),int(augment_ymax)),(0,255,0),2)

			save_dir = DataAugmentBase + save_dir + i
			cv2.imshow("transform", img)
			cv2.waitKey(0)
			generate_quantity -= 1
			img = img.astype(np.uint8)
			cv2.imwrite(save_dir, img)

