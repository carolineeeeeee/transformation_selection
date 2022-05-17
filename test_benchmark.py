import torch
import torch.nn as nn
from robustbench import load_model
import pandas as pd
import pickle
import numpy as np
from albumentations.augmentations.transforms import *
import albumentations as A
import random
import cv2
import os
from torchvision import datasets, transforms
import skimage as sk

CIFAR_10_C_PATH = '/Users/caroline/Desktop/REforML/projects/automating_requirements/verifying/CIFAR-10-C/'
CIFAR_10_C_LABEL = '/Users/caroline/Desktop/REforML/projects/automating_requirements/verifying/CIFAR-10-C/labels.npy'
CIFAR_10_PATH = '/Users/caroline/Desktop/REforML/projects/automating_requirements/verifying/cifar-10-batches-py/'

CIFAR_10_C_PATH = '/u/boyue/ICSE2022-resubmission/CIFAR-10-C/'
CIFAR_10_C_LABEL = '/u/boyue/ICSE2022-resubmission/CIFAR-10-C/labels.npy'
CIFAR_10_PATH = '/u/boyue/ICSE2022-resubmission/cifar-10-batches-py/'



ROBUSTBENCH_CIFAR10_MODEL_NAMES = [
	"Diffenderfer2021Winning_LRR_CARD_Deck", "Diffenderfer2021Winning_LRR", "Diffenderfer2021Winning_Binary_CARD_Deck",
	"Kireev2021Effectiveness_RLATAugMix", "Hendrycks2020AugMix_ResNeXt",
	"Modas2021PRIMEResNet18", "Hendrycks2020AugMix_WRN", "Kireev2021Effectiveness_RLATAugMixNoJSD",
	"Diffenderfer2021Winning_Binary", "Kireev2021Effectiveness_AugMixNoJSD", 
	"Kireev2021Effectiveness_Gauss50percent", "Kireev2021Effectiveness_RLAT", 
	"Addepalli2021Towards_WRN34", "Standard"]
	#  "Rebuffi2021Fixing_70_16_cutmix_extra_L2","Rebuffi2021Fixing_70_16_cutmix_extra_Linf",

def get_model(model_name: str, pretrained: bool = True, val: bool = True) -> nn.Module:
	"""Return a pretrained model give a model name
	:param model_name: model name, choose from [alexnet, googlenet, resnet50, resnext50, vgg-16]
	:type model_name: str
	:param pretrained: Whether the model should be pretrained, defaults to True
	:type pretrained: bool, optional
	:param val: Whether the model in validation model (prevent weight from updating), defaults to True
	:type val: bool, optional
	:raises ValueError: Invalid Model Name
	:raises ValueError: Invalid Model Name
	:return: a pytorch model
	:rtype: nn.Module
	Source: https://github.com/carolineeeeeee/automating_requirements
	"""
	if model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES:
		if 'L2' in model_name:
			model = load_model(model_name=model_name, dataset='cifar10', threat_model="L2")
		elif 'Linf' in model_name:
			model = load_model(model_name=model_name, dataset='cifar10', threat_model="Linf")
		model = load_model(model_name=model_name, dataset='cifar10', threat_model="corruptions")
	else:
		raise ValueError(f"Invalid Model Name: {model_name}")
	if val:
		model.eval()
	return model

def convert_array_to_img(data, index):
	image_array = data[index]
	red = np.reshape(image_array[:1024], (32, 32))
	green = np.reshape(image_array[1024:2048], (32, 32))
	blue = np.reshape(image_array[2048:], (32, 32))

	final = np.stack((red, green, blue), axis=-1)
	#print(final.shape)

	return final


def Color_jitter (x, i):
	c = np.linspace(0, 1, 100)
	s = np.linspace(0, 1, 100)
	h = np.linspace(0, 1, 100)
	np.random.shuffle(s)
	np.random.shuffle(h)
	np.random.shuffle(c)

	color_jitter = A.ReplayCompose([ColorJitter(brightness=0, contrast=c[i], saturation=s[i], hue=h[i], always_apply=True)])
	transformed_img = color_jitter(image=x)
	# to get the parameters that are actually used
	return transformed_img['image']

def Sharpen_img(x):
	sharpen = A.ReplayCompose([Sharpen(always_apply=True)])
	sharpened_img = sharpen(image=x)
	return sharpened_img['image']

def brightness(x, i):
	"""
	Source: https://github.com/hendrycks/robustness
	"""
	c = np.linspace(0, 1, 100)	 

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 2] = np.clip(x[:, :, 2] + c[i], 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255#, c[i]

def Random_Gamma (x):
	RG = A.ReplayCompose([RandomGamma(gamma_limit=(0, 300), always_apply=True)])
	transformed_img = RG (image=x)
	return transformed_img['image']#, arguments

def entry_128(x, i):
	bright_x = brightness(x, i)
	bright_and_gamma = Random_Gamma(bright_x)
	return bright_and_gamma

def entry_5(x, i):
	bright_x = brightness(x, i)
	bright_and_sharp = Sharpen_img(bright_x)
	return bright_and_sharp

def contrast(x, i):
	"""
	Source: https://github.com/hendrycks/robustness
	"""
	c = np.linspace(0.01, 0.9, 100)

	x = np.array(x) / 255.
	means = np.mean(x, axis=(0, 1), keepdims=True)
	return np.clip((x - means) * c[i] + means, 0, 1) * 255

def parse_cifar10(save_path, T=Color_jitter):
	"""
	:param label_path:
	:param use_filename: whether using filename as the key or number as the key
	:return:
	Source: https://github.com/carolineeeeeee/automating_requirements
	"""
	if not os.path.isdir(save_path):
		os.mkdir(save_path)

	with open(CIFAR_10_PATH + 'test_batch', 'rb') as fo:
		test_orig_images_dict = pickle.load(fo, encoding='bytes')
		test_orig_data = test_orig_images_dict[b'data']
		test_orig_labels = test_orig_images_dict[b'labels']

	label_files = np.load(CIFAR_10_C_PATH + 'labels.npy')
	m = len(test_orig_labels)
	for n in range(len(label_files)):
		# randomly sample image
		m = random.randint(0, len(test_orig_data)-1)
		orig_img = convert_array_to_img(test_orig_data, m)
		# randomly sample from color jitter, then save
		j = random.randint(0, 99)
		transformed_img = np.array(T(orig_img, j))
		label = label_files[m]
		#print(transformed_img)
		if not os.path.isdir(save_path  + '/' + str(label)):
			os.mkdir(save_path  + '/' + str(label))
		file_name = str(j) + '_' + str(m) + '.jpg'
		cv2.imwrite(save_path + '/' + str(label) + '/' + file_name, transformed_img)
	return test_orig_labels

def parse_cifar10_c(save_path):
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	label_files = np.load(CIFAR_10_C_PATH+'/labels.npy')
	contrast_images = np.load(CIFAR_10_C_PATH+ '/contrast.npy')
	for i in range(len(contrast_images)):
		image = contrast_images[i]
		label = label_files[i]
		if not os.path.isdir(save_path  + '/' + str(label)):
			os.mkdir(save_path  + '/' + str(label))
		file_name = str(i) + '.jpg'
		cv2.imwrite(save_path + '/' + str(label) + '/' + file_name, image)

	return 0


if __name__ == '__main__':
	#print(torch.cuda.is_available())
	#exit()
	#RQ3 = 'RQ3_images'
	#CIFAR_10_C = 'CIFAR_10_C'
	#RQ3_C = 'RQ3_contrast'
	#transformation = entry_128

	for t in [brightness, entry_128, entry_5, contrast]:
		print(t.__name__)
		folder_name = 'RQ3_' + t.__name__
		parse_cifar10(folder_name, T=t)

		transform = transforms.ToTensor()
		dataset = datasets.ImageFolder(folder_name, transform=transform)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

		for model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES:
			model = get_model(model_name)
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			accs = []
			num_epochs = 0
			model = model.to(device)
			for i, (data, target) in enumerate(dataloader):
				data = data.to(device)
				target = target.to(device)
				transformed_result = model(data)
				pred = torch.argmax(transformed_result, dim=1)
				acc = torch.sum(pred == target)
				accs.append(acc/32)
				num_epochs = i
				break
			print(model_name + ' final accuracy: ' + str(sum(accs)/num_epochs))