from re import S
import torchvision
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
	#print(image_array[:1024].shape)
	#print(image_array[1024:2048].shape)
	#print(image_array[2048:].shape)
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
	#arguments = str(transformed_img['replay']['transforms'][0]['params']['contrast']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['saturation']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['hue'])
	return transformed_img['image']

def Sharpen_img(x):
	sharpen = A.ReplayCompose([Sharpen(always_apply=True)])
	sharpened_img = sharpen(image=x)
	return sharpened_img['image']

def brightness(x, i):
	#c = [.1, .2, .3, .4, .5]
	c = np.linspace(0, 1, 100)	 

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 2] = np.clip(x[:, :, 2] + c[i], 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255#, c[i]

def Random_Gamma (x):
	#c = [(80, 120), (90, 130), (100, 140), (120, 150), (130, 160)]
	RG = A.ReplayCompose([RandomGamma(gamma_limit=(0, 300), always_apply=True)])
	transformed_img = RG (image=x)
	# to get the parameters that are actually used
	#arguments = str(transformed_img['replay']['transforms'][0]['params']['gamma'])
	
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
#def contrast(x, severity=1):
	#c = [0.4, .3, .2, .1, .05]
	c = np.linspace(0.01, 0.9, 100)

	x = np.array(x) / 255.
	means = np.mean(x, axis=(0, 1), keepdims=True)
	return np.clip((x - means) * c[i] + means, 0, 1) * 255

def parse_cifar10(save_path, T=Color_jitter):
	"""
	:param label_path:
	:param use_filename: whether using filename as the key or number as the key
	:return:
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

'''

def load_cifar10_data(data_path: pathlib2.Path):
	"""
	Read labels information from a dataset
	Labels.csv is a mapping for filenames and labels.
	:param data_path: directory containing all images and a labels.csv
	:return:
	"""
	if (data_path / 'labels.csv').exists():
		df = pd.read_csv(data_path / 'labels.csv', index_col=0)   # should contain 2 columns: filename, label
		df['original_filename'] = df['filename']
		df['original_path'] = df['original_filename'].apply(lambda filename: os.path.join(str(data_path), filename))
		return df
	else:
		raise OSError(f"label file not found, labels.csv should be present in {data_path}")

def run_model(model_name: str, bootstrap_df: pd.DataFrame, cpu: bool = False, batch_size: int = 10,
			  dataset_class: Dataset = Cifar10Dataset):
	model = get_model(model_name, pretrained=True, val=True)
	device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
	logger.info(f"Device: {str(device)}")
	model.to(device)
	batches = bootstrap_df['bootstrap_iter_id'].unique()
	pbar = tqdm(total=len(bootstrap_df))
	prediction_records = []
	err_top_5 = 0
	err_top_1 = 0
	total_image = 0
	for bootstrap_iter_id in batches:
		df = bootstrap_df[bootstrap_df['bootstrap_iter_id'] == bootstrap_iter_id]
		dataset = dataset_class(df)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
												 num_workers=1)

		for i, data in enumerate(dataloader):
			original_images = data['original_image'].to(device)
			new_images = data['new_image'].to(device)

			labels = data['label'].to(device)
			transformed_result = model(new_images)
			original_result = model(original_images)

			new_pred = torch.argmax(transformed_result, dim=1)
			original_pred = torch.argmax(original_result, dim=1)

			actual_batch_size = len(new_pred)
			top_1_error_count = actual_batch_size - int(torch.sum(labels == new_pred))
			top_5_error_count = actual_batch_size - int(
				torch.sum(torch.sum(torch.eq(labels.unsqueeze(1).repeat(1, 5), new_pred.topk(5).indices),
									dim=1)))
			total_image += actual_batch_size
			new_pred = new_pred.tolist()
			original_pred = original_pred.tolist()
			for k, label in enumerate(labels.tolist()):
				prediction_records.append({
					'model_name': model_name,
					'bootstrap_iter_id': bootstrap_iter_id,
					'dataload_itr_id': i,
					'within_iter_id': k,
					'label': label,
					'new_prediction': new_pred[k],
					'original_prediction': original_pred[k],
					'transformation': data['transformation'][k],
					'original_filename': data['original_filename'][k],
					'new_filename': data['new_filename'][k],
					'original_path': data['original_path'][k],
					'new_path': data['new_path'][k],
					'vd_score': float(data['vd_score'][k]),
				})
			pbar.set_postfix({f'Iteration': bootstrap_iter_id})
			pbar.update(actual_batch_size)
	records_df = pd.DataFrame(data=prediction_records)
	return records_df

'''


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
		#parse_cifar10(RQ3)
		parse_cifar10(folder_name, T=t)
		#parse_cifar10_c(CIFAR_10_C)
		#exit()
		transform = transforms.ToTensor()
		dataset = datasets.ImageFolder(folder_name, transform=transform)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

		#images, labels = next(iter(dataloader))
		#print(images[0])
		#cv2.imshow(images[0])
		
		#testset = torchvision.datasets.CIFAR10(root='./RQ3_images', train=False,
		#                                   download=True)
		#testloader = torch.utils.data.DataLoader(testset, batch_size=100,
		#                                        shuffle=False, num_workers=2)
		for model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES:
			model = get_model(model_name)
			#print(model_name)
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
				#print(i, acc/32)
				accs.append(acc/32)
				num_epochs = i
				break
			print(model_name + ' final accuracy: ' + str(sum(accs)/num_epochs))