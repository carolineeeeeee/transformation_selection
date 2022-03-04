from albumentations.augmentations.transforms import *
import albumentations as A
# need python 3.7
# to install this, use pip3= install -U albumentations
# to get emboss and sharpen use pip install git+https://github.com/albumentations-team/albumentations.git
import cv2
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from io import BytesIO
from PIL import Image
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import os
import matlab.engine
os.environ['KMP_DUPLICATE_LIB_OK']='True'

IQA = 'vif/vifvec_release'
IQA_PATH = '/Users/caroline/Desktop/REforML/HVS/image-quality-tools/metrix_mux/metrix/' + IQA + '/'
matlabPyrToolsPath = "/Users/caroline/Desktop/REforML/HVS/image-quality-tools/metrix_mux/metrix/vif/vifvec_release/matlabPyrTools"

#orig_img = cv2.imread(orig_image_path)

#orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) # not needed for the image generation part

# /////////////// Distortions ///////////////
# a few problems:
# 1. how to not center crop (done, all of them works now with one assumption: the image width >= height)
# 2. how to have continueous parameter range (done)
# 3. how to not use tensorflow (done, nothing currently uses tensowflow)


def save_array(dest, arr):
	img = Image.fromarray(arr.astype(np.uint8))
	img.save(dest)

#def gaussian_noise(x, severity=1):
def gaussian_noise(x, c=0.08):
#	c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

	x = np.array(x) / 255.
	return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


#def shot_noise(x, severity=1):
def shot_noise(x, c=60):
#	c = [60, 25, 12, 5, 3][severity - 1]

	x = np.array(x) / 255.
	return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


#def impulse_noise(x, severity=1):
def impulse_noise(x, c=0.03):
#	c = [.03, .06, .09, 0.17, 0.27][severity - 1]

	x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
	return np.clip(x, 0, 1) * 255


#def speckle_noise(x, severity=1):
def speckle_noise(x, c=0.15):
#	c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

	x = np.array(x) / 255.
	return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

# not inlcuded
def fgsm(x, source_net, severity=1):
	c = [8, 16, 32, 64, 128][severity - 1]

	x = V(x, requires_grad=True)
	logits = source_net(x)
	source_net.zero_grad()
	loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
	loss.backward()

	return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))

#def gaussian_blur(x, severity=1):
def gaussian_blur(x, c=3):
#	c = [1, 2, 3, 4, 6][severity - 1]

	x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
	return np.clip(x, 0, 1) * 255

#def glass_blur(x, severity=1):
def glass_blur(x, sigma=0.9, max_delta=3, iterations=4):
	# sigma, max_delta, iterations
#	c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
	c = (sigma, max_delta, iterations)
	x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

	# locally shuffle pixels
	for i in range(c[2]):
		for h in range(224 - c[1], c[1], -1):
			for w in range(224 - c[1], c[1], -1):
				dx, dy = np.random.randint(-c[1], c[1], size=(2,))
				h_prime, w_prime = h + dy, w + dx
				# swap
				x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

	return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255

def disk(radius, alias_blur=0.1, dtype=np.float32):
	if radius <= 8:
		L = np.arange(-8, 8 + 1)
		ksize = (3, 3)
	else:
		L = np.arange(-radius, radius + 1)
		ksize = (5, 5)
	X, Y = np.meshgrid(L, L)
	aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
	aliased_disk /= np.sum(aliased_disk)

	# supersample disk to antialias
	return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

#def defocus_blur(x, severity=1):
def defocus_blur(x, c=(3, 0.1)):
#	c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

	x = np.array(x) / 255.
	kernel = disk(radius=c[0], alias_blur=c[1])

	channels = []
	for d in range(3):
		channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
	channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

	return np.clip(channels, 0, 1) * 255

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

#def motion_blur(x, severity=1):
def motion_blur(x, c=(10,3)):
#	c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

	output = BytesIO()
	x.save(output, format='PNG')
	x = MotionImage(blob=output.getvalue())

	x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

	x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
					 cv2.IMREAD_UNCHANGED)

	if x.shape != (224, 224):
		return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
	else:  # greyscale to RGB
		return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def clipped_zoom(img, zoom_factor):
	h,w,c = img.shape
	# ceil crop height(= crop width)
	ch = int(np.ceil(h / zoom_factor))
	top = (h - ch) // 2
	new_img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
	# trim off any extra pixels
	trim_top = (new_img.shape[0] - h) // 2
	result_image = new_img[trim_top:trim_top + h, trim_top:trim_top + h]
	
	if result_image.shape == img.shape:
		return result_image
	else:
		# pad with zeros for addition later 
		img[trim_top:trim_top + h, trim_top:trim_top + h] = result_image
		return img

#def zoom_blur(x, severity=1):
def zoom_blur(x, max_zoom=1.11, step=0.01):
	#c = [np.arange(1, 1.11, 0.01),
	#	 np.arange(1, 1.16, 0.01),
	#	 np.arange(1, 1.21, 0.02),
	#	 np.arange(1, 1.26, 0.02),
	#	 np.arange(1, 1.31, 0.03)][severity - 1]
	c = np.arange(1, max_zoom, step)

	h,w = x.size
	x = (np.array(x) / 255.).astype(np.float32)
	squared_x = np.zeros((max(h,w), max(h,w), 3)) 
	mid_point = h//2 # let's assume h is bigger
	print(h//2,h//2-w//2,h//2+(w-w//2) )
	squared_x[h//2-w//2:h//2+(w-w//2), 0:h] = x # has to be in the middle
	out = np.zeros_like(squared_x)

	for zoom_factor in c:
		out += clipped_zoom(squared_x, zoom_factor)

	squared_x = (squared_x + out) / (len(c) + 1)
	result = np.clip(squared_x, 0, 1) * 255
	return result[h//2-w//2:h//2+(w-w//2), 0:h] 


# def barrel(x, severity=1):
#     c = [(0,0.03,0.03), (0.05,0.05,0.05), (0.1,0.1,0.1),
#          (0.2,0.2,0.2), (0.1,0.3,0.6)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#
#     x = WandImage(blob=output.getvalue())
#     x.distort('barrel', c)
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def plasma_fractal(mapsize=256, wibbledecay=3):
	"""
	Generate a heightmap using diamond-square algorithm.
	Return square 2d array, side length 'mapsize', of floats in range 0-255.
	'mapsize' must be a power of two.
	"""
	assert (mapsize & (mapsize - 1) == 0)
	maparray = np.empty((mapsize, mapsize), dtype=np.float_)
	maparray[0, 0] = 0
	stepsize = mapsize
	wibble = 100

	def wibbledmean(array):
		return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

	def fillsquares():
		"""For each square of points stepsize apart,
		   calculate middle value as mean of points + wibble"""
		cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
		squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
		squareaccum += np.roll(squareaccum, shift=-1, axis=1)
		maparray[stepsize // 2:mapsize:stepsize,
		stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

	def filldiamonds():
		"""For each diamond of points stepsize apart,
		   calculate middle value as mean of points + wibble"""
		mapsize = maparray.shape[0]
		drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
		ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
		ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
		lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
		ltsum = ldrsum + lulsum
		maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
		tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
		tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
		ttsum = tdrsum + tulsum
		maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

	while stepsize >= 2:
		fillsquares()
		filldiamonds()
		stepsize //= 2
		wibble /= wibbledecay

	maparray -= maparray.min()
	return maparray / maparray.max()

#def fog(x, severity=1):
def fog(x, c=(1.5, 2)):
#	c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

	x = np.array(x) / 255.
	max_val = x.max()
	print(x.shape)
	h,w,ch = x.shape
	plasma_size = w.bit_length()
	x += c[0] * plasma_fractal(mapsize=2**plasma_size, wibbledecay=c[1])[:h, :w][..., np.newaxis]
	return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

#def frost(x, severity=1):
def frost(x, c=(1, 0.1)):
	idx = np.random.randint(5)
	filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpeg', 'frost5.jpeg', 'frost6.jpeg'][idx]
	frost = Image.open(filename)

	#print(frost)
	x = np.asarray(x)
	h,w,ch = x.shape
	frost = np.asarray(frost.resize((w, h)))
	# randomly crop and convert to rgb
	frost = frost[..., [2, 1, 0]]
	#x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
	#frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

	return np.clip(c[0] * x + c[1] * frost, 0, 255)

def snow(x, c=(0.1, 0.3, 3, 0.5, 10, 4, 0.8)):
	#c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
	#	 (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
	#	 (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
	#	 (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
	#	 (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
	
	x = np.array(x, dtype=np.float32) / 255.
	h,w,ch = x.shape
	snow_layer = np.random.normal(size=(w,w), loc=c[0], scale=c[1])  # [:2] for monochrome
	snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
	
	snow_layer[snow_layer < c[3]] = 0
	
	snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
	output = BytesIO()
	snow_layer.save(output, format='PNG')
	snow_layer = MotionImage(blob=output.getvalue())

	snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

	snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
							  cv2.IMREAD_UNCHANGED) / 255.
	snow_layer = snow_layer[..., np.newaxis]	
	x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
	snow_layer = snow_layer[w//2-h//2:w//2+(h-h//2), 0:w]
	return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

# not used
def spatter(x, severity=1):
	c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
		 (0.65, 0.3, 3, 0.68, 0.6, 0),
		 (0.65, 0.3, 2, 0.68, 0.5, 0),
		 (0.65, 0.3, 1, 0.65, 1.5, 1),
		 (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
	x = np.array(x, dtype=np.float32) / 255.

	liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

	liquid_layer = gaussian(liquid_layer, sigma=c[2])
	liquid_layer[liquid_layer < c[3]] = 0
	if c[5] == 0:
		liquid_layer = (liquid_layer * 255).astype(np.uint8)
		dist = 255 - cv2.Canny(liquid_layer, 50, 150)
		dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
		_, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
		dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
		dist = cv2.equalizeHist(dist)
		#     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
		#     ker -= np.mean(ker)
		ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
		dist = cv2.filter2D(dist, cv2.CV_8U, ker)
		dist = cv2.blur(dist, (3, 3)).astype(np.float32)

		m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
		m /= np.max(m, axis=(0, 1))
		m *= c[4]

		# water is pale turqouise
		color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
								238 / 255. * np.ones_like(m[..., :1]),
								238 / 255. * np.ones_like(m[..., :1])), axis=2)

		color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
		x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

		return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
	else:
		m = np.where(liquid_layer > c[3], 1, 0)
		m = gaussian(m.astype(np.float32), sigma=c[4])
		m[m < 0.8] = 0
		#         m = np.abs(m) ** (1/c[4])

		# mud brown
		color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
								42 / 255. * np.ones_like(x[..., :1]),
								20 / 255. * np.ones_like(x[..., :1])), axis=2)

		color *= m[..., np.newaxis]
		x *= (1 - m[..., np.newaxis])

		return np.clip(x + color, 0, 1) * 255

def contrast(x, c=0.4):
#def contrast(x, severity=1):
#	c = [0.4, .3, .2, .1, .05][severity - 1]

	x = np.array(x) / 255.
	means = np.mean(x, axis=(0, 1), keepdims=True)
	return np.clip((x - means) * c + means, 0, 1) * 255


#def brightness(x, severity=1):
def brightness(x, c=0.1):
#	c = [.1, .2, .3, .4, .5][severity - 1]

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255


#def saturate(x, severity=1):
def saturate(x, c=(0.3, 0)):
	#c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

	x = np.array(x) / 255.
	x = sk.color.rgb2hsv(x)
	x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
	x = sk.color.hsv2rgb(x)

	return np.clip(x, 0, 1) * 255


#def jpeg_compression(x, severity=1):
def jpeg_compression(x, c=25):
	#c = [25, 18, 15, 10, 7][severity - 1]

	output = BytesIO()
	x.save(output, 'JPEG', quality=c)
	x = Image.open(output)

	return x


#def pixelate(x, severity=1):
def pixelate(x, c=0.6):
	#c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

	x = x.resize((int(224 * c), int(224 * c)), Image.BOX)
	x = x.resize((224, 224), Image.BOX)

	return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
# this is geometric, let's not use it
def elastic_transform(image, severity=1):
	c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
		 (244 * 2, 244 * 0.08, 244 * 0.2),
		 (244 * 0.05, 244 * 0.01, 244 * 0.02),
		 (244 * 0.07, 244 * 0.01, 244 * 0.02),
		 (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

	image = np.array(image, dtype=np.float32) / 255.
	shape = image.shape
	shape_size = shape[:2]

	# random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size,
					   [center_square[0] + square_size, center_square[1] - square_size],
					   center_square - square_size])
	pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

	dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
				   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
	dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
				   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
	dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
	indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
	return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////

#test_transformations = jpeg_compression
#transformed_img = Image.fromarray(test_transformations(orig_img).astype(np.uint8))
#transformed_img = test_transformations(orig_img)
#transformed_img.save('transformed_jpeg.jpeg')
'''
l_transformations = [jpeg_compression, impulse_noise, shot_noise, defocus_blur, glass_blur, gaussian_blur, motion_blur, snow, frost, fog, contrast, brightness, pixelate]
z
for test_transformations in l_transformations:
	transformed_img = test_transformations(orig_img)
	name = "transformation_" + test_transformations.__name__ + "_t.jpeg"
	try:
		transformed_img.save(name)
	except:
		save_array(name, transformed_img)
'''
#cv2.imwrite('transformed.jpeg', )
#exit()

# /////////////// Begin Albumentation ///////////////


# note that they use different image libraries

orig_image_path = "frankfurt_000000_000294_leftImg8bit.png" 
orig_img = cv2.imread(orig_image_path)
# below are the transformations and an example of how to run them
# for the experiment, I think it's eaiser to create a pickle file to save a dict of image names to replays because some of the parameters are really long
'''
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter
color_jitter = A.ReplayCompose([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True)])
print("-----------color_jitter-----------")
transformed_img = color_jitter(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_colorjitter.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("brightness: " +  str(transformed_img['replay']['transforms'][0]['params']['brightness']))
print("contrast: " +  str(transformed_img['replay']['transforms'][0]['params']['contrast']))
print("saturation: " +  str(transformed_img['replay']['transforms'][0]['params']['saturation']))
print("hue: " +  str(transformed_img['replay']['transforms'][0]['params']['hue']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift
RGB_Shift = A.ReplayCompose([RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=True)])
print("-----------RGB_Shift-----------")
transformed_img = RGB_Shift(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_rgbshift.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("r_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['r_shift']))
print("g_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['g_shift']))
print("b_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['b_shift']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast
Random_Brightness_Contrast = A.ReplayCompose([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True)])
print("-----------Random_Brightness_Contrast-----------")
transformed_img = Random_Brightness_Contrast(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_random_brightness_contrast.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("alpha: " +  str(transformed_img['replay']['transforms'][0]['params']['alpha']))
print("beta: " +  str(transformed_img['replay']['transforms'][0]['params']['beta']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGamma
Random_Gamma = A.ReplayCompose([RandomGamma(gamma_limit=(0, 300), always_apply=True)])
print("-----------Random_Gamma-----------")
transformed_img = Random_Gamma(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_gama.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("gamma: " +  str(transformed_img['replay']['transforms'][0]['params']['gamma']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSunFlare
Random_Sun_Flare = A.ReplayCompose([RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=1000, src_color=(255, 255, 255), always_apply=True)])
print("-----------Random_Sun_Flare-----------")
transformed_img = Random_Sun_Flare(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_randomSunFlare.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("circles: " +  str(transformed_img['replay']['transforms'][0]['params']['circles']))
print("flare_center_x: " +  str(transformed_img['replay']['transforms'][0]['params']['flare_center_x']))
print("flare_center_y: " +  str(transformed_img['replay']['transforms'][0]['params']['flare_center_y']))

To_Gray = ToGray(p=1)

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GlassBlur
glass_blur = A.ReplayCompose([GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=True, mode='fast')])
print("-----------glass_blur-----------")
transformed_img = glass_blur(image=orig_img)
# writing the image (commented out)

cv2.imwrite('transformed_glassblur.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("ksize: " +  str(transformed_img['replay']['transforms'][0]['params']['ksize']))
print("dxy: " +  str(transformed_img['replay']['transforms'][0]['params']['dxy']))
#print("flare_center_y: " +  str(transformed_img['replay']['transforms'][0]['params']['flare_center_y']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomFog
random_fog = A.ReplayCompose([RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=True)])
print("-----------random_fog-----------")
transformed_img = random_fog(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_randomFog.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("haze_list: " +  str(transformed_img['replay']['transforms'][0]['params']['haze_list']))
print("fog_coef: " +  str(transformed_img['replay']['transforms'][0]['params']['fog_coef']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomRain
random_rain = A.ReplayCompose([RandomRain(slant_lower=10, slant_upper=10, drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.5, rain_type=None, always_apply=True)])
print("-----------random_rain-----------")
transformed_img = random_rain(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_randomRain.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("drop_length: " +  str(transformed_img['replay']['transforms'][0]['params']['drop_length']))
print("rain_drops: " +  str(transformed_img['replay']['transforms'][0]['params']['rain_drops']))


#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomSnow
random_snow = A.ReplayCompose([RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=True)])
print("-----------random_snow-----------")
transformed_img = random_snow(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_albu.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
#print("drop_length: " +  str(transformed_img['replay']['transforms'][0]['params']['drop_length']))
#print("rain_drops: " +  str(transformed_img['replay']['transforms'][0]['params']['rain_drops']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomShadow
random_shadow = A.ReplayCompose([RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=True)])
print("-----------random_shadow-----------")
transformed_img = random_shadow(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("vertices_list: " +  str(transformed_img['replay']['transforms'][0]['params']['vertices_list']))
#print("rain_drops: " +  str(transformed_img['replay']['transforms'][0]['params']['rain_drops']))
'''
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur
blur = A.ReplayCompose([Blur (blur_limit=[3, 25], always_apply=True)])
print("-----------blur-----------")
transformed_img = blur(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_blur.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("ksize: " +  str(transformed_img['replay']['transforms'][0]['params']['ksize']))
'''
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.CLAHE
clahe = A.ReplayCompose([CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True)])
print("-----------clahe-----------")
transformed_img = clahe(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("clip_limit: " +  str(transformed_img['replay']['transforms'][0]['params']['clip_limit']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelDropout
channel_dropout = A.ReplayCompose([ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=True)])
print("-----------channel_dropout-----------")
transformed_img = channel_dropout(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("channels_to_drop: " +  str(transformed_img['replay']['transforms'][0]['params']['channels_to_drop']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ChannelShuffle
channel_shuffle = A.ReplayCompose([ChannelShuffle(always_apply=True)])
print("-----------channel_shuffle-----------")
transformed_img = channel_shuffle(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("channels_shuffled: " +  str(transformed_img['replay']['transforms'][0]['params']['channels_shuffled']))

# https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Downscale
downscale = A.ReplayCompose([Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=True)])
print("-----------downscale-----------")
transformed_img = downscale(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_downScale.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("scale: " +  str(transformed_img['replay']['transforms'][0]['params']['scale']))
print("interpolation: " +  str(transformed_img['replay']['transforms'][0]['params']['interpolation']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Emboss
emboss = A.ReplayCompose([Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=True) ])
print("-----------emboss-----------")
transformed_img = emboss(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("emboss_matrix: " +  str(transformed_img['replay']['transforms'][0]['params']['emboss_matrix']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Equalize
equalize = A.ReplayCompose([Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=True)]) # no parameter


#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.FromFloat
from_float = A.ReplayCompose([FromFloat(dtype='uint16', max_value=None, always_apply=True)]) # no parameter

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussNoise
gaussian_noise = A.ReplayCompose([GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=True)])
print("-----------gaussian_noise-----------")
transformed_img = gaussian_noise(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_gaussiannoise.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("gauss: " +  str(transformed_img['replay']['transforms'][0]['params']['gauss']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
gaussian_blur = A.ReplayCompose([GaussianBlur(blur_limit=(21, 21), sigma_limit=1, always_apply=True)])
print("-----------gaussian_blur-----------")
transformed_img = gaussian_blur(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_gaussianblur.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("ksize: " +  str(transformed_img['replay']['transforms'][0]['params']['ksize']))
print("sigma: " +  str(transformed_img['replay']['transforms'][0]['params']['sigma']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HueSaturationValue
hue_saturation = A.ReplayCompose([HueSaturationValue(hue_shift_limit=100, sat_shift_limit=100, val_shift_limit=100, always_apply=True)])
print("-----------hue_saturation-----------")
transformed_img = hue_saturation(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("hue_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['hue_shift']))
print("sat_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['sat_shift']))
print("val_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['val_shift']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ISONoise
ISO_noise = A.ReplayCompose([ISONoise(color_shift=(0.01, 1), intensity=(0.5, 5), always_apply=True)])
print("-----------ISO_noise-----------")
transformed_img = ISO_noise(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("color_shift: " +  str(transformed_img['replay']['transforms'][0]['params']['color_shift']))
print("intensity: " +  str(transformed_img['replay']['transforms'][0]['params']['intensity']))
print("random_state: " +  str(transformed_img['replay']['transforms'][0]['params']['random_state']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ImageCompression
image_compression = A.ReplayCompose([ImageCompression (quality_lower=99, quality_upper=100, always_apply=True)])
print("-----------image_compression-----------")
transformed_img = image_compression(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_imagecompression.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("quality: " +  str(transformed_img['replay']['transforms'][0]['params']['quality']))
print("image_type: " +  str(transformed_img['replay']['transforms'][0]['params']['image_type']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MedianBlur
median_blur = A.ReplayCompose([MedianBlur(blur_limit=7, always_apply=True)])
print("-----------median_blur-----------")
transformed_img = median_blur(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_medianblur.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("ksize: " +  str(transformed_img['replay']['transforms'][0]['params']['ksize']))
'''
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MotionBlur
motion_blur = A.ReplayCompose([MotionBlur(blur_limit=3,always_apply=True)])
print("-----------motion_blur-----------")
transformed_img = motion_blur(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_motionblur.jpeg', transformed_img['image'])
# to get the parameters that are actually used
print(transformed_img['replay']['transforms'][0]['params'])
print("kernel: " +  str(transformed_img['replay']['transforms'][0]['params']['kernel']))
'''
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.MultiplicativeNoise
multiplicative_noise = A.ReplayCompose([MultiplicativeNoise(multiplier=(0.7, 1.13), per_channel=False, elementwise=False, always_apply=True)])
print("-----------multiplicative_noise-----------")
transformed_img = multiplicative_noise(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("multiplier: " +  str(transformed_img['replay']['transforms'][0]['params']['multiplier']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Posterize
posterize = A.ReplayCompose([Posterize(num_bits=2, always_apply=True)])
print("-----------posterize-----------")
transformed_img = posterize(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("num_bits: " +  str(transformed_img['replay']['transforms'][0]['params']['num_bits']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Sharpen
sharpen = A.ReplayCompose([Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True)])
print("-----------sharpen-----------")
transformed_img = sharpen(image=orig_img)
# writing the image (commented out)
#cv2.imwrite('transformed.jpeg', transformed_img['image'])
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
print("sharpening_matrix: " +  str(transformed_img['replay']['transforms'][0]['params']['sharpening_matrix']))

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToFloat
#to_float = A.ReplayCompose([ToFloat(max_value=None, always_apply=True)]) # no parameter

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomToneCurve
tone_curve = A.ReplayCompose([RandomToneCurve(scale=0.9, always_apply=True)])
print("-----------tone_curve-----------")
transformed_img = tone_curve(image=orig_img)
# writing the image (commented out)
cv2.imwrite('transformed_tone.jpeg', transformed_img['image'])
'''
# to get the parameters that are actually used
#print(transformed_img['replay']['transforms'][0]['params'])
#print("sharpening_matrix: " +  str(transformed_img['replay']['transforms'][0]['params']['sharpening_matrix']))

# using replay
#image2 = cv2.imread('images/image_3.jpg')
#image2_data = A.ReplayCompose.replay(transformed_img['replay'], image=image2)

# /////////////// End Albumentation ///////////////

# IQA to multiple parameters is hard. Unless we keep the other ones constant and only change one at a time?
# /////////////// Begin IQA ///////////////

orig_image_path = "frankfurt_000000_000294_leftImg8bit.png" 

eng = matlab.engine.start_matlab()
#print(IQA_PATH)
eng.addpath(IQA_PATH, nargout=0)
eng.addpath(matlabPyrToolsPath, nargout=0)
eng.addpath(matlabPyrToolsPath+ '/MEX', nargout=0)
#orig_img = Image.open(orig_image_path)
#orig_img = cv2.imread(orig_image_path)
#transformed_img = jpeg_compression(orig_img, c=150)

#transformed_img = motion_blur(orig_img, c=(10,0.8)) #gaussian_blur(orig_img, c=1.5)
transformed_img = transformed_img['image']
img_grey = cv2.cvtColor(np.asarray(orig_img).astype('float32'), cv2.COLOR_BGR2GRAY)
transformed_grey = cv2.cvtColor(np.asarray(transformed_img).astype('float32'), cv2.COLOR_BGR2GRAY)
#print(img_grey)
#print(transformed_grey)
VIF_value = eng.vifvec(matlab.double(np.asarray(img_grey).tolist()), matlab.double(np.asarray(transformed_grey).tolist()))
print(VIF_value)
eng.quit()
# /////////////// End IQA ///////////////



