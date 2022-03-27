import pandas as pd
import json

# transformations from https://albumentations.ai/docs/getting_started/transforms_and_targets/
# accessed March 15, 2022
col_names = ['Name',
             'Source',
             'Description',
             'Patameters']

Transformations = []

# AdvancedBlur
AdvancedBlur = [ 'Advanced Blur', 'albumentations', 'Blur the input image using a Generalized Normal filter with a randomly selected parameters. This transform also adds multiplicative noise to generated kernel before convolution.']
AdvancedBlur_parameters = {}
AdvancedBlur_parameters['blur_limit'] = 'maximum Gaussian kernel size for blurring the input image. Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma as round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1. If set single value blur_limit will be in range (0, blur_limit). Default: (3, 7).'
AdvancedBlur_parameters['sigmaX_limit'] = 'Gaussian kernel standard deviation. Must be in range [0, inf). If set single value sigmaX_limit will be in range (0, sigma_limit). If set to 0 sigma will be computed as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8. Default: 0.'
AdvancedBlur_parameters['sigmaY_limit'] = 'Same as sigmaY_limit for another dimension.'
AdvancedBlur_parameters['rotate_limit'] = 'Range from which a random angle used to rotate Gaussian kernel is picked. If limit is a single int an angle is picked from (-rotate_limit, rotate_limit). Default: (-90, 90).'
AdvancedBlur_parameters['beta_limit'] = 'Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution tails heavier than normal, values above 1.0 make it lighter than normal. Default: (0.5, 8.0).'
AdvancedBlur_parameters['noise_limit'] = 'Multiplicative factor that control strength of kernel noise. Must be positive and preferably centered around 1.0. If set single value noise_limit will be in range (0, noise_limit). Default: (0.75, 1.25).'
AdvancedBlur.append(json.dumps(AdvancedBlur_parameters))
Transformations.append(AdvancedBlur)

# Blur
Blur = ['Blur', 'albumentations', 'Blur the input image using a random-sized kernel.']
Blur_parameters = {}
Blur_parameters['blur_limit'] = 'maximum kernel size for blurring the input image. Should be in range [3, inf). Default: (3, 7).'
Blur.append(json.dumps(Blur_parameters))
Transformations.append(Blur)

# CLAHE
CLAHE = ['CLAHE', 'albumentations', 'Apply Contrast Limited Adaptive Histogram Equalization to the input image.']
CLAHE_parameters = {}
CLAHE_parameters['clip_limit'] = 'upper threshold value for contrast limiting. If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).'
CLAHE_parameters['tile_grid_size'] = 'size of grid for histogram equalization. Default: (8, 8).'
CLAHE.append(json.dumps(CLAHE_parameters))
Transformations.append(CLAHE)

# ChannelDropout
ChannelDropout = ['Channel Dropout', 'albumentations', 'Randomly Drop Channels in the input Image.']
ChannelDropout_parameters = {}
ChannelDropout_parameters['channel_drop_range'] = 'range from which we choose the number of channels to drop.'
ChannelDropout_parameters['fill_value'] = 'pixel value for the dropped channel.'
ChannelDropout.append(json.dumps(ChannelDropout_parameters))
Transformations.append(ChannelDropout)

# ChannelShuffle
ChannelShuffle = ['Channel Shuffle', 'albumentations', 'Randomly rearrange channels of the input RGB image.']
ChannelShuffle_parameters = {}
ChannelShuffle.append(json.dumps(ChannelShuffle_parameters))
Transformations.append(ChannelShuffle)

# ColorJitter
ColorJitter = ['Color Jitter', 'albumentations', 'Randomly changes the brightness, contrast, and saturation of an image.']
ColorJitter_parameters = {}
ColorJitter_parameters['brightness'] = 'How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.'
ColorJitter_parameters['contrast'] = 'How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.'
ColorJitter_parameters['saturation'] = 'How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.'
ColorJitter_parameters['hue'] = 'How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.'
ColorJitter.append(json.dumps(ColorJitter_parameters))
Transformations.append(ColorJitter)

# Downscale
Downscale = ['Downscale', 'albumentations', 'Decreases image quality by downscaling and upscaling back.']
Downscale_parameters = {}
Downscale_parameters['scale_min'] = 'lower bound on the image scale. Should be < 1.'
Downscale_parameters['scale_max'] = 'lower bound on the image scale. Should be .'
Downscale_parameters['interpolation'] = 'cv2 interpolation method. cv2.INTER_NEAREST by default'
Downscale.append(json.dumps(Downscale_parameters))
Transformations.append(Downscale)

# Emboss
Emboss = ['Emboss', 'albumentations', 'Emboss the input image and overlays the result with the original image.']
Emboss_parameters = {}
Emboss_parameters['alpha'] = 'range to choose the visibility of the embossed image. At 0, only the original image is visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).'
Emboss_parameters['strength'] = 'strength range of the embossing. Default: (0.2, 0.7).'
Emboss.append(json.dumps(Emboss_parameters))
Transformations.append(Emboss)

# Equalize
Equalize = ['Equalize', 'albumentations', 'Equalize the image histogram.']
Equalize_parameters = {}
Equalize_parameters['mode'] = '{"cv", "pil"}. Use OpenCV or Pillow equalization method.'
Equalize_parameters['by_channels'] = 'If True, use equalization by channels separately, else convert image to YCbCr representation and use equalization by Y channel.'
Equalize_parameters['mask'] = 'If given, only the pixels selected by the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable. Function signature must include image argument.'
Equalize_parameters['mask_params'] = 'Params for mask function.'
Equalize.append(json.dumps(Equalize_parameters))
Transformations.append(Equalize)

# FDA
FDA = ['FDA', 'albumentations', 'Fourier Domain Adaptation. Simple "style transfer".']
FDA_parameters = {}
FDA_parameters['reference_images'] = 'List of file paths for reference images or list of reference images.'
FDA_parameters['beta_limit'] = 'coefficient beta from paper. Recommended less 0.3.'
FDA_parameters['read_fn'] = 'Used-defined function to read image. Function should get image path and return numpy array of image pixels.'
FDA.append(json.dumps(FDA_parameters))
Transformations.append(FDA)

# FancyPCA
FancyPCA = ['Fancy PCA', 'albumentations', 'Augment RGB image using FancyPCA from Krizhevsky"s paper "ImageNet Classification with Deep Convolutional Neural Networks"']
FancyPCA_parameters = {}
FancyPCA_parameters['alpha'] = 'how much to perturb/scale the eigen vecs and vals. scale is samples from gaussian distribution (mu=0, sigma=alpha)'
FancyPCA.append(json.dumps(FancyPCA_parameters))
Transformations.append(FancyPCA)

# FromFloat
FromFloat = ['From Float', 'albumentations', 'Take an input array where all values should lie in the range [0, 1.0], multiply them by max_value and then cast the resulted value to a type specified by dtype. If max_value is None the transform will try to infer the maximum value for the data type from the dtype argument.']
FromFloat_parameters = {}
FromFloat_parameters['max_value'] = 'maximum possible input value. Default: None.'
FromFloat_parameters['dtype'] = 'data type of the output.'
FromFloat.append(json.dumps(FromFloat_parameters))
Transformations.append(FromFloat)

# GaussNoise
GaussNoise = ['Gauss Noise', 'albumentations', 'Apply gaussian noise to the input image.']
GaussNoise_parameters = {}
GaussNoise_parameters['var_limit'] = 'variance range for noise. If var_limit is a single float, the range will be (0, var_limit). Default: (10.0, 50.0).'
GaussNoise_parameters['mean'] = 'mean of the noise. Default: 0'
GaussNoise_parameters['per_channel'] = 'if set to True, noise will be sampled for each channel independently. Otherwise, the noise will be sampled once for all channels. Default: True'
GaussNoise.append(json.dumps(GaussNoise_parameters))
Transformations.append(GaussNoise)

# GaussianBlur
GaussianBlur = ['Gaussian Blur', 'albumentations', 'Blur the input image using a Gaussian filter with a random kernel size.']
GaussianBlur_parameters = {}
GaussianBlur_parameters['blur_limit'] = 'maximum Gaussian kernel size for blurring the input image. Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma as round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1. If set single value blur_limit will be in range (0, blur_limit). Default: (3, 7).'
GaussianBlur_parameters['sigma_limit'] = 'Gaussian kernel standard deviation. Must be in range [0, inf). If set single value sigma_limit will be in range (0, sigma_limit). If set to 0 sigma will be computed as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8. Default: 0.'
GaussianBlur.append(json.dumps(GaussianBlur_parameters))
Transformations.append(GaussianBlur)

# GlassBlur
GlassBlur = ['Glass Blur', 'albumentations', 'Apply glass noise to the input image.']
GlassBlur_parameters = {}
GlassBlur_parameters['sigma'] = 'standard deviation for Gaussian kernel.'
GlassBlur_parameters['max_delta'] = 'max distance between pixels which are swapped.'
GlassBlur_parameters['iterations'] = 'number of repeats. Should be in range [1, inf). Default: (2).'
GlassBlur_parameters['mode'] = 'mode of computation: fast or exact. Default: "fast".'
GlassBlur.append(json.dumps(GlassBlur_parameters))
Transformations.append(GlassBlur)

# HistogramMatching
HistogramMatching = ['Histogram Matching', 'albumentations', 'Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches the histogram of the reference image. If the images have multiple channels, the matching is done independently for each channel, as long as the number of channels is equal in the input image and the reference.']
HistogramMatching_parameters = {}
HistogramMatching_parameters['reference_images'] = 'List of file paths for reference images or list of reference images.'
HistogramMatching_parameters['blend_ratio'] = 'Tuple of min and max blend ratio. Matched image will be blended with original with random blend factor for increased diversity of generated images.'
HistogramMatching_parameters['read_fn'] = 'Used-defined function to read image. Function should get image path and return numpy array of image pixels.'
HistogramMatching.append(json.dumps(HistogramMatching_parameters))
Transformations.append(HistogramMatching)

# HueSaturationValue
HueSaturationValue = ['Hue Saturation Value', 'albumentations', 'Randomly change hue, saturation and value of the input image.']
HueSaturationValue_parameters = {}
HueSaturationValue_parameters['hue_shift_limit'] = 'range for changing hue. If hue_shift_limit is a single int, the range will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).'
HueSaturationValue_parameters['sat_shift_limit'] = 'range for changing saturation. If sat_shift_limit is a single int, the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).'
HueSaturationValue_parameters['val_shift_limit'] = 'range for changing value. If val_shift_limit is a single int, the range will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).'
HueSaturationValue.append(json.dumps(HueSaturationValue_parameters))
Transformations.append(HueSaturationValue)

# ISONoise
ISONoise = ['ISO Noise', 'albumentations', 'Apply camera sensor noise.']
ISONoise_parameters = {}
ISONoise_parameters['color_shift'] = 'variance range for color hue change. Measured as a fraction of 360 degree Hue angle in HLS colorspace.'
ISONoise_parameters['intensity'] = 'Multiplicative factor that control strength of color and luminace noise.'
ISONoise.append(json.dumps(ISONoise_parameters))
Transformations.append(ISONoise)

# ImageCompression
ImageCompression = ['Image Compression', 'albumentations', 'Decrease Jpeg, WebP compression of an image.']
ImageCompression_parameters = {}
ImageCompression_parameters['quality_lower'] = 'lower bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.'
ImageCompression_parameters['quality_upper'] = 'upper bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.'
ImageCompression_parameters['compression_type'] = 'should be ImageCompressionType.JPEG or ImageCompressionType.WEBP. Default: ImageCompressionType.JPEG'
ImageCompression.append(json.dumps(ImageCompression_parameters))
Transformations.append(ImageCompression)

# InvertImg
InvertImg = ['Invert Img', 'albumentations', 'Invert the input image by subtracting pixel values from 255.']
InvertImg_parameters = {}
InvertImg.append(json.dumps(InvertImg_parameters))
Transformations.append(InvertImg)

# MedianBlur
MedianBlur = ['Median Blur', 'albumentations', 'Blur the input image using a median filter with a random aperture linear size.']
MedianBlur_parameters = {}
MedianBlur_parameters['blur_limit'] = 'maximum aperture linear size for blurring the input image. Must be odd and in range [3, inf). Default: (3, 7).'
MedianBlur.append(json.dumps(MedianBlur_parameters))
Transformations.append(MedianBlur)

# MotionBlur
MotionBlur = ['Motion Blur', 'albumentations', 'Apply motion blur to the input image using a random-sized kernel.']
MotionBlur_parameters = {}
MotionBlur_parameters['blur_limit'] = 'maximum aperture linear size for blurring the input image. Must be odd and in range [3, inf). Default: (3, 7).'
MotionBlur.append(json.dumps(MotionBlur_parameters))
Transformations.append(MotionBlur)

# MultiplicativeNoise
MultiplicativeNoise = ['Multiplicative Noise', 'albumentations', 'Multiply image to random number or array of numbers.']
MultiplicativeNoise_parameters = {}
MultiplicativeNoise_parameters['multiplier'] = 'If single float image will be multiplied to this number. If tuple of float multiplier will be in range [multiplier[0], multiplier[1]). Default: (0.9, 1.1).'
MultiplicativeNoise_parameters['per_channel'] = 'If False, same values for all channels will be used. If True use sample values for each channels. Default False.'
MultiplicativeNoise_parameters['elementwise'] = 'If False multiply multiply all pixels in an image with a random value sampled once. If True Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.'
MultiplicativeNoise.append(json.dumps(MultiplicativeNoise_parameters))
Transformations.append(MultiplicativeNoise)

# Normalize
Normalize = ['Normalize', 'albumentations', 'Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)']
Normalize_parameters = {}
Normalize_parameters['mean'] = 'mean values'
Normalize_parameters['std'] = 'std values'
Normalize_parameters['max_pixel_value'] = 'maximum possible pixel value'
Normalize.append(json.dumps(Normalize_parameters))
Transformations.append(Normalize)

# PixelDistributionAdaptation
PixelDistributionAdaptation = ['Pixel Distribution Adaptation', 'albumentations', 'Another naive and quick pixel-level domain adaptation. It fits a simple transform (such as PCA, StandardScaler or MinMaxScaler) on both original and reference image, transforms original image with transform trained on this image and then performs inverse transformation using transform fitted on reference image.']
PixelDistributionAdaptation_parameters = {}
PixelDistributionAdaptation_parameters['reference_images'] = 'List of file paths for reference images or list of reference images.'
PixelDistributionAdaptation_parameters['blend_ratio'] = 'Tuple of min and max blend ratio. Matched image will be blended with original with random blend factor for increased diversity of generated images.'
PixelDistributionAdaptation_parameters['read_fn'] = 'Used-defined function to read image. Function should get image path and return numpy array of image pixels. Usually it\'s default read_rgb_image when images paths are used as reference, otherwise it could be identity function lambda x: x if reference images have been read in advance.'
PixelDistributionAdaptation_parameters['transform_type'] = 'type of transform; "pca", "standard", "minmax" are allowed.'
PixelDistributionAdaptation.append(json.dumps(PixelDistributionAdaptation_parameters))
Transformations.append(PixelDistributionAdaptation)

# Posterize
Posterize = ['Posterize', 'albumentations', 'Reduce the number of bits for each color channel.']
Posterize_parameters = {}
Posterize_parameters['num_bits'] = 'number of high bits. If num_bits is a single value, the range will be [num_bits, num_bits]. Must be in range [0, 8]. Default: 4.'
Posterize.append(json.dumps(Posterize_parameters))
Transformations.append(Posterize)

# RGBShift
RGBShift = ['RGB Shift', 'albumentations', 'Randomly shift values for each channel of the input RGB image.']
RGBShift_parameters = {}
RGBShift_parameters['r_shift_limit'] = 'range for changing values for the red channel. If r_shift_limit is a single int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).'
RGBShift_parameters['g_shift_limit'] = 'range for changing values for the green channel. If g_shift_limit is a single int, the range will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).'
RGBShift_parameters['b_shift_limit'] = 'range for changing values for the blue channel. If b_shift_limit is a single int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).'
RGBShift.append(json.dumps(RGBShift_parameters))
Transformations.append(RGBShift)

# RandomBrightnessContrast
RandomBrightnessContrast = ['Random Brightness Contrast', 'albumentations', 'Randomly change brightness and contrast of the input image.']
RandomBrightnessContrast_parameters = {}
RandomBrightnessContrast_parameters['brightness_limit'] = 'factor range for changing brightness. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).'
RandomBrightnessContrast_parameters['contrast_limit'] = 'factor range for changing contrast. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).'
RandomBrightnessContrast_parameters['brightness_by_max'] = 'If True adjust contrast by image dtype maximum, else adjust contrast by image mean.'
RandomBrightnessContrast.append(json.dumps(RandomBrightnessContrast_parameters))
Transformations.append(RandomBrightnessContrast)

# RandomFog
RandomFog = ['Random Fog', 'albumentations', 'Simulates fog for the image']
RandomFog_parameters = {}
RandomFog_parameters['fog_coef_lower'] = 'lower limit for fog intensity coefficient. Should be in [0, 1] range.'
RandomFog_parameters['fog_coef_upper'] = 'upper limit for fog intensity coefficient. Should be in [0, 1] range.'
RandomFog_parameters['alpha_coef'] = 'transparency of the fog circles. Should be in [0, 1] range.'
RandomFog.append(json.dumps(RandomFog_parameters))
Transformations.append(RandomFog)

# RandomGamma
RandomGamma = ['Random Gamma', 'albumentations', ' Controls the overall brightness of an image.']
RandomGamma_parameters = {}
RandomGamma_parameters['gamma_limit'] = 'If gamma_limit is a single float value, the range will be (-gamma_limit, gamma_limit). Default: (80, 120).'
RandomGamma.append(json.dumps(RandomGamma_parameters))
Transformations.append(RandomGamma)

# Random Rain
RandomRain = ['Random Rain', 'albumentations', 'Adds rain effects.']
RandomRain_parameters = {}
RandomRain_parameters['slant_lower'] = 'should be in range [-20, 20].'
RandomRain_parameters['slant_upper'] = 'should be in range [-20, 20].'
RandomRain_parameters['drop_length'] = 'should be in range [0, 100].'
RandomRain_parameters['drop_width'] = 'should be in range [1, 5].'
RandomRain_parameters['drop_color'] = 'rain lines color.'
RandomRain_parameters['blur_value'] = 'rainy view are blurry'
RandomRain_parameters['brightness_coefficient'] = 'rainy days are usually shady. Should be in range [0, 1].'
RandomRain_parameters['rain_type'] = 'One of [None, "drizzle", "heavy", "torrestial"]'
RandomRain.append(json.dumps(RandomRain_parameters))
Transformations.append(RandomRain)

# RandomShadow
RandomShadow = ['Random Shadow', 'albumentations', 'Simulates shadows for the image']
RandomShadow_parameters = {}
RandomShadow_parameters['shadow_roi'] = 'region of the image where shadows will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].'
RandomShadow_parameters['num_shadows_lower'] = 'Lower limit for the possible number of shadows. Should be in range [0, num_shadows_upper].'
RandomShadow_parameters['num_shadows_upper'] = 'Lower limit for the possible number of shadows. Should be in range [num_shadows_lower, inf].'
RandomShadow_parameters['shadow_dimension'] = 'number of edges in the shadow polygons'
RandomShadow.append(json.dumps(RandomShadow_parameters))
Transformations.append(RandomShadow)

# RandomSnow
RandomSnow = ['Random Snow', 'albumentations', 'Bleach out some pixel values simulating snow.']
RandomSnow_parameters = {}
RandomSnow_parameters['snow_point_lower'] = 'lower_bond of the amount of snow. Should be in [0, 1] range'
RandomSnow_parameters['snow_point_upper'] = 'upper_bond of the amount of snow. Should be in [0, 1] range'
RandomSnow_parameters['brightness_coeff'] = 'larger number will lead to a more snow on the image. Should be >= 0'
RandomSnow.append(json.dumps(RandomSnow_parameters))
Transformations.append(RandomSnow)

# RandomSunFlare
RandomSunFlare = ['Random Sun Flare', 'albumentations', 'Simulates Sun Flare for the image']
RandomSunFlare_parameters = {}
RandomSunFlare_parameters['flare_roi'] = 'region of the image where flare will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].'
RandomSunFlare_parameters['angle_lower'] = 'should be in range [0, angle_upper].'
RandomSunFlare_parameters['angle_upper'] = 'should be in range [angle_lower, 1].'
RandomSunFlare_parameters['num_flare_circles_lower'] = 'lower limit for the number of flare circles. Should be in range [0, num_flare_circles_upper].'
RandomSunFlare_parameters['num_flare_circles_upper'] = 'upper limit for the number of flare circles. Should be in range [num_flare_circles_lower, inf].'
RandomSunFlare_parameters['src_radius'] = ''
RandomSunFlare_parameters['src_color'] = 'color of the flare'
RandomSunFlare.append(json.dumps(RandomSunFlare_parameters))
Transformations.append(RandomSunFlare)

# RandomToneCurve
RandomToneCurve = ['Random Tone Curve', 'albumentations', 'Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.']
RandomToneCurve_parameters = {}
RandomToneCurve_parameters['scale'] = 'standard deviation of the normal distribution. Used to sample random distances to move two control points that modify the image\'s curve. Values should be in range [0, 1]. Default: 0.1'
RandomToneCurve.append(json.dumps(RandomToneCurve_parameters))
Transformations.append(RandomToneCurve)

# RingingOvershoot
RingingOvershoot = ['Ringing Overshoot', 'albumentations', 'Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.']
RingingOvershoot_parameters = {}
RingingOvershoot_parameters['blur_limit'] = 'maximum kernel size for sinc filter. Should be in range [3, inf). Default: (7, 15).'
RingingOvershoot_parameters['cutoff'] = 'range to choose the cutoff frequency in radians. Should be in range (0, np.pi) Default: (np.pi / 4, np.pi / 2).'
RingingOvershoot.append(json.dumps(RingingOvershoot_parameters))
Transformations.append(RingingOvershoot)

# Sharpen
Sharpen = ['Sharpen', 'albumentations', 'Sharpen the input image and overlays the result with the original image.']
Sharpen_parameters = {}
Sharpen_parameters['alpha'] = 'range to choose the visibility of the sharpened image. At 0, only the original image is visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).'
Sharpen_parameters['lightness'] = 'range to choose the lightness of the sharpened image. Default: (0.5, 1.0).'
Sharpen.append(json.dumps(Sharpen_parameters))
Transformations.append(Sharpen)

# Solarize
Solarize = ['Solarize', 'albumentations', 'Invert all pixel values above a threshold.']
Solarize_parameters = {}
Solarize_parameters['threshold'] = 'range for solarizing threshold. If threshold is a single value, the range will be [threshold, threshold]. Default: 128.'
Solarize.append(json.dumps(Solarize_parameters))
Transformations.append(Solarize)

# Superpixels
Superpixels = ['Superpixels', 'albumentations', 'Transform images partially/completely to their superpixel representation.']
Superpixels_parameters = {}
Superpixels_parameters['p_replace'] = 'Defines for any segment the probability that the pixels within that segment are replaced by their average color (otherwise, the pixels are not changed). Examples: * A probability of 0.0 would mean, that the pixels in no segment are replaced by their average color (image is not changed at all). * A probability of 0.5 would mean, that around half of all segments are replaced by their average color. * A probability of 1.0 would mean, that all segments are replaced by their average color (resulting in a voronoi image). Behaviour based on chosen data types for this parameter: * If a float, then that flat will always be used. * If tuple (a, b), then a random probability will be sampled from the interval [a, b] per image.'
Superpixels_parameters['n_segments'] = 'Rough target number of how many superpixels to generate (the algorithm may deviate from this number). Lower value will lead to coarser superpixels. Higher values are computationally more intensive and will hence lead to a slowdown * If a single int, then that value will always be used as the number of segments. * If a tuple (a, b), then a value from the discrete interval [a..b] will be sampled per image.'
Superpixels_parameters['max_size'] = 'Maximum image size at which the augmentation is performed. If the width or height of an image exceeds this value, it will be downscaled before the augmentation so that the longest side matches max_size. This is done to speed up the process. The final output image has the same size as the input image. Note that in case p_replace is below 1.0, the down-/upscaling will affect the not-replaced pixels too. Use None to apply no down-/upscaling.'
Superpixels_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
Superpixels.append(json.dumps(Superpixels_parameters))
Transformations.append(Superpixels)

# TemplateTransform
TemplateTransform = ['Template Transform', 'albumentations', 'Apply blending of input image with specified templates']
TemplateTransform_parameters = {}
TemplateTransform_parameters['templates'] = 'Images as template for transform.'
TemplateTransform_parameters['img_weight'] = 'If single float will be used as weight for input image. If tuple of float img_weight will be in range [img_weight[0], img_weight[1]). Default: 0.5.'
TemplateTransform_parameters['template_weight'] = 'If single float will be used as weight for template. If tuple of float template_weight will be in range [template_weight[0], template_weight[1]). Default: 0.5.'
TemplateTransform_parameters['template_transform'] = 'transformation object which could be applied to template, must produce template the same size as input image.'
TemplateTransform_parameters['name'] = '(Optional) Name of transform, used only for deserialization.'
TemplateTransform.append(json.dumps(TemplateTransform_parameters))
Transformations.append(TemplateTransform)

# ToFloat
ToFloat = ['To Float', 'albumentations', 'Divide pixel values by max_value to get a float32 output array where all values lie in the range [0, 1.0]. If max_value is None the transform will try to infer the maximum value by inspecting the data type of the input image.']
ToFloat_parameters = {}
ToFloat_parameters['max_value'] = 'maximum possible input value. Default: None.'
ToFloat.append(json.dumps(ToFloat_parameters))
Transformations.append(ToFloat)

# ToGray
ToGray = ['To Gray', 'albumentations', 'Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.']
ToGray_parameters = {}
ToGray.append(json.dumps(ToGray_parameters))
Transformations.append(ToGray)

# ToSepia
ToSepia = ['To Sepia', 'albumentations', 'Applies sepia filter to the input RGB image']
ToSepia_parameters = {}
ToSepia.append(json.dumps(ToSepia_parameters))
Transformations.append(ToSepia)

# UnsharpMask
UnsharpMask = ['Unsharp Mask', 'albumentations', 'Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.']
UnsharpMask_parameters = {}
UnsharpMask_parameters['blur_limit'] = 'maximum Gaussian kernel size for blurring the input image. Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma as round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1. If set single value blur_limit will be in range (0, blur_limit). Default: (3, 7).'
UnsharpMask_parameters['sigma_limit'] = 'Gaussian kernel standard deviation. Must be in range [0, inf). If set single value sigma_limit will be in range (0, sigma_limit). If set to 0 sigma will be computed as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8. Default: 0.'
UnsharpMask_parameters['alpha'] = 'range to choose the visibility of the sharpened image. At 0, only the original image is visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).'
UnsharpMask_parameters['threshold'] = 'Value to limit sharpening only for areas with high pixel difference between original image and it\'s smoothed version. Higher threshold means less sharpening on flat areas. Must be in range [0, 255]. Default: 10.'
UnsharpMask.append(json.dumps(UnsharpMask_parameters))
Transformations.append(UnsharpMask)

# Affine
Affine = ['Affine', 'albumentations', 'Augmentation to apply affine transformations to images. Affine transformations involve: - Translation ("move" image on the x-/y-axis) - Rotation - Scaling ("zoom" in/out) - Shear (move one side of the image, turning a square into a trapezoid)']
Affine_parameters = {}
Affine_parameters['scale'] = 'Scaling factor to use, where 1.0 denotes "no change" and 0.5 is zoomed out to 50 percent of the original size. * If a single number, then that value will be used for all images. * If a tuple (a, b), then a value will be uniformly sampled per image from the interval [a, b]. That value will be used identically for both x- and y-axis. * If a dictionary, then it is expected to have the keys x and/or y. Each of these keys can have the same values as described above. Using a dictionary allows to set different values for the two axis and sampling will then happen independently per axis, resulting in samples that differ between the axes.'
Affine_parameters['translate_percent'] = 'Translation as a fraction of the image height/width (x-translation, y-translation), where 0 denotes "no change" and 0.5 denotes "half of the axis size". * If None then equivalent to 0.0 unless translate_px has a value other than None. * If a single number, then that value will be used for all images. * If a tuple (a, b), then a value will be uniformly sampled per image from the interval [a, b]. That sampled fraction value will be used identically for both x- and y-axis. * If a dictionary, then it is expected to have the keys x and/or y. Each of these keys can have the same values as described above. Using a dictionary allows to set different values for the two axis and sampling will then happen independently per axis, resulting in samples that differ between the axes.'
Affine_parameters['translate_px'] = 'Translation in pixels. * If None then equivalent to 0 unless translate_percent has a value other than None. * If a single int, then that value will be used for all images. * If a tuple (a, b), then a value will be uniformly sampled per image from the discrete interval [a..b]. That number will be used identically for both x- and y-axis. * If a dictionary, then it is expected to have the keys x and/or y. Each of these keys can have the same values as described above. Using a dictionary allows to set different values for the two axis and sampling will then happen independently per axis, resulting in samples that differ between the axes.'
Affine_parameters['rotate'] = 'Rotation in degrees (NOT radians), i.e. expected value range is around [-360, 360]. Rotation happens around the center of the image, not the top left corner as in some other frameworks. * If a number, then that value will be used for all images. * If a tuple (a, b), then a value will be uniformly sampled per image from the interval [a, b] and used as the rotation value.'
Affine_parameters['shear'] = 'Shear in degrees (NOT radians), i.e. expected value range is around [-360, 360], with reasonable values being in the range of [-45, 45]. * If a number, then that value will be used for all images as the shear on the x-axis (no shear on the y-axis will be done). * If a tuple (a, b), then two value will be uniformly sampled per image from the interval [a, b] and be used as the x- and y-shear value. * If a dictionary, then it is expected to have the keys x and/or y. Each of these keys can have the same values as described above. Using a dictionary allows to set different values for the two axis and sampling will then happen independently per axis, resulting in samples that differ between the axes.'
Affine_parameters['interpolation'] = 'OpenCV interpolation flag.'
Affine_parameters['mask_interpolation'] = 'OpenCV interpolation flag.'
Affine_parameters['cval'] = 'The constant value to use when filling in newly created pixels. (E.g. translating by 1px to the right will create a new 1px-wide column of pixels on the left of the image). The value is only used when mode=constant. The expected value range is [0, 255] for uint8 images.'
Affine_parameters['cval_mask'] = 'Same as cval but only for masks.'
Affine_parameters['mode'] = 'OpenCV border flag.'
Affine_parameters['fit_output'] = 'Whether to modify the affine transformation so that the whole output image is always contained in the image plane (True) or accept parts of the image being outside the image plane (False). This can be thought of as first applying the affine transformation and then applying a second transformation to "zoom in" on the new image so that it fits the image plane, This is useful to avoid corners of the image being outside of the image plane after applying rotations. It will however negate translation and scaling.'
Affine.append(json.dumps(Affine_parameters))
Transformations.append(Affine)

# CenterCrop
CenterCrop = ['Center Crop', 'albumentations', 'Crop the central part of the input.']
CenterCrop_parameters = {}
CenterCrop_parameters['height'] = 'height of the crop.'
CenterCrop_parameters['width'] = 'width of the crop.'
CenterCrop.append(json.dumps(CenterCrop_parameters))
Transformations.append(CenterCrop)

# CoarseDropout
CoarseDropout = ['Coarse Dropout', 'albumentations', 'CoarseDropout of the rectangular regions in the image.']
CoarseDropout_parameters = {}
CoarseDropout_parameters['max_holes'] = 'Maximum number of regions to zero out.'
CoarseDropout_parameters['max_height'] = 'Maximum height of the hole.'
CoarseDropout_parameters['max_width'] = 'Maximum width of the hole.'
CoarseDropout_parameters['min_holes'] = 'Minimum number of regions to zero out. If None, min_holes is be set to max_holes. Default: None.'
CoarseDropout_parameters['min_height'] = 'Minimum height of the hole. Default: None. If None, min_height is set to max_height. Default: None. If float, it is calculated as a fraction of the image height.'
CoarseDropout_parameters['min_width'] = 'Minimum width of the hole. If None, min_height is set to max_width. Default: None. If float, it is calculated as a fraction of the image width.'
CoarseDropout_parameters['fill_value'] = 'value for dropped pixels.'
CoarseDropout_parameters['mask_fill_value'] = 'fill value for dropped pixels in mask. If None - mask is not affected. Default: None.'
CoarseDropout.append(json.dumps(CoarseDropout_parameters))
Transformations.append(CoarseDropout)

# Crop
Crop = ['Crop', 'albumentations', 'Crop region from image.']
Crop_parameters = {}
Crop_parameters['x_min'] = 'Minimum upper left x coordinate.'
Crop_parameters['y_min'] = 'Minimum upper left y coordinate.'
Crop_parameters['x_max'] = 'Maximum lower right x coordinate.'
Crop_parameters['y_max'] = 'Maximum lower right y coordinate.'
Crop.append(json.dumps(Crop_parameters))
Transformations.append(Crop)

# CropAndPad
CropAndPad = ['Crop And Pad', 'albumentations', 'Crop and pad images by pixel amounts or fractions of image sizes. Cropping removes pixels at the sides (i.e. extracts a subimage from a given full image). Padding adds pixels to the sides (e.g. black pixels). This transformation will never crop images below a height or width of 1.']
CropAndPad_parameters = {}
CropAndPad_parameters['px'] = 'The number of pixels to crop (negative values) or pad (positive values) on each side of the image. Either this or the parameter percent may be set, not both at the same time. * If None, then pixel-based cropping/padding will not be used. * If int, then that exact number of pixels will always be cropped/padded. * If a tuple of two int s with values a and b, then each side will be cropped/padded by a random amount sampled uniformly per image and side from the interval [a, b]. If however sample_independently is set to False, only one value will be sampled per image and used for all sides. * If a tuple of four entries, then the entries represent top, right, bottom, left. Each entry may be a single int (always crop/pad by exactly that value), a tuple of two int s a and b (crop/pad by an amount within [a, b]), a list of int s (crop/pad by a random value that is contained in the list).'
CropAndPad_parameters['percent'] = 'The number of pixels to crop (negative values) or pad (positive values) on each side of the image given as a fraction of the image height/width. E.g. if this is set to -0.1, the transformation will always crop away 10% of the image\'s height at both the top and the bottom (both 10% each), as well as 10% of the width at the right and left. Expected value range is (-1.0, inf). Either this or the parameter px may be set, not both at the same time. * If None, then fraction-based cropping/padding will not be used. * If float, then that fraction will always be cropped/padded. * If a tuple of two float s with values a and b, then each side will be cropped/padded by a random fraction sampled uniformly per image and side from the interval [a, b]. If however sample_independently is set to False, only one value will be sampled per image and used for all sides. * If a tuple of four entries, then the entries represent top, right, bottom, left. Each entry may be a single float (always crop/pad by exactly that percent value), a tuple of two float s a and b (crop/pad by a fraction from [a, b]), a list of float s (crop/pad by a random value that is contained in the list).'
CropAndPad_parameters['pad_mode'] = 'OpenCV border mode.'
CropAndPad_parameters['pad_cval'] = 'The constant value to use if the pad mode is BORDER_CONSTANT. * If number, then that value will be used. * If a tuple of two number s and at least one of them is a float, then a random number will be uniformly sampled per image from the continuous interval [a, b] and used as the value. If both number s are int s, the interval is discrete. * If a list of number, then a random value will be chosen from the elements of the list and used as the value.'
CropAndPad_parameters['pad_cval_mask'] = 'Same as pad_cval but only for masks.'
CropAndPad_parameters['keep_size'] = 'After cropping and padding, the result image will usually have a different height/width compared to the original input image. If this parameter is set to True, then the cropped/padded image will be resized to the input image\'s size, i.e. the output shape is always identical to the input shape.'
CropAndPad_parameters['sample_independently'] = 'If False and the values for px/percent result in exactly one probability distribution for all image sides, only one single value will be sampled from that probability distribution and used for all sides. I.e. the crop/pad amount then is the same for all sides. If True, four values will be sampled independently, one per side.'
CropAndPad_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
CropAndPad.append(json.dumps(CropAndPad_parameters))
Transformations.append(CropAndPad)

# CropNonEmptyMaskIfExists
CropNonEmptyMaskIfExists = ['Crop Non Empty Mask If Exists', 'albumentations', 'Crop area with mask if mask is non-empty, else make random crop.']
CropNonEmptyMaskIfExists_parameters = {}
CropNonEmptyMaskIfExists_parameters['height'] = 'vertical size of crop in pixels'
CropNonEmptyMaskIfExists_parameters['width'] = 'horizontal size of crop in pixels'
CropNonEmptyMaskIfExists_parameters['ignore_values'] = 'values to ignore in mask, 0 values are always ignored (e.g. if background value is 5 set ignore_values=[5] to ignore)'
CropNonEmptyMaskIfExists_parameters['ignore_channels'] = 'channels to ignore in mask (e.g. if background is a first channel set ignore_channels=[0] to ignore)'
CropNonEmptyMaskIfExists.append(json.dumps(CropNonEmptyMaskIfExists_parameters))
Transformations.append(CropNonEmptyMaskIfExists)

# ElasticTransform
ElasticTransform = ['Elastic Transform', 'albumentations', 'Elastic deformation of images']
ElasticTransform_parameters = {}
ElasticTransform_parameters['alpha'] = ''
ElasticTransform_parameters['sigma'] = 'Gaussian filter parameter.'
ElasticTransform_parameters['alpha_affine'] = 'The range will be (-alpha_affine, alpha_affine)'
ElasticTransform_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
ElasticTransform_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
ElasticTransform_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
ElasticTransform_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
ElasticTransform_parameters['approximate'] = 'Whether to smooth displacement map with fixed kernel size. Enabling this option gives ~2X speedup on large images.'
ElasticTransform_parameters['same_dxdy'] = 'Whether to use same random generated shift for x and y. Enabling this option gives ~2X speedup.'
ElasticTransform.append(json.dumps(ElasticTransform_parameters))
Transformations.append(ElasticTransform)

# Flip
Flip = ['Flip', 'albumentations', 'Flip the input either horizontally, vertically or both horizontally and vertically.']
Flip_parameters = {}
Flip.append(json.dumps(Flip_parameters))
Transformations.append(Flip)

# GridDistortion
GridDistortion = ['Grid Distortion', 'albumentations', '']
GridDistortion_parameters = {}
GridDistortion_parameters['num_steps'] = 'count of grid cells on each side.'
GridDistortion_parameters['distort_limit'] = 'If distort_limit is a single float, the range will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).'
GridDistortion_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
GridDistortion_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
GridDistortion_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
GridDistortion_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
GridDistortion.append(json.dumps(GridDistortion_parameters))
Transformations.append(GridDistortion)

# GridDropout
GridDropout = ['Grid Dropout', 'albumentations', 'GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.']
GridDropout_parameters = {}
GridDropout_parameters['ratio'] = 'the ratio of the mask holes to the unit_size (same for horizontal and vertical directions). Must be between 0 and 1. Default: 0.5.'
GridDropout_parameters['unit_size_min'] = 'minimum size of the grid unit. Must be between 2 and the image shorter edge. If \'None\', holes_number_x and holes_number_y are used to setup the grid. Default: None.'
GridDropout_parameters['unit_size_max'] = 'maximum size of the grid unit. Must be between 2 and the image shorter edge. If \'None\', holes_number_x and holes_number_y are used to setup the grid. Default: None.'
GridDropout_parameters['holes_number_x'] = 'the number of grid units in x direction. Must be between 1 and image height//2. If None, grid unit height is set equal to the grid unit width or image height, whatever is smaller.'
GridDropout_parameters['holes_number_y'] = 'the number of grid units in y direction. Must be between 1 and image width//2. If \'None\', grid unit width is set as image_width//10. Default: None.'
GridDropout_parameters['shift_x'] = 'offsets of the grid start in x direction from (0,0) coordinate. Clipped between 0 and grid unit_width - hole_width. Default: 0.'
GridDropout_parameters['shift_y'] = 'offsets of the grid start in y direction from (0,0) coordinate. Clipped between 0 and grid unit height - hole_height. Default: 0.'
GridDropout_parameters['random_offset'] = 'weather to offset the grid randomly between 0 and grid unit size - hole size If \'True\', entered shift_x, shift_y are ignored and set randomly. Default: False.'
GridDropout_parameters['fill_value'] = 'value for the dropped pixels. Default = 0'
GridDropout_parameters['mask_fill_value'] = 'value for the dropped pixels in mask. If None, transformation is not applied to the mask. Default: None.'
GridDropout.append(json.dumps(GridDropout_parameters))
Transformations.append(GridDropout)

# HorizontalFlip
HorizontalFlip = ['Horizontal Flip', 'albumentations', 'Flip the input horizontally around the y-axis.']
HorizontalFlip_parameters = {}
HorizontalFlip.append(json.dumps(HorizontalFlip_parameters))
Transformations.append(HorizontalFlip)

# LongestMaxSize
LongestMaxSize = ['Longest Max Size', 'albumentations', 'Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.']
LongestMaxSize_parameters = {}
LongestMaxSize_parameters['max_size'] = 'maximum size of the image after the transformation. When using a list, max size will be randomly selected from the values in the list.'
LongestMaxSize_parameters['interpolation'] = 'interpolation method. Default: cv2.INTER_LINEAR.'
LongestMaxSize.append(json.dumps(LongestMaxSize_parameters))
Transformations.append(LongestMaxSize)

# MaskDropout
MaskDropout = ['Mask Dropout', 'albumentations', 'Image & mask augmentation that zero out mask and image regions corresponding to randomly chosen object instance from mask. Mask must be single-channel image, zero values treated as background. Image can be any number of channels.']
MaskDropout_parameters = {}
MaskDropout_parameters['max_objects'] = 'Maximum number of labels that can be zeroed out. Can be tuple, in this case it\'s [min, max]'
MaskDropout_parameters['image_fill_value'] = 'Fill value to use when filling image. Can be \'inpaint\' to apply inpaining (works only for 3-chahnel images)'
MaskDropout_parameters['mask_fill_value'] = 'Fill value to use when filling mask.'
MaskDropout.append(json.dumps(MaskDropout_parameters))
Transformations.append(MaskDropout)

# OpticalDistortion
OpticalDistortion = ['Optical Distortion', 'albumentations', '']
OpticalDistortion_parameters = {}
OpticalDistortion_parameters['distort_limit'] = 'If distort_limit is a single float, the range will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).'
OpticalDistortion_parameters['shift_limit'] = 'If shift_limit is a single float, the range will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).'
OpticalDistortion_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
OpticalDistortion_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
OpticalDistortion_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
OpticalDistortion_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
OpticalDistortion.append(json.dumps(OpticalDistortion_parameters))
Transformations.append(OpticalDistortion)

# PadIfNeeded
PadIfNeeded = ['Pad If Needed', 'albumentations', 'Pad side of the image / max if side is less than desired number.']
PadIfNeeded_parameters = {}
PadIfNeeded_parameters['min_height'] = 'minimal result image height.'
PadIfNeeded_parameters['min_width'] = 'minimal result image width.'
PadIfNeeded_parameters['pad_height_divisor'] = 'if not None, ensures image height is dividable by value of this argument.'
PadIfNeeded_parameters['pad_width_divisor'] = 'if not None, ensures image width is dividable by value of this argument.'
PadIfNeeded_parameters['position'] = 'Position of the image. should be PositionType.CENTER or PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT. Default: PositionType.CENTER.'
PadIfNeeded_parameters['border_mode'] = 'OpenCV border mode.'
PadIfNeeded_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
PadIfNeeded_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
PadIfNeeded.append(json.dumps(PadIfNeeded_parameters))
Transformations.append(PadIfNeeded)

# Perspective
Perspective = ['Perspective', 'albumentations', 'Perform a random four point perspective transform of the input.']
Perspective_parameters = {}
Perspective_parameters['scale'] = 'standard deviation of the normal distributions. These are used to sample the random distances of the subimage\'s corners from the full image\'s corners. If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).'
Perspective_parameters['keep_size'] = 'Whether to resize image\'s back to their original size after applying the perspective transform. If set to False, the resulting images may end up having different shapes and will always be a list, never an array. Default: True'
Perspective_parameters['pad_mode'] = 'OpenCV border mode.'
Perspective_parameters['pad_val'] = 'padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0'
Perspective_parameters['mask_pad_val'] = 'padding value for mask if border_mode is cv2.BORDER_CONSTANT. Default: 0'
Perspective_parameters['fit_output'] = 'If True, the image plane size and position will be adjusted to still capture the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.) Otherwise, parts of the transformed image may be outside of the image plane. This setting should not be set to True when using large scale values as it could lead to very large images. Default: False'
Perspective.append(json.dumps(Perspective_parameters))
Transformations.append(Perspective)

# PiecewiseAffine
PiecewiseAffine = ['Piecewise Affine', 'albumentations', 'Apply affine transformations that differ between local neighbourhoods. This augmentation places a regular grid of points on an image and randomly moves the neighbourhood of these point around via affine transformations. This leads to local distortions.']
PiecewiseAffine_parameters = {}
PiecewiseAffine_parameters['scale'] = 'Each point on the regular grid is moved around via a normal distribution. This scale factor is equivalent to the normal distribution\'s sigma. Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of the image if absolute_scale=False (default), so this scale can be the same for different sized images. Recommended values are in the range 0.01 to 0.05 (weak to strong augmentations). * If a single float, then that value will always be used as the scale. * If a tuple (a, b) of float s, then a random value will be uniformly sampled per image from the interval [a, b].'
PiecewiseAffine_parameters['nb_rows'] = 'Number of rows of points that the regular grid should have. Must be at least 2. For large images, you might want to pick a higher value than 4. You might have to then adjust scale to lower values. * If a single int, then that value will always be used as the number of rows. * If a tuple (a, b), then a value from the discrete interval [a..b] will be uniformly sampled per image.'
PiecewiseAffine_parameters['nb_cols'] = 'Number of columns. Analogous to nb_rows.'
PiecewiseAffine_parameters['interpolation'] = 'The order of interpolation. The order has to be in the range 0-5: - 0: Nearest-neighbor - 1: Bi-linear (default) - 2: Bi-quadratic - 3: Bi-cubic - 4: Bi-quartic - 5: Bi-quintic'
PiecewiseAffine_parameters['mask_interpolation'] = 'same as interpolation but for mask.'
PiecewiseAffine_parameters['cval'] = 'The constant value to use when filling in newly created pixels.'
PiecewiseAffine_parameters['cval_mask'] = '{\'constant\', \'edge\', \'symmetric\', \'reflect\', \'wrap\'}, optional Points outside the boundaries of the input are filled according to the given mode. Modes match the behaviour of numpy.pad.'
PiecewiseAffine_parameters['mode'] = 'OpenCV border mode.'
PiecewiseAffine_parameters['absolute_scale'] = 'Take scale as an absolute value rather than a relative value.'
PiecewiseAffine_parameters['keypoints_threshold'] = 'Used as threshold in conversion from distance maps to keypoints. The search for keypoints works by searching for the argmin (non-inverted) or argmax (inverted) in each channel. This parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit as a keypoint. Use None to use no min/max. Default: 0.01'
PiecewiseAffine.append(json.dumps(PiecewiseAffine_parameters))
Transformations.append(PiecewiseAffine)

# PixelDropout
PixelDropout = ['Pixel Dropout', 'albumentations', 'Set pixels to 0 with some probability.']
PixelDropout_parameters = {}
PixelDropout_parameters['dropout_prob'] = 'pixel drop probability. Default: 0.01'
PixelDropout_parameters['per_channel'] = 'if set to True drop mask will be sampled fo each channel, otherwise the same mask will be sampled for all channels. Default: False'
PixelDropout_parameters['drop_value'] = 'Value that will be set in dropped place. If set to None value will be sampled randomly, default ranges will be used: - uint8 - [0, 255] - uint16 - [0, 65535] - uint32 - [0, 4294967295] - float, double - [0, 1] Default: 0'
PixelDropout_parameters['mask_drop_value'] = 'Value that will be set in dropped place in masks. If set to None masks will be unchanged. Default: 0'
PixelDropout.append(json.dumps(PixelDropout_parameters))
Transformations.append(PixelDropout)

# RandomCrop
RandomCrop = ['Random Crop', 'albumentations', 'Crop a random part of the input.']
RandomCrop_parameters = {}
RandomCrop_parameters['height'] = 'height of the crop.'
RandomCrop_parameters['width'] = 'width of the crop.'
RandomCrop.append(json.dumps(RandomCrop_parameters))
Transformations.append(RandomCrop)

# RandomCropNearBBox
RandomCropNearBBox = ['Random Crop Near BBox', 'albumentations', 'Crop bbox from image with random shift by x,y coordinates']
RandomCropNearBBox_parameters = {}
RandomCropNearBBox_parameters['max_part_shift'] = 'Max shift in height and width dimensions relative to cropping_bbox dimension. If max_part_shift is a single float, the range will be (max_part_shift, max_part_shift). Default (0.3, 0.3).'
RandomCropNearBBox_parameters['cropping_box_key'] = 'Additional target key for cropping box. Default cropping_bbox'
RandomCropNearBBox.append(json.dumps(RandomCropNearBBox_parameters))
Transformations.append(RandomCropNearBBox)

# RandomGridShuffle
RandomGridShuffle = ['Random Grid Shuffle', 'albumentations', 'Random shuffle grid\'s cells on image.']
RandomGridShuffle_parameters = {}
RandomGridShuffle_parameters['grid'] = 'size of grid for splitting image.'
RandomGridShuffle.append(json.dumps(RandomGridShuffle_parameters))
Transformations.append(RandomGridShuffle)

# RandomResizedCrop
RandomResizedCrop = ['Random Resized Crop', 'albumentations', 'Torchvision\'s variant of crop a random part of the input and rescale it to some size.']
RandomResizedCrop_parameters = {}
RandomResizedCrop_parameters['height'] = 'height after crop and resize.'
RandomResizedCrop_parameters['width'] = 'width after crop and resize.'
RandomResizedCrop_parameters['scale'] = 'range of size of the origin size cropped'
RandomResizedCrop_parameters['ratio'] = 'range of aspect ratio of the origin aspect ratio cropped'
RandomResizedCrop_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
RandomResizedCrop.append(json.dumps(RandomResizedCrop_parameters))
Transformations.append(RandomResizedCrop)

# RandomRotate90
RandomRotate90 = ['Random Rotate 90', 'albumentations', 'Randomly rotate the input by 90 degrees zero or more times.']
RandomRotate90_parameters = {}
RandomRotate90.append(json.dumps(RandomRotate90_parameters))
Transformations.append(RandomRotate90)

# RandomScale
RandomScale = ['Random Scale', 'albumentations', 'Randomly resize the input. Output image size is different from the input image size.']
RandomScale_parameters = {}
RandomScale_parameters['scale_limit'] = 'scaling factor range. If scale_limit is a single float value, the range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).'
RandomScale_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
RandomScale.append(json.dumps(RandomScale_parameters))
Transformations.append(RandomScale)

# RandomSizedBBoxSafeCrop
RandomSizedBBoxSafeCrop = ['Random Sized BBox Safe Crop', 'albumentations', 'Crop a random part of the input and rescale it to some size without loss of bboxes.']
RandomSizedBBoxSafeCrop_parameters = {}
RandomSizedBBoxSafeCrop_parameters['height'] = 'height after crop and resize.'
RandomSizedBBoxSafeCrop_parameters['width'] = 'width after crop and resize.'
RandomSizedBBoxSafeCrop_parameters['erosion_rate'] = 'erosion rate applied on input image height before crop.'
RandomSizedBBoxSafeCrop_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
RandomSizedBBoxSafeCrop.append(json.dumps(RandomSizedBBoxSafeCrop_parameters))
Transformations.append(RandomSizedBBoxSafeCrop)

# RandomSizedCrop
RandomSizedCrop = ['Random Sized Crop', 'albumentations', 'Crop a random part of the input and rescale it to some size without loss of bboxes.']
RandomSizedCrop_parameters = {}
RandomSizedCrop_parameters['min_max_height'] = 'crop size limits.'
RandomSizedCrop_parameters['height'] = 'height after crop and resize.'
RandomSizedCrop_parameters['width'] = 'width after crop and resize.'
RandomSizedCrop_parameters['w2h_ratio'] = 'aspect ratio of crop.'
RandomSizedCrop_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
RandomSizedCrop.append(json.dumps(RandomSizedCrop_parameters))
Transformations.append(RandomSizedCrop)

# Resize
Resize = ['Resize', 'albumentations', 'Resize the input to the given height and width.']
Resize_parameters = {}
Resize_parameters['height'] = 'desired height of the output.'
Resize_parameters['width'] = 'desired width of the output.'
Resize_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
Resize.append(json.dumps(Resize_parameters))
Transformations.append(Resize)

# Rotate
Rotate = ['Rotate', 'albumentations', 'Rotate the input by an angle selected randomly from the uniform distribution.']
Rotate_parameters = {}
Rotate_parameters['limit'] = 'range from which a random angle is picked. If limit is a single int an angle is picked from (-limit, limit). Default: (-90, 90)'
Rotate_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
Rotate_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
Rotate_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
Rotate_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
Rotate.append(json.dumps(Rotate_parameters))
Transformations.append(Rotate)

# SafeRotate
SafeRotate = ['Safe Rotate', 'albumentations', 'Rotate the input inside the input\'s frame by an angle selected randomly from the uniform distribution. The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and after resizing, it returns to its original shape with the original aspect ratio of the image. For these reason we may see some artifacts.']
SafeRotate_parameters = {}
SafeRotate_parameters['limit'] = 'range from which a random angle is picked. If limit is a single int an angle is picked from (-limit, limit). Default: (-90, 90)'
SafeRotate_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
SafeRotate_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
SafeRotate_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
SafeRotate_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
SafeRotate.append(json.dumps(SafeRotate_parameters))
Transformations.append(SafeRotate)

# ShiftScaleRotate
ShiftScaleRotate = ['Shift Scale Rotate', 'albumentations', 'Randomly apply affine transforms: translate, scale and rotate the input.']
ShiftScaleRotate_parameters = {}
ShiftScaleRotate_parameters['shift_limit'] = 'shift factor range for both height and width. If shift_limit is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).'
ShiftScaleRotate_parameters['scale_limit'] = 'scaling factor range. If scale_limit is a single float value, the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).'
ShiftScaleRotate_parameters['rotate_limit'] = 'rotation range. If rotate_limit is a single int value, the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).'
ShiftScaleRotate_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
ShiftScaleRotate_parameters['border_mode'] = 'flag that is used to specify the pixel extrapolation method. Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101'
ShiftScaleRotate_parameters['value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT.'
ShiftScaleRotate_parameters['mask_value'] = 'padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.'
ShiftScaleRotate_parameters['shift_limit_x'] = 'shift factor range for width. If it is set then this value instead of shift_limit will be used for shifting width. If shift_limit_x is a single float value, the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in the range [0, 1]. Default: None.'
ShiftScaleRotate_parameters['shift_limit_y'] = 'shift factor range for height. If it is set then this value instead of shift_limit will be used for shifting height. If shift_limit_y is a single float value, the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie in the range [0, 1]. Default: None.'
ShiftScaleRotate.append(json.dumps(SafeRotate_parameters))
Transformations.append(ShiftScaleRotate)

# SmallestMaxSize
SmallestMaxSize = ['Smallest Max Size', 'albumentations', 'Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.']
SmallestMaxSize_parameters = {}
SmallestMaxSize_parameters['max_size'] = 'maximum size of smallest side of the image after the transformation. When using a list, max size will be randomly selected from the values in the list.'
SmallestMaxSize_parameters['interpolation'] = 'flag that is used to specify the interpolation algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.'
SmallestMaxSize.append(json.dumps(SmallestMaxSize_parameters))
Transformations.append(SmallestMaxSize)

# Transpose
Transpose = ['Transpose', 'albumentations', 'Transpose the input by swapping rows and columns.']
Transpose_parameters = {}
Transpose.append(json.dumps(Transpose_parameters))
Transformations.append(Transpose)

# VerticalFlip
VerticalFlip = ['Vertical Flip', 'albumentations', 'Flip the input vertically around the x-axis.']
VerticalFlip_parameters = {}
VerticalFlip.append(json.dumps(VerticalFlip_parameters))
Transformations.append(VerticalFlip)


#pytorch
# CenterCrop
torchvision_CenterCrop = ['Center Crop', 'torchvision', 'Crops the given image at the center.']
torchvision_CenterCrop_parameters = {}
torchvision_CenterCrop_parameters['size'] = 'Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).'
torchvision_CenterCrop.append(json.dumps(torchvision_CenterCrop_parameters))
Transformations.append(torchvision_CenterCrop)

# ColorJitter
torchvision_ColorJitter = ['Color Jitter', 'torchvision', 'Randomly change the brightness, contrast, saturation and hue of an image.']
torchvision_ColorJitter_parameters = {}
torchvision_ColorJitter_parameters['brightness'] = 'How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.'
torchvision_ColorJitter_parameters['contrast'] = 'How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.'
torchvision_ColorJitter_parameters['saturation'] = ' How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.'
torchvision_ColorJitter_parameters['hue'] = 'How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.'
torchvision_ColorJitter.append(json.dumps(torchvision_ColorJitter_parameters))
Transformations.append(torchvision_ColorJitter)

# FiveCrop
torchvision_FiveCrop = ['Five Crop', 'torchvision', 'Crop the given image into four corners and the central crop.']
torchvision_FiveCrop_parameters = {}
torchvision_FiveCrop_parameters['size'] = ' Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop of size (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).'
torchvision_FiveCrop.append(json.dumps(torchvision_FiveCrop_parameters))
Transformations.append(torchvision_FiveCrop)

# Grayscale
torchvision_Grayscale = ['Grayscale', 'torchvision', 'Convert image to grayscale.']
torchvision_Grayscale_parameters = {}
torchvision_Grayscale_parameters['num_output_channels'] = '(1 or 3) number of channels desired for output image'
torchvision_Grayscale.append(json.dumps(torchvision_Grayscale_parameters))
Transformations.append(torchvision_Grayscale)

# Pad
torchvision_Pad = ['Pad', 'torchvision', 'Pad the given image on all sides with the given “pad” value.']
torchvision_Pad_parameters = {}
torchvision_Pad_parameters['padding'] = 'Padding on each border. If a single int is provided this is used to pad all borders. If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively. If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.'
torchvision_Pad_parameters['fill'] = 'Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant. Only number is supported for torch Tensor. Only int or str or tuple value is supported for PIL Image.'
torchvision_Pad_parameters['padding_mode'] = 'Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.'
torchvision_Pad.append(json.dumps(torchvision_Pad_parameters))
Transformations.append(torchvision_Pad)

# RandomAffine
torchvision_RandomAffine = ['RandomAffine', 'torchvision', 'Random affine transformation of the image keeping center invariant.']
torchvision_RandomAffine_parameters = {}
torchvision_RandomAffine_parameters['degrees'] = 'Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Set to 0 to deactivate rotations.'
torchvision_RandomAffine_parameters['translate'] = 'tuple of maximum absolute fraction for horizontal and vertical translations. For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.'
torchvision_RandomAffine_parameters['scale'] = 'scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b. Will keep original scale by default.'
torchvision_RandomAffine_parameters['shear'] = 'Range of degrees to select from. If shear is a number, a shear parallel to the x axis in the range (-shear, +shear) will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied. Will not apply shear by default.'
torchvision_RandomAffine_parameters['interpolation'] = 'Desired interpolation enum defined by torchvision.transforms.InterpolationMode. Default is InterpolationMode.NEAREST. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.'
torchvision_RandomAffine_parameters['fill'] = ' Pixel fill value for the area outside the transformed image. Default is 0. If given a number, the value is used for all bands respectively.'
torchvision_RandomAffine_parameters['fillcolor'] = ''
torchvision_RandomAffine_parameters['resample'] = ''
torchvision_RandomAffine_parameters['center'] = 'Optional center of rotation, (x, y). Origin is the upper left corner. Default is the center of the image.'
torchvision_RandomAffine.append(json.dumps(torchvision_RandomAffine_parameters))
Transformations.append(torchvision_RandomAffine)

# RandomApply
torchvision_RandomApply = ['RandomApply', 'torchvision', 'Apply randomly a list of transformations with a given probability.']
torchvision_RandomApply_parameters = {}
torchvision_RandomApply_parameters['transforms'] = 'list of transformations'
torchvision_RandomApply_parameters['p'] = 'probability'
torchvision_RandomApply.append(json.dumps(torchvision_RandomApply_parameters))
Transformations.append(torchvision_RandomApply)

# RandomCrop
torchvision_RandomCrop = ['RandomCrop', 'torchvision', 'Crop the given image at a random location. ']
torchvision_RandomCrop_parameters = {}
torchvision_RandomCrop_parameters['size'] = 'Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).'
torchvision_RandomCrop_parameters['padding'] = 'Optional padding on each border of the image. Default is None. If a single int is provided this is used to pad all borders. If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively. If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.'
torchvision_RandomCrop_parameters['pad_if_needed'] = ' It will pad the image if smaller than the desired size to avoid raising an exception. Since cropping is done after padding, the padding seems to be done at a random offset.'
torchvision_RandomCrop_parameters['fill'] = ' Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant. Only number is supported for torch Tensor. Only int or str or tuple value is supported for PIL Image.'
torchvision_RandomCrop_parameters['padding_mode'] = 'Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.'
torchvision_RandomCrop.append(json.dumps(torchvision_RandomCrop_parameters))
Transformations.append(torchvision_RandomCrop)

# RandomGrayscale
torchvision_RandomGrayscale = ['RandomGrayscale', 'torchvision', 'Crop the given image at a random location. ']
torchvision_RandomGrayscale_parameters = {}
torchvision_RandomGrayscale_parameters['p'] = 'probability that image should be converted to grayscale.'
torchvision_RandomGrayscale.append(json.dumps(torchvision_RandomGrayscale_parameters))
Transformations.append(torchvision_RandomGrayscale)

# RandomHorizontalFlip
torchvision_RandomHorizontalFlip = ['RandomHorizontalFlip', 'torchvision', 'Horizontally flip the given image randomly with a given probability.']
torchvision_RandomHorizontalFlip_parameters = {}
torchvision_RandomHorizontalFlip_parameters['p'] = 'probability that image should be converted to grayscale.'
torchvision_RandomHorizontalFlip.append(json.dumps(torchvision_RandomHorizontalFlip_parameters))
Transformations.append(torchvision_RandomHorizontalFlip)

# RandomPerspective
torchvision_RandomPerspective = ['RandomPerspective', 'torchvision', 'Performs a random perspective transformation of the given image with a given probability. ']
torchvision_RandomPerspective_parameters = {}
torchvision_RandomPerspective_parameters['distortion_scale'] = 'argument to control the degree of distortion and ranges from 0 to 1. Default is 0.5.'
torchvision_RandomPerspective_parameters['p'] = 'probability of the image being transformed. Default is 0.5.'
torchvision_RandomPerspective_parameters['interpolation'] = 'desired interpolation enum defined by torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.'
torchvision_RandomPerspective_parameters['fill'] = 'Pixel fill value for the area outside the transformed image. Default is 0. If given a number, the value is used for all bands respectively.'
torchvision_RandomPerspective.append(json.dumps(torchvision_RandomPerspective_parameters))
Transformations.append(torchvision_RandomPerspective)

# RandomResizedCrop
torchvision_RandomResizedCrop = ['RandomResizedCrop', 'torchvision', 'Crop a random portion of image and resize it to a given size.']
torchvision_RandomResizedCrop_parameters = {}
torchvision_RandomResizedCrop_parameters['size'] = 'expected output size of the crop, for each edge. If size is an int instead of sequence like (h, w), a square output size (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).'
torchvision_RandomResizedCrop_parameters['scale'] = 'Specifies the lower and upper bounds for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.'
torchvision_RandomResizedCrop_parameters['ratio'] = 'lower and upper bounds for the random aspect ratio of the crop, before resizing.'
torchvision_RandomResizedCrop_parameters['interpolation'] = 'desired interpolation enum defined by torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.'
torchvision_RandomResizedCrop.append(json.dumps(torchvision_RandomResizedCrop_parameters))
Transformations.append(torchvision_RandomResizedCrop)

# RandomRotation
torchvision_RandomRotation = ['RandomRotation', 'torchvision', 'Rotate the image by angle']
torchvision_RandomRotation_parameters = {}
torchvision_RandomRotation_parameters['degrees'] = 'Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).'
torchvision_RandomRotation_parameters['interpolation'] = 'desired interpolation enum defined by torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.'
torchvision_RandomRotation_parameters['expand'] = 'Optional expansion flag. If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.'
torchvision_RandomRotation_parameters['center'] = ' Optional center of rotation, (x, y). Origin is the upper left corner. Default is the center of the image.'
torchvision_RandomRotation_parameters['fill'] = 'Pixel fill value for the area outside the rotated image. Default is 0. If given a number, the value is used for all bands respectively.'
torchvision_RandomRotation_parameters['resample'] = ''
torchvision_RandomRotation.append(json.dumps(torchvision_RandomRotation_parameters))
Transformations.append(torchvision_RandomRotation)
 
# RandomVerticalFlip
torchvision_RandomVerticalFlip = ['RandomVerticalFlip', 'torchvision', 'Vertically flip the given image randomly with a given probability.']
torchvision_RandomVerticalFlip_parameters = {}
torchvision_RandomVerticalFlip_parameters['p'] = ' probability of the image being flipped. Default value is 0.5'
torchvision_RandomVerticalFlip.append(json.dumps(torchvision_RandomRotation_parameters))
Transformations.append(torchvision_RandomVerticalFlip)

# Resize
torchvision_Resize = ['Resize', 'torchvision', 'Resize the input image to the given size']
torchvision_Resize_parameters = {}
torchvision_Resize_parameters['size'] = 'Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).'
torchvision_Resize_parameters['interpolation'] = 'desired interpolation enum defined by torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR. If input is Tensor, only InterpolationMode.NEAREST, InterpolationMode.BILINEAR are supported. For backward compatibility integer values (e.g. PIL.Image.NEAREST) are still acceptable.'
torchvision_Resize_parameters['max_size'] = 'The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater than max_size after being resized according to size, then the image is resized again so that the longer edge is equal to max_size. As a result, size might be overruled, i.e the smaller edge may be shorter than size. This is only supported if size is an int (or a sequence of length 1 in torchscript mode).'
torchvision_Resize_parameters['antialias'] = 'antialias flag. If img is PIL Image, the flag is ignored and anti-alias is always used. If img is Tensor, the flag is False by default and can be set to True for InterpolationMode.BILINEAR only mode. This can help making the output for PIL images and tensors closer.'
torchvision_Resize.append(json.dumps(torchvision_Resize_parameters))
Transformations.append(torchvision_Resize)

# TenCrop
torchvision_TenCrop = ['TenCrop', 'torchvision', 'Crop the given image into four corners and the central crop plus the flipped version of these (horizontal flipping is used by default).']
torchvision_TenCrop_parameters = {}
torchvision_TenCrop_parameters['size'] = 'Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).'
torchvision_TenCrop_parameters['vertical_flip'] = 'Use vertical flipping instead of horizontal'
torchvision_TenCrop.append(json.dumps(torchvision_TenCrop_parameters))
Transformations.append(torchvision_TenCrop)

# GaussianBlur
torchvision_GaussianBlur = ['GaussianBlur', 'torchvision', 'Blurs image with randomly chosen Gaussian blur.']
torchvision_GaussianBlur_parameters = {}
torchvision_GaussianBlur_parameters['kernel_size'] = 'Size of the Gaussian kernel.'
torchvision_GaussianBlur_parameters['sigma'] = 'Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.'
torchvision_GaussianBlur.append(json.dumps(torchvision_GaussianBlur_parameters))
Transformations.append(torchvision_GaussianBlur)

# RandomInvert
torchvision_RandomInvert = ['RandomInvert', 'torchvision', 'Inverts the colors of the given image randomly with a given probability. ']
torchvision_RandomInvert_parameters = {}
torchvision_RandomInvert_parameters['p'] = 'probability of the image being color inverted. Default value is 0.5'
torchvision_RandomInvert.append(json.dumps(torchvision_RandomInvert_parameters))
Transformations.append(torchvision_RandomInvert)

# RandomPosterize
torchvision_RandomPosterize = ['RandomPosterize', 'torchvision', 'Posterize the image randomly with a given probability by reducing the number of bits for each color channel.']
torchvision_RandomPosterize_parameters = {}
torchvision_RandomPosterize_parameters['bits'] = 'number of bits to keep for each channel (0-8)'
torchvision_RandomPosterize_parameters['p'] = 'probability of the image being color inverted. Default value is 0.5'
torchvision_RandomPosterize.append(json.dumps(torchvision_RandomPosterize_parameters))
Transformations.append(torchvision_RandomPosterize)

# RandomSolarize
torchvision_RandomSolarize = ['RandomSolarize', 'torchvision', 'Solarize the image randomly with a given probability by inverting all pixel values above a threshold.']
torchvision_RandomSolarize_parameters = {}
torchvision_RandomSolarize_parameters['threshold'] = 'all pixels equal or above this value are inverted.'
torchvision_RandomSolarize_parameters['p'] = 'probability of the image being color inverted. Default value is 0.5'
torchvision_RandomSolarize.append(json.dumps(torchvision_RandomSolarize_parameters))
Transformations.append(torchvision_RandomSolarize)

# RandomAdjustSharpness
torchvision_RandomSolarize = ['RandomAdjustSharpness', 'torchvision', 'Adjust the sharpness of the image randomly with a given probability.']
torchvision_RandomSolarize_parameters = {}
torchvision_RandomSolarize_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_RandomSolarize_parameters['p'] = 'probability of the image being color inverted. Default value is 0.5'
torchvision_RandomSolarize.append(json.dumps(torchvision_RandomSolarize_parameters))
Transformations.append(torchvision_RandomSolarize)

# RandomAutocontrast
torchvision_RandomAutocontrast = ['RandomAutocontrast', 'torchvision', 'Autocontrast the pixels of the given image randomly with a given probability.']
torchvision_RandomAutocontrast_parameters = {}
torchvision_RandomAutocontrast_parameters['p'] = 'probability of the image being autocontrasted. Default value is 0.5'
torchvision_RandomAutocontrast.append(json.dumps(torchvision_RandomAutocontrast_parameters))
Transformations.append(torchvision_RandomAutocontrast)

# RandomEqualize
torchvision_RandomEqualize = ['RandomEqualize', 'torchvision', 'Equalize the histogram of the given image randomly with a given probability.']
torchvision_RandomEqualize_parameters = {}
torchvision_RandomEqualize_parameters['p'] = 'probability of the image being equalized. Default value is 0.5'
torchvision_RandomEqualize.append(json.dumps(torchvision_RandomEqualize_parameters))
Transformations.append(torchvision_RandomEqualize)

# adjust_brightness
torchvision_adjust_brightness = ['adjust_brightness', 'torchvision', 'Adjust brightness of an image.']
torchvision_adjust_brightness_parameters = {}
torchvision_adjust_brightness_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_brightness_parameters['brightness_factor'] = 'How much to adjust the brightness.'
torchvision_adjust_brightness.append(json.dumps(torchvision_adjust_brightness_parameters))
Transformations.append(torchvision_adjust_brightness)

# adjust_contrast
torchvision_adjust_contrast = ['adjust_contrast', 'torchvision', 'Adjust contrast of an image.']
torchvision_adjust_contrast_parameters = {}
torchvision_adjust_contrast_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_contrast_parameters['brightness_factor'] = 'How much to adjust the contrast.'
torchvision_adjust_contrast.append(json.dumps(torchvision_adjust_brightness_parameters))
Transformations.append(torchvision_adjust_contrast)

# adjust_gamma
torchvision_adjust_gamma = ['adjust_gamma', 'torchvision', 'Adjust contrast of an image.']
torchvision_adjust_gamma_parameters = {}
torchvision_adjust_gamma_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_gamma_parameters['gamma'] = 'Non negative real number, same as 𝛾 in the equation. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.'
torchvision_adjust_gamma_parameters['gain'] = ' The constant multiplier.'
torchvision_adjust_gamma.append(json.dumps(torchvision_adjust_gamma_parameters))
Transformations.append(torchvision_adjust_gamma)

# adjust_hue
torchvision_adjust_hue = ['adjust_hue', 'torchvision', 'Adjust hue of an image. The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel (H). The image is then converted back to original image mode. hue_factor is the amount of shift in H channel and must be in the interval [-0.5, 0.5].']
torchvision_adjust_hue_parameters = {}
torchvision_adjust_hue_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_hue_parameters['hue_factor'] = 'How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.'
torchvision_adjust_hue.append(json.dumps(torchvision_adjust_hue_parameters))
Transformations.append(torchvision_adjust_hue)

# adjust_saturation
torchvision_adjust_saturation = ['adjust_saturation', 'torchvision', 'Adjust color saturation of an image.']
torchvision_adjust_saturation_parameters = {}
torchvision_adjust_saturation_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_saturation_parameters['saturation_factor'] = ' How much to adjust the saturation. 0 will give a black and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.'
torchvision_adjust_saturation.append(json.dumps(torchvision_adjust_saturation_parameters))
Transformations.append(torchvision_adjust_saturation)

# adjust_sharpness
torchvision_adjust_sharpness = ['adjust_sharpness', 'torchvision', 'Adjust the sharpness of an image.']
torchvision_adjust_sharpness_parameters = {}
torchvision_adjust_sharpness_parameters['img'] = 'Image to be adjusted.'
torchvision_adjust_sharpness_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_adjust_sharpness.append(json.dumps(torchvision_adjust_sharpness_parameters))
Transformations.append(torchvision_adjust_sharpness)

# affine
torchvision_affine = ['affine', 'torchvision', 'Apply affine transformation on the image keeping image center invariant.']
torchvision_affine_parameters = {}
torchvision_affine_parameters['img'] = 'Image to be adjusted.'
torchvision_affine_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_affine_parameters['img'] = 'Image to be adjusted.'
torchvision_affine_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_affine_parameters['img'] = 'Image to be adjusted.'
torchvision_affine_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_affine_parameters['img'] = 'Image to be adjusted.'
torchvision_affine_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_affine_parameters['img'] = 'Image to be adjusted.'
torchvision_affine_parameters['sharpness_factor'] = 'How much to adjust the sharpness. Can be any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.'
torchvision_affine.append(json.dumps(torchvision_affine_parameters))
Transformations.append(torchvision_affine)







transfs = pd.DataFrame(Transformations, columns=col_names)
transfs.to_csv('transformations.csv')