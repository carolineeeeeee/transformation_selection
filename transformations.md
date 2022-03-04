# Transformations Collection
No.| name | source | description | CV-HAZOP location, parameter | supported guide word | effects 
----|---|--------|----------|------------|------|-------
1|Gaussian Noise| imagenet-c | N/A |  Observer, Quality | Less, No | 
2|Shot Noise| imagenet-c | Camera sensor noise|  Observer, Quality | Less, No | 
3|Impulse Noise|imagenet-c | N/A |  Observer, Quality | Less, No | 
4|Defocus Blur|imagenet-c | N/A |  Observer, Focusing | Less | 
5|Frosted Glass Blur|imagenet-c | N/A |  Medium, Texture | Other than | 
6|Motion Blur|imagenet-c| N/A |  Observer, Focusing | Less | 
7|Zoom Blur|imagenet-c| N/A | Observer, Focusing |  Other than | 
8|Snow|imagenet-c| N/A |  Medium, Texture | More, Other than | 
9|Frost|imagenet-c|N/A |  Medium, Texture | More, Other than | 
10|Fog|imagenet-c|N/A |  Medium, Texture | Less, Other than | 
11|Brightness|imagenet-c| N/A |  Light Sources, Intensity | Adjust lighting and brightness. | 
12|Contrast|imagenet-c| N/A |  Observer, Quality | All | overexposure, underexposure
13|Elastic|imagenet-c| N/A|Observer, Lenses geometry | Part of| deform 
14|Pixelate|imagenet-c| N/A | Observer, Quantization/Sampling | Less | compress 
15|JPEG Compression|imagenet-c| N/A |  Observer, Resolution (spatial) | Less | 
16|Advanced Blur| albumentations|Blur the input image using a Generalized Normal filter with a randomly selected parameters. This transform also adds multiplicative noise to generated kernel before convolution.| Observer, Focusing; Observer, Quality| Less, Other than; Less|
17|Blur|albumentations| Blur the input image using a random-sized kernel. |  Observer, Focusing | Less | 
18|CLAHE|albumentations| Apply Contrast Limited Adaptive Histogram Equalization to the input image. | Observer, Quality; Light Sources, Texture | All| 
19|Channel Dropout|albumentations| Randomly Drop Channels in the input Image. | Observer, Spectral efficiency | Less|
20|Channel Shuffle|albumentations| Randomly rearrange channels of the input RGB image. | Observer, Spectral efficiency | Other than|
21|Color Jitter|albumentations|Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision, this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8 overflow, but we use value saturation. | Light Sources, Intensity; Light Sources, Spectrum; Observer, Quality | All;All;All |  
22|Downscale|albumentations| Decreases image quality by downscaling and upscaling back. | Observer, Quantization/Sampling | Less | 
23|Emboss|albumentations| Emboss the input image and overlays the result with the original image. | -,- | - |   
24|Equalize|albumentations|Equalize the image histogram. | Observer, Quality |  More | contrast
25|FDA|albumentations|Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA Simple "style transfer". | Observer, Spectral efficiency | All |
26|Fancy PCA|albumentations|Augment RGB image using FancyPCA from Krizhevsky's paper "ImageNet Classification with Deep Convolutional Neural Networks" | -,- |  -| Not relevant (a transformation for NN training) 
27|Flip|albumentations|Flip the input either horizontally, vertically or both horizontally and vertically.| Observer, Viewing orientation| Other than, Reverse | 
28|From Float|albumentations| Take an input array where all values should lie in the range [0, 1.0], multiply them by max_value and then cast the resulted value to a type specified by dtype. If max_value is None the transform will try to infer the maximum value for the data type from the dtype argument. | Observer, Quantization/Sampling |  Less |
29|Gauss Noise|albumentations|Apply gaussian noise to the input image. | Observer, Quality | Less, No | 
30|Gaussian Blur|albumentations| Blur the input image using a Gaussian filter with a random kernel size. |  Observer, Focusing | Less | 
31|Glass Blur|albumentations| Apply glass effect to the input image. | Medium, Texture | Other than | 
32| Horizontal Flip|albumentations| Flip the input horizontally around the y-axis.| Observer, Viewing orientation| Other than, Reverse |  
33|HistogramMatching|albumentations|Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches the histogram of the reference image. If the images have multiple channels, the matching is done independently for each channel, as long as the number of channels is equal in the input image and the reference. | Observer, Quality |  Other than | contrast
34|Hue Saturation Value |albumentations| Randomly change hue, saturation and value of the input image. |  Light Sources, Spectrum| All | 
35|ISO Noise|albumentations| Apply camera sensor noise. | Observer, Quality | Less | 
36|Image Compression|albumentations| Decrease Jpeg, WebP compression of an image. |  Observer, Resolution (spatial) | Less | 
37|Invert Img|albumentations| Invert the input image by subtracting pixel values from 255. | Observer, Quantization/Sampling|  Reverse |
38|Jpeg Compression| albumentations| Decrease Jpeg compression of an image.|Observer, Resolution (spatial) | Less |
39|Median Blur|albumentations| Blur the input image using a median filter with a random aperture linear size. |  Observer, Focusing | Less | 
40|Motion Blur|albumentations| Apply motion blur to the input image using a random-sized kernel. |Observer, Focusing | Less | 
41|Multiplicative Noise|albumentations| Multiply image to random number or array of numbers. |Observer, Quality| Less | 
42|Normalize|albumentations| Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value) | Observer, Quality |  More | contrast
43|Pixel Dropout|albumentations|Set pixels to 0 with some probability.| Observer, Quality | Less |
44|Posterize|albumentations| Reduce the number of bits for each color channel. |Observer, Quantization/Sampling| Less |
45|Random Brightness|albumentations|Randomly change brightness of the input image.|Light Sources, Intensity | All | 
46|Random Contrast|albumentations|Randomly change contrast of the input image.| Observer, Quality | All | overexposure, underexposure
47|RGB Shift|albumentations| Randomly shift values for each channel of the input RGB image. |  Light Sources, Spectrum| All | 
48|Random Brightness Contrast|albumentations| Randomly change brightness and contrast of the input image. | Light Sources, Intensity; Observer, Quality| All | 
49|Random Fog|albumentations| Simulates fog for the image | Medium, Texture | More, Other than | 
50|Random Gamma|albumentations|  Controls the overall brightness of an image |  Light Sources, Intensity | All | 
51|Random Grid Shuffle|albumantations|Random shuffle grid's cells on image.| -,-|-|
52|Random Shadow|albumentations|Simulates shadows for the image |Objects, Shadowing | More, Other than| 
53|Random Rain|albumentations| Adds rain effects. | Medium, Texture; Light Sources, Intensity | More; Less | 
54|Random Snow|albumentations| Bleach out some pixel values simulating snow. | Observer, Quality | Less |
55|Random Sun Flare|albumentations| Simulates Sun Flare for the image | Light Sources, Beam properties| Less | 
56|Random Tone Curve|albumantations|Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.| Observer, Spectral efficiency | All|
57|Sharpen|albumentations| Sharpen the input image and overlays the result with the original image. | Observer, Focusing |  More | 
58|Solarize|albumentations| Invert all pixel values above a threshold. | Observer, Quantization/Sampling|  Reverse |
59|Superpixels|albumentations| Transform images partially/completely to their superpixel representation. This implementation uses skimage's version of the SLIC algorithm. | Observer, Quantization/Sampling|  More |
60|Template Transform|albumentations|Apply blending of input image with specified templates| -,- | -|
61|To Float|albumentations| Divide pixel values by max_value to get a float32 output array where all values lie in the range [0, 1.0]. If max_value is None the transform will try to infer the maximum value by inspecting the data type of the input image. | Observer, Quantization/Sampling|  More |
62|To Gray|albumentations| Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image. | Observer, Spectral efficiency | Less|
63|To Sepia|albumentations|Applies sepia filter to the input RGB image | Observer, Spectral efficiency | Other than|
64|Transpose| albumentations|Transpose the input by swapping rows and columns.| Observer, Viewing orientation| Other than, Reverse |  
65|Unsharp Mask| albumentations|Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.|Observer, Focusing |  More | 
66|Vertical Flip| albumentations| Flip the input vertically around the x-axis.|Observer, Viewing orientation| Other than, Reverse |  
END

+  -,- means does not correspond to CV HAZOP entries
+ If more than one CV HAZOP entry is specifed for each transformation, it means that they are combinations