import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

def blur_jpeg_transform(resize_dims):
	crop_func = transforms.CenterCrop(resize_dims)
	flip_func = transforms.RandomHorizontalFlip()
	rz_func = transforms.Lambda(lambda img: custom_resize(img,resize_dims))

	transforms_apply =  transforms.Compose([
				rz_func,
				transforms.Lambda(lambda img: data_augment(img)),
				crop_func,
				flip_func,
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			])
	return transforms_apply

def default_transform(resize_dims):
    return transforms.Compose([
            transforms.Resize(resize_dims),
            transforms.CenterCrop(resize_dims),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])

def data_augment(img):
	img = np.array(img)

	if random() < 0.1:
		sig = sample_continuous([0.0, 3.0])
		gaussian_blur(img, sig)

	if random() < 0.1:
		method = sample_discrete(['cv2', 'pil'])
		qual = sample_discrete(list(range(30, 100 + 1)))
		img = jpeg_from_key(img, qual, method)
	return Image.fromarray(img)


def sample_continuous(s):
	if len(s) == 1:
		return s[0]
	if len(s) == 2:
		rg = s[1] - s[0]
		return random() * rg + s[0]
	raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
	if len(s) == 1:
		return s[0]
	return choice(s)


def gaussian_blur(img, sigma):
	gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
	gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
	gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
	img_cv2 = img[:,:,::-1]
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
	result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
	decimg = cv2.imdecode(encimg, 1)
	return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
	out = BytesIO()
	img = Image.fromarray(img)
	img.save(out, format='jpeg', quality=compress_val)
	img = Image.open(out)
	# load from memory before ByteIO closes
	img = np.array(img)
	out.close()
	return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
	method = jpeg_dict[key]
	return method(img, compress_val)

def custom_resize(img, resize_dims):
	return TF.resize(img, resize_dims, interpolation=Image.BILINEAR)

def get_transform(name, resize_dims = 128):
    if name == 'default':
        return default_transform(resize_dims)
    else:
        return blur_jpeg_transform(resize_dims)