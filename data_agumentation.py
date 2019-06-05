from os import listdir
from os.path import isfile, join
# import tensorflow as tf 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# images = []
data_path = "/Users/apple/Downloads/data-colab/data_sperm/val/normal/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for f in onlyfiles:
	print(f)
	filepath = data_path + f
	print(filepath)
	# img_raw = tf.io.read_file(filepath)
	# img_tensor = tf.image.decode_image(img_raw)
	# x = tf.image.decode_image(img_raw)
	img = Image.open(filepath)
	x = np.array(img)

	x_fliplr = np.fliplr(x)
	x_flipud = np.flipud(x)
	filename_aug1 = data_path + 'aug_lr_' + f
	filename_aug2 = data_path + 'aug_ud_' + f
	# x_fliplr = tf.convert_to_tensor(x_fliplr)
	# x_flipud = tf.convert_to_tensor(x_flipud)
	# tf.io.write_file(filename_aug1, x_fliplr)
	# tf.io.write_file(filename_aug2, x_flipud)
	x_fliplr = Image.fromarray(x_fliplr)
	x_flipud = Image.fromarray(x_flipud)
	x_fliplr.save(filename_aug1)
	x_flipud.save(filename_aug2)
print("------Done------")
