### Script to convert image data to tf.train.Example to feed into Tensorflow Object Detection API (https://github.com/aliciajin/models/tree/master/research/object_detection)
### Partially based on: https://github.com/swirlingsand/deeper-traffic-lights/blob/master/data_conversion_bosch.py
### Author: Yiqian Jin(aliciajin)
### Date: May 2019 

import json
import tensorflow as tf
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path_train', 
	'/users/apple/git/models/research/object_detection/samples/morph/data/morph_train.record', 
	'Path to output TFRecord')
flags.DEFINE_string('output_path_val', 
	'/users/apple/git/models/research/object_detection/samples/morph/data/morph_val.record', 
	'Path to output TFRecord')
flags.DEFINE_string('input_path', '/Users/apple/Documents/cs231n/project_v2/', 'Path to input img')
flags.DEFINE_string('num_train', 0, 'num of datapoint for training')
FLAGS = flags.FLAGS

# output_path = "/users/apple/Documents/cs231n/project_v2/tf_examples"
# input_path = "/Users/apple/Documents/cs231n/project_v2/"

LABEL_DICT =  {
	"N" : 1,
	"A" : 2,
	}


def create_tf_example(example, folder):
	
	height = 582 # Image height
	width = 776 # Image width

	img_path = FLAGS.input_path + folder + example['filename']
	img_path = img_path.encode()
	with tf.gfile.GFile(img_path, 'rb') as fid:
		encoded_image = fid.read()

	image_format = 'png'.encode() 

	xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [] # List of normalized right x coordinates in bounding box
	            # (1 per box)
	ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [] # List of normalized bottom y coordinates in bounding box
	            # (1 per box)
	classes_text = [] # List of string class name of bounding box (1 per box)
	classes = [] # List of integer class id of bounding box (1 per box)

	for _, region_val in example['regions'].items():
		if region_val['shape_attributes']['name'] != 'rect':
			continue
		x_min = region_val['shape_attributes']['x']
		y_min = region_val['shape_attributes']['y']
		x_max = x_min + region_val['shape_attributes']['width']
		y_max = y_min + region_val['shape_attributes']['height']
		label = region_val['region_attributes']['Normal/Abnormal']

		xmins.append(float(x_min / width))
		xmaxs.append(float(x_max / width))
		ymins.append(float(y_min / height))
		ymaxs.append(float(y_max / height))
		classes_text.append(label.encode())
		classes.append(int(LABEL_DICT[label]))
	if len(classes) == 0:
		return None
    ## function based on https://github.com/swirlingsand/deeper-traffic-lights/blob/master/data_conversion_bosch.py
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(img_path),
		'image/source_id': dataset_util.bytes_feature(img_path),
		'image/encoded': dataset_util.bytes_feature(encoded_image),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))

	return tf_example



def main(_):
	folders = ['Annotated_image_data_set_one_0-426/', 'Annotated_image_data_set_two_427_1003/']
	writer_train = tf.python_io.TFRecordWriter(FLAGS.output_path_train)
	writer_val = tf.python_io.TFRecordWriter(FLAGS.output_path_val)

	count = 0
	num_train = FLAGS.num_train
	for folder in folders:
		label_file = FLAGS.input_path + folder + "via_region_data.json"
		with open(label_file, "r") as read_file:
			data = json.load(read_file)
			print('data size: ', len(data))
			for ex_name in data:
				example = data[ex_name]
				tf_example = create_tf_example(example, folder)
				if tf_example:
					count += 1
					if count <= num_train:
						writer_train.write(tf_example.SerializeToString())
					else:
						writer_val.write(tf_example.SerializeToString())
	print('valid rect images: ', count)
	print('num train: ', num_train)
	print('num val: ', count-num_train)
	writer_train.close()
	writer_train.close()

if __name__ == '__main__':
	tf.app.run()
