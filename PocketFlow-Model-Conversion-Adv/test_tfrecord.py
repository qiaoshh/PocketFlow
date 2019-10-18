import numpy as np
import tensorflow as tf

IMAGE_HEI = 299
IMAGE_WID = 299
IMAGE_CHN = 3
NB_CLASSES = 2

def parse_example_proto(example_serialized):
  feature_map = {
    'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
  }
  features = tf.parse_single_example(example_serialized, feature_map)

  image_buffer = features['image/encoded']
  label = features['image/class/label']

  return image_buffer, label

def parse_fn(example_serialized):
  image_buffer, label = parse_example_proto(example_serialized)

  image = tf.image.decode_jpeg(image_buffer, channels=IMAGE_CHN)
  image = tf.expand_dims(image, axis=0)
  image = tf.image.resize_bicubic(image, [IMAGE_HEI, IMAGE_WID])
  label = tf.one_hot(label, NB_CLASSES)
  tf.logging.info('image: {} / label: {}'.format(image.shape, label.shape))

  image = tf.squeeze(image, axis=0)
  label = tf.squeeze(label, axis=0)

  return image, label

### Main Entry ###

tf.logging.set_verbosity(tf.logging.INFO)

file_pattern = '/data1/jonathanwu/datasets/Microscope-Image-Cls/train-*-of-*'
#file_pattern = '/data1/jonathanwu/datasets/Microscope-Image-Cls/valid-*-of-*'
filenames = tf.data.Dataset.list_files(file_pattern, shuffle=True)
dataset = filenames.apply(
  tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))
dataset = dataset.map(parse_fn, num_parallel_calls=16)
dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=1))
dataset = dataset.batch(128)
dataset = dataset.prefetch(1)
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

sess = tf.Session()
nb_mbtcs = 0
nb_smpls_pos = 0
nb_smpls_neg = 0
while True:
  try:
    images_np, labels_np = sess.run([images, labels])
    nb_mbtcs += 1
    nb_smpls_pos += np.sum(labels_np[:, 1])
    nb_smpls_neg += np.sum(labels_np[:, 0])
  except:
    tf.logging.info('end of data')
    break
tf.logging.info('# of mini-batches: %d' % nb_mbtcs)
tf.logging.info('# of samples: %d (pos) / %d (neg)' % (nb_smpls_pos, nb_smpls_neg))
