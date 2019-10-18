import os
import math
import threading
from random import shuffle
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', '/data2/nfs_share/loktarxiao/microscope_images_classificaiton/datasets/mix_up',
                           'directory path to raw image files')
tf.app.flags.DEFINE_string('tfrecord_dir', '/data1/jonathanwu/datasets/Microscope-Image-Cls-Png',
                           'directory path to tfrecord files')
tf.app.flags.DEFINE_integer('nb_threads', 16, '# of parallel threads')

CLASS_IDXS = {'normal': 0, 'tumor': 1}

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def process_images_impl(fpath_n_labels, coder, idx_thread, config):
  idx_shard_low = config['nb_shards_per_thread'] * idx_thread
  idx_shard_hgh = min(idx_shard_low + config['nb_shards_per_thread'], config['nb_shards'])
  for idx_shard in range(idx_shard_low, idx_shard_hgh):
    tfrecord_fname = config['file_pattern'] % (idx_shard, config['nb_shards'])
    writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfrecord_dir, tfrecord_fname))
    tf.logging.info('creating a TFRecordWriter for ' + tfrecord_fname)

    idx_smpl_low = config['nb_smpls_per_shard'] * idx_shard
    idx_smpl_hgh = min(idx_smpl_low + config['nb_smpls_per_shard'], len(fpath_n_labels))
    for idx_smpl in range(idx_smpl_low, idx_smpl_hgh):
      fpath, label = fpath_n_labels[idx_smpl]

      with open(fpath, 'rb') as i_file:
        image_data_png = i_file.read()
        image_data_jpeg = coder.png_to_jpeg(image_data_png)
        image = coder.decode_jpeg(image_data_jpeg)
        image_hei, image_wid = image.shape[0], image.shape[1]
        assert len(image.shape) == 3 and image.shape[2] == 3

      example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(image_hei),
        'image/width': _int64_feature(image_wid),
        'image/class/label': _int64_feature(CLASS_IDXS[label]),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(label)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data_png))
      }))
      writer.write(example.SerializeToString())

    writer.close()
    tf.logging.info('tfrecord file generated: ' + tfrecord_fname)

def convert_images_to_tfrecord(sub_dir_name, nb_shards, file_pattern):
  # obtain list of file names and corresponding labels
  image_dir = os.path.join(FLAGS.image_dir, sub_dir_name)
  fpath_n_labels = []
  for class_name in os.listdir(image_dir):
    class_dir = os.path.join(image_dir, class_name)
    fpath_n_labels += [(os.path.join(class_dir, fname), class_name) for fname in os.listdir(class_dir)]
  shuffle(fpath_n_labels)
  tf.logging.info('# of samples (%s): %d' % (sub_dir_name, len(fpath_n_labels)))

  # convert images to tfrecord files
  coord = tf.train.Coordinator()
  threads = []
  coder = ImageCoder()
  config = {
    'nb_shards': nb_shards,
    'file_pattern': file_pattern,
    'nb_smpls_per_shard': int(math.ceil(len(fpath_n_labels) / nb_shards)),
    'nb_shards_per_thread': int(math.ceil(nb_shards / FLAGS.nb_threads)),
  }
  for idx_thread in range(FLAGS.nb_threads):
    args = (fpath_n_labels, coder, idx_thread, config)
    t = threading.Thread(target=process_images_impl, args=args)
    t.start()
    threads += [t]
  coord.join(threads)

def main(unused_argv):
  # setup the TF logging routine
  tf.logging.set_verbosity(tf.logging.INFO)

  # convert training & validation samples to tfrecord files
  convert_images_to_tfrecord(
    sub_dir_name='train', nb_shards=256, file_pattern='train-%04d-of-%04d')
  convert_images_to_tfrecord(
    sub_dir_name='valid', nb_shards=16, file_pattern='valid-%04d-of-%04d')

if __name__ == '__main__':
  tf.app.run()
