# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inspect the model save as checkpoint files."""

import os
import re
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor
from tensorflow.python.keras import backend as K

from utils import build_graphs
from microscope_model import MicroscopeModel

FLAGS = tf.app.flags.FLAGS

# common configurations
tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_string('model_dir_src', './models_cpr_eval', 'model directory - source')
tf.app.flags.DEFINE_string('model_dir_dst', './models_cpr_final', 'model directory - destination')

# model configurations
tf.app.flags.DEFINE_string('model_scope', 'pruned_model', 'model scope')
tf.app.flags.DEFINE_string('data_format', 'channels_first', 'data format of the final model')
tf.app.flags.DEFINE_string('input_coll', 'images_final', 'input tensor\'s collection')
tf.app.flags.DEFINE_string('output_coll', 'logits_final', 'output tensor\'s collection')
tf.app.flags.DEFINE_boolean('enbl_chn_prune', True, 'enable compression with channel pruning')
tf.app.flags.DEFINE_boolean('enbl_fake_prune', False, 'enable fake pruning (for speed test only)')
tf.app.flags.DEFINE_float('fake_prune_ratio', 0.5, 'fake pruning ratio')

def create_session():
  """Create a TensorFlow session with GPU memory growth allowed.

  Returns:
  * sess: TensorFlow session
  """

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  return sess

def set_learning_phase():
  """Set the learning phase to 'inference' (this will save lots of trouble)."""

  model_scope = 'pruned_model'
  with tf.Graph().as_default():
    # set the learning phase to 'inference'
    K.set_learning_phase(0)

    # model definition
    with tf.variable_scope(model_scope):
      model = MicroscopeModel(data_format=FLAGS.data_format)
      vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)

    # add input & output tensors to certain collections
    tf.add_to_collection('images_final', model.inputs[0])
    tf.add_to_collection('logits_final', model.outputs[0])

    # restore variables from checkpoint files
    sess = create_session()
    save_path = tf.train.latest_checkpoint(FLAGS.model_dir_src)
    saver = tf.train.Saver(vars_all)
    saver.restore(sess, save_path)
    saver.save(sess, os.path.join(FLAGS.model_dir_dst, 'model.ckpt'))

def is_initialized(sess, var):
  """Check whether a variable is initialized.

  Args:
  * sess: TensorFlow session
  * var: variabile to be checked
  """

  try:
    sess.run(var)
    return True
  except tf.errors.FailedPreconditionError:
    return False

def apply_fake_pruning(kernel):
  """Apply fake pruning to the convolutional kernel.

  Args:
  * kernel: original convolutional kernel

  Returns:
  * kernel: randomly pruned convolutional kernel
  """

  tf.logging.info('kernel shape: {}'.format(kernel.shape))
  kernel = np.random.random(size=kernel.shape)
  nb_chns = kernel.shape[2]
  idxs_all = np.arange(nb_chns)
  np.random.shuffle(idxs_all)
  idxs_pruned = idxs_all[:int(nb_chns * FLAGS.fake_prune_ratio)]
  kernel[:, :, idxs_pruned, :] = 0.0

  return kernel

def insert_alt_routines(sess):
  """Insert alternative rountines for convolutional layers.

  Args:
  * sess: TensorFlow session
  * graph_trans_mthd: graph transformation method

  Returns:
  * op_outputs_old: output nodes to be swapped in the old graph
  * op_outputs_new: output nodes to be swapped in the new graph
  """

  pattern = re.compile(r'/Conv2D$')
  op_outputs_old, op_outputs_new = [], []
  for op in tf.get_default_graph().get_operations():
    if re.search(pattern, op.name) is not None:
      # skip un-initialized variables, which is not needed in the final *.pb file
      if not is_initialized(sess, op.inputs[1]):
        continue

      # detect which channels to be pruned
      tf.logging.info('transforming OP: ' + op.name)
      kernel = sess.run(op.inputs[1])
      if FLAGS.enbl_fake_prune:
        kernel = apply_fake_pruning(kernel)
      kernel_chn_in = kernel.shape[2]
      strides = op.get_attr('strides')
      padding = op.get_attr('padding').decode('utf-8')
      data_format = op.get_attr('data_format').decode('utf-8')
      dilations = op.get_attr('dilations')
      nnzs = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
      tf.logging.info('reducing %d channels to %d' % (kernel_chn_in, nnzs.size))
      kernel_gthr = np.zeros((1, 1, kernel_chn_in, nnzs.size))
      kernel_gthr[0, 0, nnzs, np.arange(nnzs.size)] = 1.0
      kernel_shrk = kernel[:, :, nnzs, :]

      # replace channel pruned convolutional with cheaper operations
      x = tf.gather(op.inputs[0], nnzs, axis=(1 if FLAGS.data_format == 'channels_first' else 3))
      x = tf.nn.conv2d(
        x, kernel_shrk, strides, padding, data_format=data_format, dilations=dilations)

      # obtain old and new routines' outputs
      op_outputs_old += [op.outputs[0]]
      op_outputs_new += [x]

  return op_outputs_old, op_outputs_new

def inspect_model(meta_path):
  """Inspect the model."""

  # convert checkpoint files to a *.pb model
  with tf.Graph().as_default() as graph:
    sess = create_session()

    # restore the graph
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, meta_path.replace('.meta', ''))

    # build relation graphs between tensors & operations
    t2o_graph, o2o_graph, o2c_graph = build_graphs(FLAGS.model_scope)

    # inspect the graph structure
    sorc_op_name = 'pruned_model/conv2d_1/Conv2D'
    sink_op_name = 'pruned_model/conv2d_2/Conv2D'
    for op in graph.get_operations():
      if op.name == sorc_op_name:
        sorc_op = op
      elif op.name == sink_op_name:
        sink_op = op

    depth = 0
    base_ops = [sorc_op]
    while sink_op not in base_ops:
      depth += 1
      tf.logging.info('depth = %d' % depth)
      base_ops_new = []
      for base_op in base_ops:
        tf.logging.info('{} -> {}'.format(base_op.name, [op.name for op in o2o_graph[base_op]['outputs']]))
        base_ops_new += o2o_graph[base_op]['outputs']
      base_ops = base_ops_new

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # set the learning phase to 'inference'; data format may be changed if needed
    set_learning_phase()

    # inspect the model
    meta_path = os.path.join(FLAGS.model_dir_dst, 'model.ckpt.meta')
    inspect_model(meta_path)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
