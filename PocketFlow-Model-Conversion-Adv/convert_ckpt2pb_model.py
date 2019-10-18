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
"""Export *.pb models from checkpoint files."""

import os
import re
import shutil
import traceback
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor
from tensorflow.python.client import timeline
from tensorflow.python.keras import backend as K

from microscope_model import MicroscopeModel
from graph_utils import build_graphs
from graph_utils import get_conv2d_op_n_vars  # for *.ckpt models
from graph_utils import get_conv2d_op_n_tensors  # for *.pb models

FLAGS = tf.app.flags.FLAGS

# common configurations
# tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_string('ckpt_model_dir', './ckpt.models.cpr', '*.ckpt model directory')
tf.app.flags.DEFINE_string('pb_model_dir', './pb.models.cpr', '*.pb model directory')
tf.app.flags.DEFINE_string('pb_fname_full', 'model_full.pb', '*.pb model\'s file name - full')
tf.app.flags.DEFINE_string('pb_fname_prnd', 'model_prnd.pb', '*.pb model\'s file name - pruned')

# model configurations
tf.app.flags.DEFINE_boolean('enbl_dst', False, 'enable the distillation loss for training')
tf.app.flags.DEFINE_string('model_scope', 'pruned_model', 'model scope')
tf.app.flags.DEFINE_string('data_format', 'channels_last', 'data format of the final model')
tf.app.flags.DEFINE_boolean('enbl_chn_prune', True, 'enable compression with channel pruning')
tf.app.flags.DEFINE_boolean('enbl_ci_prune', True, 'enable pruning for input channels')
tf.app.flags.DEFINE_boolean('enbl_co_prune', True, 'enable pruning for output channels')
tf.app.flags.DEFINE_boolean('enbl_fake_prune', False, 'enable fake pruning (for speed test only)')
tf.app.flags.DEFINE_float('fake_prune_ratio', 0.5, 'fake pruning ratio')
tf.app.flags.DEFINE_string('fake_prune_policy', 'uniform', 'fake pruning policy: \'uniform\' | \'heurist\'')
tf.app.flags.DEFINE_integer('nb_iters_wrmp', 32, '# of warm-up iterations for speed test')
tf.app.flags.DEFINE_integer('nb_iters_spdt', 32, '# of actual iterations for speed test')

def create_session():
  """Create a TensorFlow session with GPU memory growth allowed.

  Returns:
  * sess: TensorFlow session
  """

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  return sess

def apply_fake_pruning(sess):
  """Apply fake pruning to convolutional kernels.

  Args:
  * sess: TensorFlow session
  """

  prune_ops = []
  op_n_vars = get_conv2d_op_n_vars(FLAGS.model_scope)
  for idx, (op, var) in enumerate(op_n_vars):
    tf.logging.info('op: {} / var: {} / shape: {}'.format(op.name, var.name, var.shape))
    if FLAGS.fake_prune_policy == 'uniform':
      prune_ratio = FLAGS.fake_prune_ratio
    else:
      kh, kw = int(var.shape[0]), int(var.shape[1])
      if kh == 1 and kw == 1:
        prune_ratio = FLAGS.fake_prune_ratio * 0.5
      elif kw == 1 or kw == 1:
        prune_ratio = FLAGS.fake_prune_ratio
      else:
        prune_ratio = 1.0 - (1.0 - FLAGS.fake_prune_ratio) * 0.5
    tf.logging.info('\tprune_ratio = %.2f' % prune_ratio)
    var_w_bias = var + tf.random.uniform(var.shape, minval=0, maxval=1e-3)
    chn_norms = tf.reduce_sum(tf.square(var_w_bias), axis=(0, 1, 3), keepdims=True)
    threshold = tf.contrib.distributions.percentile(chn_norms, prune_ratio * 100)
    prune_ops += [var.assign(var_w_bias * tf.cast(chn_norms >= threshold, tf.float32))]
  sess.run(prune_ops)

def save_pb_model(sess, pb_path, pb_info):
  """Save a *.pb model to file.

  Args:
  * sess: TensorFlow session
  * pb_path: *.pb model's file path
  * pb_info: *.pb model's input/output information
  """

  graph_def = tf.get_default_graph().as_graph_def()
  output_node_names = [pb_info['output_name'].replace(':0', '')]
  graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
  tf.train.write_graph(
    graph_def, os.path.dirname(pb_path), os.path.basename(pb_path), as_text=False)
  tf.logging.info('model saved to: ' + pb_path)

def restore_pb_model(pb_path, pb_info, input_map=None):
  """Restore a *.pb model from file.

  Args:
  * pb_path: *.pb model's file path
  * pb_info: *.pb model's input/output information
  * input_map: a dictionary mapping input names (as strings) in graph_def to Tensor objects
  """

  # restore the model
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(pb_path, 'rb') as i_file:
    graph_def.ParseFromString(i_file.read())
  tf.import_graph_def(graph_def, input_map=input_map, name='')
  tf.logging.info('model restored from: ' + pb_path)

  # validate input/output nodes
  net_input = tf.get_default_graph().get_tensor_by_name(pb_info['input_name'])
  net_output = tf.get_default_graph().get_tensor_by_name(pb_info['output_name'])
  tf.logging.info('input: {} / {}'.format(net_input.name, net_input.shape))
  tf.logging.info('output: {} / {}'.format(net_output.name, net_output.shape))

def test_pb_model_impl(pb_path, pb_info, input_shape):
  """Test the *.pb model.

  Args:
  * pb_path: *.pb model's file path
  * pb_info: *.pb model's input/output information
  * input_shape: input tensor's shape
  """

  with tf.Graph().as_default() as graph:
    # create an randomized input tensor
    input_name = pb_info['input_name']
    output_name = pb_info['output_name']
    input_tf = tf.random_uniform(input_shape)

    # restore the model
    restore_pb_model(pb_path, pb_info, input_map={input_name: input_tf})

    # compute the overall FLOPs
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    tf.logging.info('FLOPs: %e' % flops.total_float_ops)

    # obtain input & output nodes and then test the model
    sess = create_session()
    output_tf = graph.get_tensor_by_name(output_name)
    for __ in range(FLAGS.nb_iters_wrmp):
      sess.run(output_tf)
    time_beg = timer()
    for __ in range(FLAGS.nb_iters_spdt):
      sess.run(output_tf)
    time_ave = (timer() - time_beg) * 1000.0 / FLAGS.nb_iters_spdt
    tf.logging.info('input shape: {}'.format(input_shape))
    tf.logging.info('\tave. speed: %.2f ms / batch' % time_ave)

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(output_tf, options=options, run_metadata=run_metadata)
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline.json', 'w') as o_file:
      o_file.write(chrome_trace)

def test_pb_model(pb_path, pb_info):
  """Test the *.pb model.

  Args:
  * pb_path: *.pb model's file path
  * pb_info: *.pb model's input/output information
  """

  if FLAGS.data_format == 'channels_first':
    test_pb_model_impl(pb_path, pb_info, input_shape=(49, 3, 299, 299))
    test_pb_model_impl(pb_path, pb_info, input_shape=(1, 3, 2048, 2048))
  else:
    test_pb_model_impl(pb_path, pb_info, input_shape=(49, 299, 299, 3))
    test_pb_model_impl(pb_path, pb_info, input_shape=(1, 2048, 2048, 3))

def convert_ckpt2pb_model(ckpt_path, pb_path):
  """Convert a *.ckpt model to *.pb model.

  Args:
  * ckpt_path: *.ckpt model's file path
  * pb_path: *.pb model's file path

  Returns:
  * pb_info: *.pb model's input/output information
  """

  print(tf.__version__,tf.__path__)
  with tf.Graph().as_default() as graph:
    # model definition
    K.set_learning_phase(0)
    if FLAGS.enbl_dst:
      with tf.variable_scope('distilled_model'):
        model_dst = MicroscopeModel(data_format=FLAGS.data_format)

    with tf.variable_scope(FLAGS.model_scope):
      model = MicroscopeModel(data_format=FLAGS.data_format)
      vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FLAGS.model_scope)
      print("vars_all", vars_all)

    # display input/output tensors' names
    tf.logging.info('inputs: {}'.format([x.name for x in model.inputs]))
    tf.logging.info('outputs: {}'.format([x.name for x in model.outputs]))
    assert len(model.inputs) == 1 and len(model.outputs) == 1
    pb_info = {
      'input_name': model.inputs[0].name,
      'output_name': model.outputs[0].name,
    }

    # # restore model weights from the *.ckpt model
    # sess = create_session()
    # saver = tf.train.Saver(vars_all)
    # graph = tf.get_default_graph()
    # # print("==============",graph.get_tensor_by_name("batch_normalization/beta:0"))
    # # ckpt_path = "/home/terse/code/python/PocketFlow/models/model_pruned/model.ckpt-10000.meta"
    # print("================",ckpt_path,vars_all)
    # saver.restore(sess, ckpt_path)
    # tf.logging.info('model weights restored from: ' + ckpt_path)

    # # apply fake pruning
    # if FLAGS.enbl_fake_prune:
    #   apply_fake_pruning(sess)

    # # write the graph to a *.pb file
    # graph_def = graph.as_graph_def()
    # graph_def = tf.graph_util.convert_variables_to_constants(
    #   sess, graph_def, [pb_info['output_name'].replace(':0', '')])
    # tf.train.write_graph(
    #   graph_def, os.path.dirname(pb_path), os.path.basename(pb_path), as_text=False)
    # tf.logging.info('*.pb model generated: ' + pb_path)

    # # test the *.pb model
    # test_pb_model(pb_path, pb_info)

    return pb_info

class PbModelConvertor(object):
  """Proto buffer model convertor."""

  def __init__(self, pb_path, pb_info):
    """Constructor function.

    Args:
    * pb_path: *.pb model's file path
    * pb_info: *.pb model's input/output information
    """

    self.pb_path = pb_path
    self.pb_info = pb_info

  def run(self):
    """Run the model convertor."""

    # prune output channels
    if FLAGS.enbl_co_prune:
      with tf.Graph().as_default() as graph:
        # restore the *.pb model
        self.sess = create_session()
        restore_pb_model(self.pb_path, self.pb_info)

        # build relation graphs between tensors & operations
        self.t2o_graph, self.o2o_graph, self.o2c_graph = \
          build_graphs(FLAGS.model_scope, self.pb_info['output_name'])
        self.conv2d_op_n_tensors = get_conv2d_op_n_tensors(FLAGS.model_scope)
        print("=========281",self.conv2d_op_n_tensors)

        # edit the graph by inserting alternative routines for convolutional layers
        with tf.name_scope(FLAGS.model_scope):
          print("=========================284",FLAGS.model_scope)
          op_outputs_old, op_outputs_new = self.__prune_output_channels()
        self.__swap_outputs(op_outputs_old, op_outputs_new)

        # save the *.pb model
        save_pb_model(self.sess, self.pb_path, self.pb_info)
        os._exit(-1)

      # # test the *.pb model
      # test_pb_model(self.pb_path, self.pb_info)

    # prune input channels
    if FLAGS.enbl_ci_prune:
      with tf.Graph().as_default() as graph:
        # restore the *.pb model
        self.sess = create_session()
        restore_pb_model(self.pb_path, self.pb_info)

        # build relation graphs between tensors & operations
        self.t2o_graph, self.o2o_graph, self.o2c_graph = \
          build_graphs(FLAGS.model_scope, self.pb_info['output_name'])
        self.conv2d_op_n_tensors = get_conv2d_op_n_tensors(FLAGS.model_scope)

        # edit the graph by inserting alternative routines for convolutional layers
        with tf.name_scope(FLAGS.model_scope):
          op_outputs_old, op_outputs_new = self.__prune_input_channels()
        self.__swap_outputs(op_outputs_old, op_outputs_new)

        # save the *.pb model
        save_pb_model(self.sess, self.pb_path, self.pb_info)

      # test the *.pb model
      test_pb_model(self.pb_path, self.pb_info)

  def __prune_output_channels(self):
    """Prune output channels for convolutional layers.

    Returns:
    * op_outputs_old: output nodes to be swapped in the old graph
    * op_outputs_new: output nodes to be swapped in the new graph
    """
    print("=====================323")
    def __is_conv2d_op(op):
      return op.name.split('/')[-1] == 'Conv2D'
    def __is_fused_batch_norm_op(op):
      return op.name.split('/')[-1] == 'FusedBatchNorm'
    def __is_relu_op(op):
      return op.name.split('/')[-1] == 'Relu'
    def __is_max_pool_op(op):
      return op.name.split('/')[-1] == 'MaxPool'

    op_outputs_old, op_outputs_new = [], []
    pruned_flags = {op: False for op, __ in self.conv2d_op_n_tensors}
    prunable_op_names = ['FusedBatchNorm', 'Relu', 'MaxPool', 'Conv2D']


    for op, var in self.conv2d_op_n_tensors:
      # skip if the current Conv2D operation has been pruned or has more than one consumer
      if pruned_flags[op] or len(self.o2c_graph[op]) != 1:
        continue
      print("===345")
      # find the longest path of which all Conv2D operations have only one consumer
      consumer_op = self.o2c_graph[op][0]
      while len(self.o2c_graph[consumer_op]) == 1:
        consumer_op = self.o2c_graph[consumer_op][0]
      op_path = [op]
      while op_path[-1] != consumer_op:
        ops_next = self.o2o_graph[op_path[-1]]['outputs']
        if len(ops_next) != 1 or ops_next[0].name.split('/')[-1] not in prunable_op_names:
          break
        op_path += [ops_next[0]]

      print("====357")
      while not __is_conv2d_op(op_path[-1]):
        op_path = op_path[:-1]
      if len(op_path) == 0:
        print("====362")
        continue
      print("====36")
      tf.logging.info('=== OP Path ===')
      for op in op_path:
        tf.logging.info('\t' + op.name)
        if __is_conv2d_op(op):
          pruned_flags[op] = True

      # obtain Conv2D & FusedBatchNorm operations
      conv2d_ops = [op for op in op_path if __is_conv2d_op(op)]
      fsbn_ops = [op for op in op_path if __is_fused_batch_norm_op(op)]
      assert len(conv2d_ops) == len(fsbn_ops) + 1

      # obtain model weights related to above operations
      weights_tf = {}
      for op in conv2d_ops:
        weights_tf[op.name] = op.inputs[1]  # kernel
      for op in fsbn_ops:
        weights_tf[op.name] = op.inputs[1:]  # scale / offset / mean / variance
      weights_np = self.sess.run(weights_tf)

      # obtain list of indices to non-zero channels
      idxs_nnz_dict = {}
      for op in conv2d_ops[1:]:
        filter_np = weights_np[op.name]
        nb_chns = filter_np.shape[2]
        idxs_nnz = np.nonzero(np.sum(np.abs(filter_np), axis=(0, 1, 3)))[0]
        tf.logging.info('%s: reducing %d channels to %d' % (op.name, nb_chns, idxs_nnz.size))
        idxs_nnz_dict[op.name] = idxs_nnz


      # replace Conv-BN-ReLU-(Pool)-Conv with cheaper operations
      x = op_path[0].inputs[0]
      for op in op_path:
        print("===============392",op)
        if __is_conv2d_op(op):
          filter_np = weights_np[op.name]
          if op != op_path[0]:
            filter_np = filter_np[:, :, idxs_nnz_dict[op.name], :]
          if op != op_path[-1]:
            filter_np = filter_np[:, :, :, idxs_nnz_dict[self.o2c_graph[op][0].name]]
          strides = op.get_attr('strides')
          padding = op.get_attr('padding').decode('utf-8')
          data_format = op.get_attr('data_format').decode('utf-8')
          dilations = op.get_attr('dilations')
          x = tf.nn.conv2d(
            x, filter_np, strides, padding, data_format=data_format, dilations=dilations)
        elif __is_fused_batch_norm_op(op):
          idxs_nnz = idxs_nnz_dict[self.o2c_graph[op][0].name]
          fsbn_scal_np = weights_np[op.name][0][idxs_nnz]
          fsbn_ofst_np = weights_np[op.name][1][idxs_nnz]
          fsbn_mean_np = weights_np[op.name][2][idxs_nnz]
          fsbn_varc_np = weights_np[op.name][3][idxs_nnz]
          epsilon = op.get_attr('epsilon')
          data_format = op.get_attr('data_format').decode('utf-8')
          is_training = op.get_attr('is_training')
          x, __, __ = tf.nn.fused_batch_norm(
            x, fsbn_scal_np, fsbn_ofst_np, mean=fsbn_mean_np, variance=fsbn_varc_np,
            epsilon=epsilon, data_format=data_format, is_training=is_training)
        elif __is_relu_op(op):
          x = tf.nn.relu(x)
        elif __is_max_pool_op(op):
          ksize = op.get_attr('ksize')
          strides = op.get_attr('strides')
          padding = op.get_attr('padding').decode('utf-8')
          data_format = op.get_attr('data_format').decode('utf-8')
          x = tf.nn.max_pool(x, ksize, strides, padding, data_format=data_format)
        else:
          raise NotImplementedError('unexpected operation: ' + op.name)

      # obtain old and new routines' outputs
      op_outputs_old += [op_path[-1].outputs[0]]
      op_outputs_new += [x]

    return op_outputs_old, op_outputs_new

  def __prune_input_channels(self):
    """Prune input channels for convolutional layers.

    Returns:
    * op_outputs_old: output nodes to be swapped in the old graph
    * op_outputs_new: output nodes to be swapped in the new graph
    """

    def __is_prunable_op(op):
      return op.type in ['FusedBatchNorm', 'Relu', 'MaxPool', 'AvgPool']

    op_outputs_old, op_outputs_new = [], []
    for op, var in self.conv2d_op_n_tensors:
      # detect which channels to be pruned
      filter_np = self.sess.run(var)
      nb_chns_all = filter_np.shape[2]
      idxs_nnz = np.nonzero(np.sum(np.abs(filter_np), axis=(0, 1, 3)))[0]
      nb_chns_nnz = idxs_nnz.size
      if nb_chns_nnz == nb_chns_all:
        tf.logging.info('%s: no channel can be pruned, skipping ...' % op.name)
        continue
      else:
        tf.logging.info('%s: reducing %d channels to %d' % (op.name, nb_chns_all, nb_chns_nnz))

      # find the earliest operation that can be pruned
      op_path = [op]
      op_prev = self.t2o_graph[op.inputs[0]]['inputs'][0]
      while True:
        if __is_prunable_op(op_prev) and len(self.o2o_graph[op_prev]['outputs']) == 1:
          op_path = [op_prev] + op_path
          op_prev = self.o2o_graph[op_prev]['inputs'][0]
        else:
          break
      tf.logging.info('=== OP Path ===')
      for op in op_path:
        tf.logging.info('\t' + op.name)

      # replace AvgPool-Conv with cheaper operations
      x = op_path[0].inputs[0]
      x = tf.gather(x, idxs_nnz, axis=(1 if FLAGS.data_format == 'channels_first' else 3))
      for op in op_path:
        if op.type == 'Conv2D':
          filter_np = filter_np[:, :, idxs_nnz, :]
          strides = op.get_attr('strides')
          padding = op.get_attr('padding').decode('utf-8')
          data_format = op.get_attr('data_format').decode('utf-8')
          dilations = op.get_attr('dilations')
          x = tf.nn.conv2d(
            x, filter_np, strides, padding, data_format=data_format, dilations=dilations)
        elif op.type == 'FusedBatchNorm':
          fsbn_scal_np, fsbn_ofst_np, fsbn_mean_np, fsbn_varc_np = self.sess.run(op.inputs[1:])
          fsbn_scal_np = fsbn_scal_np[idxs_nnz]
          fsbn_ofst_np = fsbn_ofst_np[idxs_nnz]
          fsbn_mean_np = fsbn_mean_np[idxs_nnz]
          fsbn_varc_np = fsbn_varc_np[idxs_nnz]
          epsilon = op.get_attr('epsilon')
          data_format = op.get_attr('data_format').decode('utf-8')
          is_training = op.get_attr('is_training')
          x, __, __ = tf.nn.fused_batch_norm(
            x, fsbn_scal_np, fsbn_ofst_np, mean=fsbn_mean_np, variance=fsbn_varc_np,
            epsilon=epsilon, data_format=data_format, is_training=is_training)
        elif op.type == 'Relu':
          x = tf.nn.relu(x)
        elif op.type == 'MaxPool':
          ksize = op.get_attr('ksize')
          strides = op.get_attr('strides')
          padding = op.get_attr('padding').decode('utf-8')
          data_format = op.get_attr('data_format').decode('utf-8')
          x = tf.nn.max_pool(x, ksize, strides, padding, data_format=data_format)
        elif op.type == 'AvgPool':
          ksize = op.get_attr('ksize')
          strides = op.get_attr('strides')
          padding = op.get_attr('padding').decode('utf-8')
          data_format = op.get_attr('data_format').decode('utf-8')
          x = tf.nn.avg_pool(x, ksize, strides, padding, data_format=data_format)
        else:
          raise NotImplementedError('unexpected operation: ' + op.name)

      # obtain old and new routines' outputs
      op_outputs_old += [op_path[-1].outputs[0]]
      op_outputs_new += [x]

    return op_outputs_old, op_outputs_new

  def __swap_outputs(self, op_outputs_old, op_outputs_new):
    """Swap two list of output nodes.

    Args:
    * op_outputs_old: output nodes to be swapped in the old graph
    * op_outputs_new: output nodes to be swapped in the new graph
    """

    tf.logging.info('# of output pairs to be swapped: %d' % len(op_outputs_old))
    self.sess.close()
    graph_editor.swap_outputs(op_outputs_old, op_outputs_new)
    self.sess = create_session()  # open a new session

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # configure file paths
    for fname in os.listdir(FLAGS.ckpt_model_dir):
      if fname.endswith('.meta'):
        ckpt_path = os.path.join(FLAGS.ckpt_model_dir, fname.replace('.meta', ''))
    pb_path_full = os.path.join(FLAGS.pb_model_dir, FLAGS.pb_fname_full)
    pb_path_prnd = os.path.join(FLAGS.pb_model_dir, FLAGS.pb_fname_prnd)

    # # convert *.ckpt model to *.pb model
    # pb_info = convert_ckpt2pb_model(ckpt_path, pb_path_full)
    # tf.logging.info('input_name: {}'.format(pb_info['input_name']))
    # tf.logging.info('output_name: {}'.format(pb_info['output_name']))

    # convert *.pb model with graph transformation (remove pruned channels)
    pb_info = {
      'input_name': "net_input:0",
      'output_name': "net_output:0",
    }

    if FLAGS.enbl_chn_prune:
      shutil.copyfile(pb_path_full, pb_path_prnd)
      convertor = PbModelConvertor(pb_path_prnd, pb_info)
      convertor.run()

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
