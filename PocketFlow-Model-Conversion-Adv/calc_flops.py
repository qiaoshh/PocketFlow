import os
import sys
import re
from collections import defaultdict
import tensorflow as tf
import horovod.tensorflow as hvd

# This script computes the reduction in computational complexity, measured by FLOPs.
#
# Rules:
# * If a Conv2D operation has exactly one Conv2D operation as its consumer, then both input and
#     output channels can be reduced.
# * If a Conv2D operation have no (or more than two) Conv2D operation(s) as its consumer, then only
#     input channel can be reduced.

OP_PATTERN_CONV2D = re.compile(r'/Conv2D$')
TERMINAL_OP_NAME = 'output/truediv'

def get_meta_path(model_dir):
  meta_path = None
  pattern = re.compile(r'\.meta$')
  for file_name in os.listdir(model_dir):
    if re.search(pattern, file_name):
      assert meta_path is None, 'multiple *.meta files found in ' + model_dir
      meta_path = os.path.join(model_dir, file_name)

  return meta_path

def get_conv2d_ops(scope=''):
  ops = [op for op in tf.get_default_graph().get_operations()
         if op.name.startswith(scope) and re.search(OP_PATTERN_CONV2D, op.name)]

  return ops

def get_inputs_n_outputs(ops):
  tensors = set()
  for op in ops:
    tensors.update(set(op.inputs))
    tensors.update(set(op.outputs))
  tensors = list(tensors)
  tensors.sort(key=lambda x: x.name)

  return tensors

def build_graphs(scope=''):
  """Build the following graphs:

  1. Tensor-to-Operation graph
  2. Operation-to-Operation graph
  3. Operation-to-Consumer graph
  """

  # obtain all the operations & tensors within the given scope
  ops = [op for op in tf.get_default_graph().get_operations() if op.name.startswith(scope)]
  tensors = get_inputs_n_outputs(ops)

  # build the Tensor-to-Operation graph
  t2o_graph = defaultdict(lambda: {'inputs': [], 'outputs': []})
  for op in ops:
    for op_input in op.inputs:
      t2o_graph[op_input]['outputs'].append(op)
    for op_output in op.outputs:
      t2o_graph[op_output]['inputs'].append(op)

  # build the Operation-to-Operation graph
  o2o_graph = {op: {'inputs': [], 'outputs': []} for op in ops}
  for op in ops:
    for op_input in op.inputs:
      o2o_graph[op]['inputs'] += t2o_graph[op_input]['inputs']
    for op_output in op.outputs:
      o2o_graph[op]['outputs'] += t2o_graph[op_output]['outputs']

  # build the Operation-to-Consumer graph
  consumer_op_names = [op.name for op in ops if re.search(OP_PATTERN_CONV2D, op.name)]
  consumer_op_names += ['%s/%s' % (scope, TERMINAL_OP_NAME)]
  def __find_consumers(op, o2o_graph, o2c_graph, visit_flags, final_flags):
    visit_flags[op] = True
    for op_out in o2o_graph[op]['outputs']:
      if op_out.name in consumer_op_names:
        o2c_graph[op].add(op_out)
      else:
        if not final_flags[op_out]:
          __find_consumers(op_out, o2o_graph, o2c_graph, visit_flags, final_flags)
        o2c_graph[op].update(o2c_graph[op_out])
    final_flags[op] = True

  o2c_graph = {op: set() for op in ops}
  visit_flags = {op: False for op in ops}
  final_flags = {op: False for op in ops}
  for op in ops:
    __find_consumers(op, o2o_graph, o2c_graph, visit_flags, final_flags)
  for op in ops:
    o2c_graph[op] = list(o2c_graph[op])
  for op in ops:
    if re.search(OP_PATTERN_CONV2D, op.name):
      tf.logging.info(op.name)
      tf.logging.info('\tconsumer(s): {}'.format([c.name for c in o2c_graph[op]]))

  return t2o_graph, o2o_graph, o2c_graph

### Main Entry

assert len(sys.argv) == 3, '[USAGE] python calc_flops.py <model_dir> <model_scope>'
model_dir = sys.argv[1]
model_scope = sys.argv[2]  # 'model' / 'pruned_model'

tf.logging.set_verbosity(tf.logging.INFO)
with tf.Graph().as_default() as graph:
  # create a TensorFlow session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  # restore the model
  meta_path = get_meta_path(model_dir)
  tf.logging.info('meta path: ' + meta_path)
  saver = tf.train.import_meta_graph(meta_path)
  saver.restore(sess, meta_path.replace('.meta', ''))

  # locate all the Conv2D operations
  conv2d_ops = get_conv2d_ops(model_scope)

  # build graphs for FLOPs computation
  t2o_graph, o2o_graph, o2c_graph = build_graphs(model_scope)

  # compute the pruning ratio of input channels
  def __calc_preserve_ratio(krnl):
    return tf.reduce_mean(tf.cast(tf.reduce_sum(tf.square(krnl), axis=[0, 1, 3]) > 0, tf.float32))

  preserve_ratios = [__calc_preserve_ratio(op.inputs[1]) for op in conv2d_ops]
  preserve_ratios_np = sess.run(preserve_ratios)
  conv2d_info_list = []
  for idx, op in enumerate(conv2d_ops):
    ifmap = op.inputs[0]
    krnl = op.inputs[1]
    ofmap = op.outputs[0]
    kh, kw = krnl.shape[0], krnl.shape[1]
    ih, iw, ic = ifmap.shape[1], ifmap.shape[2], ifmap.shape[3]
    oh, ow, oc = ofmap.shape[1], ofmap.shape[2], ofmap.shape[3]
    flops = int(oh * ow * oc * kh * kw * ic)
    conv2d_info = {
      'op': op,
      'flops': flops,
      'pr_in': preserve_ratios_np[idx],
      'pr_out': 1.0,  # default value
    }
    conv2d_info_list.append(conv2d_info)

  # update the pruning ratio of output channels
  for conv2d_info in conv2d_info_list:
    consumers = o2c_graph[conv2d_info['op']]
    if len(consumers) == 1 and re.search(OP_PATTERN_CONV2D, consumers[0].name):
      for conv2d_info_out in conv2d_info_list:
        if conv2d_info_out['op'] == consumers[0]:
          conv2d_info['pr_out'] = conv2d_info_out['pr_in']
          tf.logging.info('%s: <pr_out> -> %f' % (conv2d_info['op'].name, conv2d_info['pr_out']))

  # display all the Conv2D operations' information
  for conv2d_info in conv2d_info_list:
    tf.logging.info('op: ' + conv2d_info['op'].name)
    tf.logging.info('\tFLOPs: %d' % conv2d_info['flops'])
    tf.logging.info('\tpreserve ratio - in: %.4f' % conv2d_info['pr_in'])
    tf.logging.info('\tpreserve ratio - out: %.4f' % conv2d_info['pr_out'])

  # compute the overall FLOPs (before & after channel pruning)
  flops_full, flops_prnd = 0, 0
  for conv2d_info in conv2d_info_list:
    flops_full += conv2d_info['flops']
    flops_prnd += int(conv2d_info['flops'] * conv2d_info['pr_in'] * conv2d_info['pr_out'])
  tf.logging.info('FLOPs: %.2e (full) / %.2e (prnd)' % (flops_full, flops_prnd))
  tf.logging.info('FLOPs\'s ratio: %.4f' % (flops_prnd / flops_full))
