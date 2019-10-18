import re
from collections import defaultdict
import tensorflow as tf

OP_PATTERN_CONV2D = re.compile(r'/Conv2D_*[0-9]*$')

def find_nearest_tensor(op, tensors):
  """Find the nearest tensor (within the candidate list) to the anchor operation.

  Example:
  * If both 'aa' and 'ff' belong to <tensors>, then <ff> will be returned.
  * If both 'bb' and 'ee' belong to <tensors>, then <bb> will be returned.
  * If both 'aa' and 'ee' belong to <tensors>, then either one may be returned.

  aa -> bb -> cc -> op
              ^
              |
  dd -> ee -> ff <- hh

  Args:
  * op: anchro operation
  * tensors: list of candidate tensors

  Returns:
  * tensor: nearest tensor to the anchor operation
  """

  tensor = None
  op_input_names = set([op_input.name for op_input in op.inputs])
  tensor_names = set([tensor.name for tensor in tensors])
  while tensor is None and op_input_names:
    intersection = op_input_names & tensor_names
    if intersection:
      for tensor in tensors:
        if tensor.name in intersection:
          break
    else:
      op_input_names_new = set()
      for op in tf.get_default_graph().get_operations():
        if op_input_names & set([op_output.name for op_output in op.outputs]):
          op_input_names_new |= set([op_input.name for op_input in op.inputs])
      op_input_names = op_input_names_new

  return tensor

def get_conv2d_op_n_vars(scope):
  """Get a list of Conv2D operation & kernel (as variable) tuples.

  Args:
  * scope: model scope

  Returns:
  * op_n_vars: list of Conv2D operation & kernel (as variable) tuples
  """

  op_n_vars = []
  trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  for op in tf.get_default_graph().get_operations():
    if op.name.startswith(scope) and re.search(OP_PATTERN_CONV2D, op.name):
      op_n_vars += [(op, find_nearest_tensor(op, trainable_vars))]

  return op_n_vars

def get_conv2d_op_n_tensors(scope):
  """Get a list of Conv2D operation & kernel (as tensor) tuples.

  Args:
  * scope: model scope

  Returns:
  * op_n_tensors: list of Conv2D operation & kernel (as tensor) tuples
  """

  op_n_tensors = []
  for op in tf.get_default_graph().get_operations():
    if op.name.startswith(scope) and re.search(OP_PATTERN_CONV2D, op.name):
      op_n_tensors += [(op, op.inputs[1])]

  return op_n_tensors
  '''
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  var_names = set([var.name for var in var_list])
  var_dict = {var.name: var for var in var_list}
  conv2d_ops = [op for op in tf.get_default_graph().get_operations()
                if op.name.startswith(scope) and re.search(OP_PATTERN_CONV2D, op.name)]
  for op in conv2d_ops:
    op_inputs = op.inputs
    op_input_names = set([op_input.name for op_input in op_inputs])
    while not (op_input_names & var_names):
      parent_ops = []
      for op_input in op_inputs:
        parent_ops += t2o_graph[op_input]['inputs']
      op_inputs = []
      for parent_op in parent_ops:
        op_inputs += parent_op.inputs
      op_input_names = set([op_input.name for op_input in op_inputs])
    assert len(op_input_names & var_names) == 1
    var_name = list(op_input_names & var_names)[0]
    op_n_vars += [(op, var_dict[var_name])]

  return op_n_vars
  '''

def build_graphs(scope, output_tensor_name=None):
  """Build relation graphs between tensors & operations.

  Args:
  * scope: model scope
  * output_tensor_name: output tensor's name

  Returns:
  * t2o_graph: tensor-to-operation graph
  * o2o_graph: operation-to-operation graph
  * o2c_graph: operation-to-consumer graph
  """

  # obtain all the operations within the given scope
  # ops = [op for op in tf.get_default_graph().get_operations() if op.name.startswith(scope)]
  ops = [op for op in tf.get_default_graph().get_operations()]


  # build the Tensor-to-Operation graph
  output_op_name = None
  t2o_graph = defaultdict(lambda: {'inputs': [], 'outputs': []})
  for op in ops:
    # print("============",op.name,op.inputs,op.outputs)
    for op_input in op.inputs:
      t2o_graph[op_input]['outputs'].append(op)
    for op_output in op.outputs:
      t2o_graph[op_output]['inputs'].append(op)
      # print("==",op_output.name,output_tensor_name)
      if op_output.name == output_tensor_name:
        output_op_name = op.name
  # os._exit(-1)

  # print("138=======================",t2o_graph)
  tf.logging.info('output operation: ' + output_op_name)

  # build the Operation-to-Operation graph
  o2o_graph = {op: {'inputs': [], 'outputs': []} for op in ops}
  for op in ops:
    for op_input in op.inputs:
      o2o_graph[op]['inputs'] += t2o_graph[op_input]['inputs']
    for op_output in op.outputs:
      o2o_graph[op]['outputs'] += t2o_graph[op_output]['outputs']

  # build the Operation-to-Consumer graph
  consumer_op_names = [op.name for op in ops if re.search(OP_PATTERN_CONV2D, op.name)]
  consumer_op_names += [output_op_name]
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
  o2c_graph = {op: list(o2c_graph[op]) for op in o2c_graph}
  for op in ops:
    if re.search(OP_PATTERN_CONV2D, op.name):
      tf.logging.info(op.name)
      tf.logging.info('\tconsumer(s): {}'.format([c.name for c in o2c_graph[op]]))

  return t2o_graph, o2o_graph, o2c_graph
