import tensorflow as tf

from microscope_model import MicroscopeModel

def create_session():
  """Create a TensorFlow session with GPU memory growth allowed.

  Returns:
  * sess: TensorFlow session
  """

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  return sess

### Main Entry

pb_path = './pb.models/model.pb'
ckpt_path = './ckpt.models/model.ckpt'

tf.logging.set_verbosity(tf.logging.INFO)

model_scope = 'model'
with tf.Graph().as_default() as graph:
  with tf.variable_scope(model_scope):
    model = MicroscopeModel()

  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  var_names = [var.name for var in var_list]

with tf.Graph().as_default() as graph:
  sess = create_session()
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(pb_path, 'rb') as i_file:
    graph_def.ParseFromString(i_file.read())
  tf.import_graph_def(graph_def)

  var_dict_tf = {}
  for var_name in var_names:
    tensor_name = var_name.replace(model_scope, 'import', 1)
    var_dict_tf[var_name] = graph.get_tensor_by_name(tensor_name)

  var_dict_np = sess.run(var_dict_tf)
  for var_name in var_dict_np:
    tf.logging.info('{}: {}'.format(var_name, var_dict_np[var_name].shape))

with tf.Graph().as_default() as graph:
  with tf.variable_scope(model_scope):
    model = MicroscopeModel()
    global_step = tf.train.get_or_create_global_step()

  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  init_ops = []
  for var in var_list:
    if var is global_step:
      init_ops += [var.initializer]
    else:
      init_ops += [var.assign(var_dict_np[var.name])]
    tf.logging.info(var.name)

  sess = create_session()
  sess.run(init_ops)
  saver = tf.train.Saver(var_list)
  saver.save(sess, ckpt_path)
