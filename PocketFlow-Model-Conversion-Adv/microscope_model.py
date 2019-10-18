from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Flatten
from tensorflow.python.keras.applications.inception_v3 import InceptionV3

# constants
IMAGE_SIZE  = (299, 299)
NUM_CLASSES = 2

class MicroscopeModel(object):
  """Microscope model based on the Inception-v3 architecture."""

  def __init__(self, data_format='channels_last'):
    """Constructor function."""

    # determine the input shape
    if data_format == 'channels_last':
      input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    elif data_format == 'channels_first':
      input_shape = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    else:
      raise ValueError('unrecognized data format: ' + data_format)

    # build the Inception-v3 backbone & final classification layer
    K.set_image_data_format(data_format)
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    # net = InceptionV3()
    # print("==============",net)
    net = InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, backend=K)
    x = net.output
    x = Conv2D(NUM_CLASSES, 8, data_format=data_format, activation='softmax', name='output')(x)
    x = Flatten(data_format=data_format)(x)
    net_final = Model(inputs=net.input, outputs=x)

    # obtain input & output tensors
    self.inputs = net_final.inputs
    self.outputs = net_final.outputs
