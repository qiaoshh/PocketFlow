# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:52:06 2019

@author: loktarxiao
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "112"

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

K.set_learning_phase(0)

h5_path = "./h5.models/DS_inception_ex5.11.h5"
save_path = './pb.models'

def main():
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Conv2D
    from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input

    data_format = 'channels_last'
    input_shape = (299, 299, 3) if data_format == 'channels_last' else (3, 299, 299)
    #input_shape = (2048, 2048, 3) if data_format == 'channels_last' else (3, 2048, 2048)

    K.set_image_data_format(data_format)
    net = InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=input_shape)
    x = net.output
    x = Conv2D(2, 8, activation='softmax', name='output')(x)
    model = Model(inputs=net.input, outputs=x)
    model.load_weights(h5_path, by_name=True)
    converted_output_node_names = [node.op.name for node in model.outputs]

    print(('Converted output node names are: %s', str(converted_output_node_names)))

    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                converted_output_node_names)

    graph_io.write_graph(constant_graph, save_path, 'model.pb',
                         as_text=False)

    sess.close()

if __name__ == "__main__":
    main()

