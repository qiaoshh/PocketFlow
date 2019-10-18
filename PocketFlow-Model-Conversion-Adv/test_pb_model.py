%pylab inline
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib as mpl

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

classify_model = "../train/pb_models/inception_ex5.11/patch_model.pb"
fcn_model = "../train/pb_models/inception_ex5.12/fcn_model.pb"

INDEXES = [
    [0,299],        [292, 591],     [584, 883],     [876, 1175],
    [1168, 1467],   [1460, 1759],   [1749, 2048]
]

def _image_read(img_path):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]

    return img

def _preprocess(image_array):
    x = image_array / 127.5
    x = x - 1

    return x

def get_patches(image):
    image_array = np.array(image, np.uint8)
    image_array = _preprocess(image_array)
    tmp_img = image_array.copy()
    res = []
    for x in INDEXES:
        for y in INDEXES:
            patch = tmp_img[x[0]:x[1], y[0]:y[1], :]
            res.append(patch)

    return np.stack(res, axis=0)

def get_whole_image(image):
    img = np.array(image, np.uint8)
    img = np.pad(img, ((119, 199), (160, 160), (0, 0)), mode='constant', constant_values=0)
    img = cv2.resize(img, (2048, 2048))
    x = _preprocess(img)
    x = np.expand_dims(x, axis=0)

    return x

def load_fcn_model(pb_path, tfconfig):
    output_graph_def = tf.GraphDef()
    output_graph_path = pb_path
    g = tf.Graph()
    with g.as_default():
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session(config=tfconfig)
        input_image = sess.graph.get_tensor_by_name("input_1:0")
        prediction = sess.graph.get_tensor_by_name("output/truediv:0")

    return sess, input_image, prediction

def load_classify_model(pb_path, tfconfig):
    output_graph_def = tf.GraphDef()
    output_graph_path = pb_path
    g = tf.Graph()
    with g.as_default():
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session(config=tfconfig)
        input_image = sess.graph.get_tensor_by_name("input_1:0")
        prediction = sess.graph.get_tensor_by_name("output/truediv:0")

    return sess, input_image, prediction

def init_model():
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess1, input_whole_image, fcn_prediction = load_fcn_model(fcn_model, tfconfig)
    sess2, input_patch_image, patch_prediction = load_classify_model(classify_model, tfconfig)

    return dict(sess1=sess1, input_whole_image=input_whole_image, fcn_prediction=fcn_prediction,
                sess2=sess2, input_patch_image=input_patch_image, patch_prediction=patch_prediction)

def inference(model, image):
    sess1 = model["sess1"]
    sess2 = model["sess2"]
    input_whole_image = model["input_whole_image"]
    input_patch_image = model["input_patch_image"]
    fcn_prediction = model["fcn_prediction"]
    patch_prediction = model["patch_prediction"]

    whole_img, patch_img = get_whole_image(image), get_patches(image)

    fcn_res = sess1.run(fcn_prediction, feed_dict={input_whole_image: whole_img})
    patch_res = sess2.run(patch_prediction, feed_dict={input_patch_image: patch_img})

    feature_map = fcn_res[0, :, :, 1]
    probability = patch_res[:, 0, 0, 1]

    return feature_map, probability

def gray2jet(gray_img):
    jet_cmp = mpl.cm.get_cmap('jet')

    return jet_cmp(gray_img)[:,:,:3]

def image_show(img, feature):
    jet_feature = cv2.resize(gray2jet(feature), img.shape[:2])
    out = img/255. + 0.4 * jet_feature
    show_img = np.concatenate((img/255., out), axis=1)

    return cv2.resize(show_img, (512, 256))

def cal_f1(feature, mask, thres=0.5):
    y_pred = np.array(feature > thres, np.int).reshape(-1)
    y_true = np.array(cv2.resize(mask, (55, 55)) > 0.2, np.int).reshape(-1)

    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    return p, r, f1

### Main Entry ###

model = init_model()

img_dataset = "./datasets/DS-1-X4/images/"
msk_dataset = "./datasets/DS-1-X4/masks/"
from glob import glob
from tqdm import tqdm

img_lst = glob(img_dataset+'*.tif')
valid_lst = ["s101611", "s102617", "s115844", "s1402230", "s1402292"]
model = init_model()

tmp_lst = []
for i in tqdm(img_lst[:]):
    name = i.split("/")[-1].split(".")[0]
    sid = name.split("+")[0]
    if sid not in valid_lst:
        continue
    img = _image_read(i)
    if not os.path.exists(msk_dataset+name+'_mask.png'):
        msk = np.ones((256, 256))
        feature, prob = inference(model, img)
        feature = 1 - feature
    else:
        msk = _image_read(msk_dataset+name+'_mask.png')
        msk = msk[:,:,0]/255.
        feature, prob = inference(model, img)

    tmp_lst.append([feature, msk])

result_lst = []
for feature, msk in tmp_lst:
    p, r, f1 = cal_f1(feature, msk, 0.51)
    result_lst.append([p, r, f1])
results = np.array(result_lst)
results.mean(axis=0)
