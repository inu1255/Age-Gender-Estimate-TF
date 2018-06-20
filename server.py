import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import inception_resnet_v1
from flask import Flask, request, jsonify
from scipy.misc import imread, imsave

app = Flask(__name__)

sess = tf.Session()
images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
train_mode = tf.placeholder(tf.bool)
age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8, phase_train=train_mode, weight_decay=1e-5)
gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./models')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("restore and continue training!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=160)

image = None
person_conf_multi = None

@app.route('/load', methods=['POST','GET'])
def load():
    global image
    f = None
    if request.method == 'POST':
        f = request.files.get('f')
    else:
        f = request.args.get('f')
    if f is None:
        return jsonify({'no': 400, "msg": "缺少文件参数"})
    try:
        image = imread(f, mode='RGB')
        # image = cv2.imread(f, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        return jsonify({'no': 404, 'msg': '找不到文件'})
    return jsonify({'no': 200})

@app.route('/detection')
def detection():
    global person_conf_multi
    global image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    rect_nums = len(rects)
    XY, data = [], []
    if rect_nums > 0:
        aligned_images = []
        for i in range(rect_nums):
            aligned_image = fa.align(image, gray, rects[i])
            aligned_images.append(aligned_image)
            (x, y, w, h) = rect_to_bb(rects[i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            XY.append([x, y, w, h])

        aligned_images = np.array(aligned_images)
        ages, genders = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})

        for i in range(len(XY)):
            data.append({'rect':XY[i],'age':float(ages[i]),'gender':int(genders[i])})

    return jsonify({'no':200,'data':data})
    
if __name__ == '__main__':
    app.run('127.0.0.1', 3009)