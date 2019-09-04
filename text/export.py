from config import kerasTextModel, keras_anchors, class_names
from text.keras_yolo3 import yolo_text, box_layer

from apphelper.image import resize_im, letterbox_image
from PIL import Image
import numpy as np
import tensorflow as tf

graph = tf.get_default_graph()  # 解决web.py 相关报错问题

anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_classes = len(class_names)

model = yolo_text(num_classes, anchors)
model.load_weights(kerasTextModel)

export_path = "models/yolov3/1"

with tf.keras.backend.get_session() as sess:

    image_shape = tf.constant([608.0,608.0],dtype=tf.float32)
    input_shape = tf.constant([608.0, 608.0], dtype=tf.float32)
    box_score = box_layer([*model.output, image_shape, input_shape], anchors, num_classes)

    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={'box': box_score[0], "text_score": box_score[1], "none_score":box_score[2]})
