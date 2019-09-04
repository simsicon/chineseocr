# coding:utf-8
from crnn.utils import strLabelConverter, resizeNormalize

from crnn.network_keras import keras_crnn as CRNN
import tensorflow as tf

from crnn import keys
from config import ocrModelKeras
import numpy as np

graph = tf.get_default_graph()

alphabet = keys.alphabetChinese
model = CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=False)
model.load_weights(ocrModelKeras)

export_path = "models/crnn/1"

with tf.keras.backend.get_session() as sess:

  out = model.output

  keys = [0]
  values = [" "]

  for idx, char in enumerate(alphabet):
    keys.append(idx+1)
    values.append(char)

  table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "ç")

  lookup_index = tf.math.argmax(out, axis=3, output_type=tf.dtypes.int32)
  texts = table.lookup(lookup_index)

  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  tf.saved_model.simple_save(
      sess,
      export_path,
      inputs={'input_image': model.input},
      outputs={'out': texts},
      legacy_init_op=legacy_init_op)

# def crnnOcr(image):
#        """
#        crnn模型，ocr识别
#        image:PIL.Image.convert("L")
#        """
#        scale = image.size[1]*1.0 / 32
#        w = image.size[0] / scale
#        w = int(w)
#        transformer = resizeNormalize((w, 32))
#        image = transformer(image)
#        image = image.astype(np.float32)
#        image = np.array([[image]])
#        global graph
#        with graph.as_default():
#           preds       = model.predict(image)

#        preds = preds[0]
#        preds = np.argmax(preds,axis=2).reshape((-1,))

#        sim_pred  = converter.decode(preds)
#        return sim_pred
