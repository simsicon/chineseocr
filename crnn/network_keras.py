import tensorflow as tf

K = tf.keras.backend
Conv2D = tf.keras.layers.Conv2D
Add = tf.keras.layers.Add
ZeroPadding2D = tf.keras.layers.ZeroPadding2D
UpSampling2D = tf.keras.layers.UpSampling2D
Concatenate = tf.keras.layers.Concatenate
MaxPooling2D = tf.keras.layers.MaxPooling2D
Input = tf.keras.layers.Input
LeakyReLU = tf.keras.layers.LeakyReLU
BatchNormalization = tf.keras.layers.BatchNormalization
Lambda = tf.keras.layers.Lambda
concatenate = tf.keras.layers.concatenate
Model = tf.keras.models.Model
l2 = tf.keras.regularizers.l2
l1 = tf.keras.regularizers.l1
Activation = tf.keras.layers.Activation
relu = tf.keras.activations.relu
MaxPool2D = tf.keras.layers.MaxPool2D
Permute = tf.keras.layers.Permute
Reshape = tf.keras.layers.Reshape
Dense = tf.keras.layers.Dense

def __tf_jpeg_process(data, image_height):

    # The whole jpeg encode/decode dance is neccessary to generate a result
    # that matches the original model's (caffe) preprocessing
    # (as good as possible)
    image = tf.image.decode_jpeg(data, channels=1,
                                 fancy_upscaling=True,
                                 dct_method="INTEGER_FAST")

    initial_height = tf.to_float(tf.shape(image)[0])
    initial_width = tf.to_float(tf.shape(image)[1])

    ratio = tf.to_float(initial_height / image_height)

    new_width = tf.to_int32(initial_width / ratio)
    new_height = tf.to_int32(image_height)

    image = tf.image.resize_images(image, (new_height, new_width),
                               method=tf.image.ResizeMethod.BILINEAR)

    image = (tf.math.divide(image, 255) - 0.5)/0.5

    image = tf.transpose(image, perm=[2,0,1])

    return image


def load_base64_tensor(_input, image_height):

    def decode_and_process(base64):
        _bytes = tf.decode_base64(base64)
        _image = __tf_jpeg_process(_bytes, image_height)

        return _image

    # we have to do some preprocessing with map_fn, since functions like
    # decode_*, resize_images and crop_to_bounding_box do not support
    # processing of batches
    image = tf.map_fn(decode_and_process, _input,
                      back_prop=False, dtype=tf.float32)

    return image


def keras_crnn(imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, lstmFlag=True):
    """
    基于pytorch 实现 keras dense ocr
    pytorch lstm 层暂时无法转换为 keras lstm层
    """
    data_format = 'channels_first'
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    imgStrInput = Input(batch_shape=(None,), dtype=tf.string, name="imgInput")
    imgInput = Lambda(load_base64_tensor, arguments={"image_height": imgH})(imgStrInput)

    # imgInput = Input(shape=(1, imgH, None), name='imgInput')

    def convRelu(i, batchNormalization=False, x=None):
        # padding: one of `"valid"` or `"same"` (case-insensitive).
        ##nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        if leakyRelu:
            activation = LeakyReLU(alpha=0.2)
        else:
            activation = Activation(relu, name='relu{0}'.format(i))

        x = Conv2D(filters=nOut,
                   kernel_size=ks[i],
                   strides=(ss[i], ss[i]),
                   padding='valid' if ps[i] == 0 else 'same',
                   dilation_rate=(1, 1),
                   activation=None, use_bias=True, data_format=data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)

        if batchNormalization:
            # torch nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
            # affine=True, track_running_stats=True)
            x = BatchNormalization(
                epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)

        x = activation(x)
        return x

    x = imgInput
    x = convRelu(0, batchNormalization=False, x=x)

    #x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(
        0), padding='valid', data_format=data_format)(x)
    x = convRelu(1, batchNormalization=False, x=x)
    #x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)

    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(
        1), padding='valid', data_format=data_format)(x)

    x = convRelu(2, batchNormalization=True, x=x)
    x = convRelu(3, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid',
                  name='cnn.pooling{0}'.format(2), data_format=data_format)(x)

    x = convRelu(4, batchNormalization=True, x=x)
    x = convRelu(5, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid',
                  name='cnn.pooling{0}'.format(3), data_format=data_format)(x)
    x = convRelu(6, batchNormalization=True, x=x)

    x = Permute((3, 2, 1))(x)

    x = Reshape((-1, 512))(x)

    out = None
    if lstmFlag:
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(nh))(x)
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        out = TimeDistributed(Dense(nclass))(x)
    else:
        out = Dense(nclass, name='linear')(x)
    out = Reshape((-1, 1, nclass), name='out')(out)

    return Model(imgStrInput, out)
