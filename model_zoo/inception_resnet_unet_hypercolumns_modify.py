"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation
(which has some additional layers and different number of
filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim,
which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""

import keras 
from keras import *
from keras.layers import *
import tensorflow as tf

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2_UNet(use_activation = True):
    # Determine proper input shape
    img_input = Input(shape = (128,128,3))
    
    shortcut_layers = []
    # Stem block: 64 x 64 x 192
    x = conv2d_bn(img_input, 32, 3, strides=1, padding='same')
    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3)

    shortcut_layers.append(x)  #  128 x 128 x 64
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    
    shortcut_layers.append(x)   # 64 x 64 x 192
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Mixed 5b (Inception-A block): 32 x 32 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 32 x 32 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)


    shortcut_layers.append(x)   # 32 x 32 x 1088


    # Mixed 6a (Reduction-A block): 16 x 16 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

        # 20x block17 (Inception-ResNet-B block): 16 x 16 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    shortcut_layers.append(x)    #  16 x 16 x 1088



    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)
    
    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)

    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    model = models.Model(img_input,x)
    model.load_weights('/home/t-zhga/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    input_img = model.input
    x = model.output

    b1= Conv2D(256,(3,3),padding='same',name='fc6_1',dilation_rate=(1,1))(x)
    b1= BatchNormalization(axis=3,name='bn_b1_1')(b1)
    b1= Activation('relu')(b1)
    b1= Conv2D(256,(1,1),padding='same',name='fc7_1',)(b1)
    b1= BatchNormalization(axis=3,name='bn_b1_2')(b1)
    b1= Activation('relu')(b1)
    b1= Conv2D(128,(1,1),padding='same',name='fc8_1',)(b1)
    b1= BatchNormalization(axis=3,name='bn_b1_3')(b1)
    b1= Activation('relu')(b1)

    b2= Conv2D(256,(3,3),padding='same',name='fc6_2',dilation_rate=(3,3))(x)
    b2= BatchNormalization(axis=3,name='bn_b2_1')(b2)
    b2= Activation('relu')(b2)
    b2= Conv2D(256,(1,1),padding='same',name='fc7_2',)(b2)
    b2= BatchNormalization(axis=3,name='bn_b2_2')(b2)
    b2= Activation('relu')(b2)
    b2= Conv2D(128,(1,1),padding='same',name='fc8_2',)(b2)
    b2= BatchNormalization(axis=3,name='bn_b2_3')(b2)
    b2= Activation('relu')(b2)
    
    b3= Conv2D(256,(3,3),padding='same',name='fc6_3',dilation_rate=(5,5))(x)
    b3= BatchNormalization(axis=3,name='bn_b3_1')(b3)
    b3= Activation('relu')(b3)
    b3= Conv2D(256,(1,1),padding='same',name='fc7_3',)(b3)
    b3= BatchNormalization(axis=3,name='bn_b3_2')(b3)
    b3= Activation('relu')(b3)
    b3= Conv2D(128,(1,1),padding='same',name='fc8_3',)(b3)
    b3= BatchNormalization(axis=3,name='bn_b3_3')(b3)
    b3= Activation('relu')(b3)
    
    b4= Conv2D(256,(3,3),padding='same',name='fc6_4',dilation_rate=(7,7))(x)
    b4= BatchNormalization(axis=3,name='bn_b4_1')(b4)
    b4= Activation('relu')(b4)
    b4= Conv2D(256,(1,1),padding='same',name='fc7_4',)(b4)
    b4= BatchNormalization(axis=3,name='bn_b4_2')(b4)
    b4= Activation('relu')(b4)
    b4= Conv2D(128,(1,1),padding='same',name='fc8_4',)(b4)
    b4= BatchNormalization(axis=3,name='bn_b4_3')(b4)
    b4= Activation('relu')(b4)


    c1,c2,c3,c4 = shortcut_layers

    s=Concatenate()([b1,b2,b3,b4,x])                         
    
    
    uc4=Conv2DTranspose(256,(3,3),name='deconv_uc4',padding = 'same')(s)
    uc4= BatchNormalization(axis=3,name='bn_uc4')(uc4)
    uc4= Activation('relu')(uc4)

    hyper_up4 = UpSampling2D(size=(16,16),name='hyper_up_pool4')(uc4)

    up4 = UpSampling2D(size=(2, 2),name='up_pool4')(uc4)

    up4=Conv2DTranspose(256,(3,3),name='deconv_up4',padding = 'same')(up4)
    up4= BatchNormalization(axis=3,name='bn_up4')(up4)
    up4= Activation('relu')(up4)

    m4 = Concatenate()([up4,c4])
    uc3=Conv2DTranspose(256,(3,3),name='deconv_uc3',padding = 'same')(m4)
    uc3= BatchNormalization(axis=3,name='bn_uc3')(uc3)
    uc3= Activation('relu')(uc3)

    hyper_up3 = UpSampling2D(size=(8,8),name='hyper_up_pool3')(uc3)
    up3 = UpSampling2D(size=(2, 2),name='up_pool3')(uc3)

    up3=Conv2DTranspose(128,(3,3),name='deconv_up3',padding = 'same')(up3)
    up3= BatchNormalization(axis=3,name='bn_up3')(up3)
    up3= Activation('relu')(up3)

    m3=Concatenate()([up3,c3])

    uc2=Conv2DTranspose(128,(3,3),name='deconv_uc2',padding = 'same')(m3)
    uc2= BatchNormalization(axis=3,name='bn_uc2')(uc2)
    uc2= Activation('relu')(uc2)

    hyper_up2 = UpSampling2D(size=(4,4),name='hyper_up_pool2')(uc2)
    up2=UpSampling2D(size=(2, 2),name='up_pool2')(uc2)

    up2=Conv2DTranspose(128,(3,3),name='deconv_up2',padding = 'same')(up2)
    up2= BatchNormalization(axis=3,name='bn_up2')(up2)
    up2= Activation('relu')(up2)

    m2=Concatenate()([up2,c2])

    uc1=Conv2DTranspose(64,(3,3),name='deconv_uc1',padding = 'same')(m2)
    uc1= BatchNormalization(axis=3,name='bn_uc1')(uc1)
    uc1= Activation('relu')(uc1)

    up1=UpSampling2D(size=(2, 2),name='up_pool1')(uc1)


    up1=Conv2DTranspose(64,(3,3),name='deconv_up1',padding = 'same')(up1)
    up1= BatchNormalization(axis=3,name='bn_up1')(up1)
    up1= Activation('relu')(up1)

    m1=Concatenate()([up1,c1,hyper_up2,hyper_up3,hyper_up4])

    m1=Conv2DTranspose(64,(3,3),name='deconv_up0',padding = 'same')(m1)
    m1= BatchNormalization(axis=3,name='bn_up0')(m1)
    m1= Activation('relu')(m1)


    f1=Conv2D(32,(1,1),name='f1')(m1)
    if use_activation:
        pixel_prob = Conv2D(1, (1, 1), activation='sigmoid', name='pixel_prob')(f1)
    else:
        pixel_prob = Conv2D(1, (1, 1), name='pixel_prob')(f1)


    return input_img,pixel_prob 
