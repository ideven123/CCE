from __future__ import print_function


import tensorflow as tf 
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,Flatten
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from keras.layers import add 
from keras.regularizers import l2
from keras.models import Model

from functools import partial 

from keras import layers ,Sequential ,backend
import keras
import keras

from keras.layers import Flatten,BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.models import Sequential

from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import six
from keras.regularizers import l2
from keras.models import Model

from keras import backend as K
# from tensorflow import keras
# from tensorflow.keras import layers,Sequential


def get_model(inputs, model, dataset): 
    """
    Retrieve model using 
    Args: 
        model: cnn or resnet20 
        dataset: cifar10, mnist, or cifar100 
    Returns: 
        probability: softmax output 

    """
    if model == 'cnn': 
        if dataset == 'cifar10': 
            return cnn_cifar10(inputs)[0]
        elif dataset == 'cifar100': 
            return cnn_cifar100(inputs)[0]
        elif dataset == 'mnist': 
            return cnn_mnist(inputs)[0]
    elif model == 'resnet20': 
        num_classes = 100 if dataset == 'cifar100' else 10 
        # return resnet_v1(input=inputs, depth=20, num_classes=num_classes, dataset=dataset)[2]
        # return resnet18(input=inputs,num_classes=num_classes)
        return resnet18v1(input=inputs,layer_dims =[2,2,2,2] ,num_classes=num_classes)
        # model = ResnetBuilder().build_resnet18(input=inputs,numclass=num_classes)
        return model
    else: 
        raise ValueError


# class BasicBlock(layers.Layer):
#     def __init__(self,filter_num,stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
#         self.bn1=layers.BatchNormalization()
#         self.relu=layers.Activation('relu')

#         self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
#         self.bn2 = layers.BatchNormalization()
#         self.stride = stride
#         if stride != 1:
#             self.downsample=Sequential()
#             self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
#         else:
#             self.downsample=lambda x:x
#     def call(self,input,training=None):
#         out=self.conv1(input)
#         out=self.bn1(out)
#         out=self.relu(out)

#         out=self.conv2(out)
#         out=self.bn2(out)

#         identity=self.downsample(input)

#         print('self.stride',self.stride,out.shape , identity.shape)
#         output=layers.add([out,identity])
#         output=tf.nn.relu(output)
#         return output

def basic_block(filter_num,stride=1):
    conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
    bn1=layers.BatchNormalization()
    relu=layers.Activation('relu')
    conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
    bn2 = layers.BatchNormalization()
    # if stride != 1:
    #     downsample=Sequential()
    #     downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
    # else:
    #     downsample=lambda x:x
    
    res_block = Sequential([
        conv1,
        bn1,
        relu,
        conv2,
        bn2

    ])
    return res_block
    # x0 = conv1(input)
    # x0 = bn1(x0)
    # x0 = relu(x0)
    # x1 = conv2(x0)
    # x1 = bn2(x1)
    # identity= downsample(input)
    # output=layers.add([x1,identity])
    # output=tf.nn.relu(output)
    # return output

def build_resblock(filter_num,blocks,strides=1):
    res_blocks= Sequential()
        # may down sample
    res_blocks.add( basic_block(filter_num,strides))
        # just down sample one time
    for pre in range(1,blocks):
        res_blocks.add(basic_block(filter_num,stride = 1))
    return res_blocks

def resnet18v1(input,layer_dims,num_classes=10):
    stem=Sequential([
            layers.Conv2D(64,(7,7),strides=(2,2),padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
        ])
    layer1= build_resblock(64,layer_dims[0])
    layer2 = build_resblock(128, layer_dims[1],strides=2)
    layer3 = build_resblock(256, layer_dims[2], strides=2)
    layer4 = build_resblock(512, layer_dims[3], strides=2)
    relu=layers.Activation('relu')
    avgpool=layers.GlobalAveragePooling2D()
    fc=layers.Dense(units = num_classes,activation="softmax",kernel_initializer='he_normal')
    
    downsample1=lambda x:x
    downsample2_1 = layers.Conv2D(128,(1,1),strides=2)
    downsample2_2 = layers.Conv2D(256,(1,1),strides=2)
    downsample2_3 = layers.Conv2D(512,(1,1),strides=2)

    x0 = stem(input)
    x1 = layer1(x0)
    identity1 = downsample1(x0)
    x1 = layers.add([x1,identity1])
    x1 = relu(x1)

    print('x1:',x1.shape)    

    x2 = layer2(x1)
    identity2 = downsample2_1(x1)
    x2 = layers.add([x2,identity2])
    x2 = relu(x2)

    print('x2:',x2.shape)  

    x3 = layer3(x2)
    identity3 = downsample2_2(x2)
    x3 = layers.add([x3,identity3])
    x3 = relu(x3)

    print('x3:',x3.shape)  

    x4 = layer4(x3)
    identity4 = downsample2_3(x3)
    x4 = layers.add([x4,identity4])
    x4 = relu(x4)

    print('x4:',x4.shape)  

    pool = avgpool(x4)
    print('pool:',pool.shape) 
    output = fc(pool)
    model = Model(inputs=input, outputs=output)
    print('output:',output.shape)  
    return output


# class ResNet(keras.Model):
#     def __init__(self,layer_dims,num_classes=10):
#         super(ResNet, self).__init__()
#         # 预处理层
#         self.stem=Sequential([
#             layers.Conv2D(64,(7,7),strides=(2,2),padding='same'),
#             layers.BatchNormalization(),
#             layers.Activation('relu'),
#             layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
#         ])
#         # resblock
#         self.layer1=self.build_resblock(64,layer_dims[0])
#         self.layer2 = self.build_resblock(128, layer_dims[1],strides=2)
#         self.layer3 = self.build_resblock(256, layer_dims[2], strides=2)
#         self.layer4 = self.build_resblock(512, layer_dims[3], strides=2)

#         # there are [b,512,h,w]
#         # 自适应
#         self.avgpool=layers.GlobalAveragePooling2D()
#         self.fc=layers.Dense(units = num_classes,activation="softmax",kernel_initializer='he_normal')



#     def call(self,input,training=None):
#         print('input0:',input.shape)
#         x0=self.stem(input)
#         # print('input:',K.int_shape(x0),x0.shape)

#         x1=self.layer1(x0)
#         # print('layer1:',K.int_shape(x1),x1.shape)

#         x2=self.layer2(x1)
#         # print('layer2:',K.int_shape(x2),x2.shape)

#         x3=self.layer3(x2)
#         # print('x3block:',K.int_shape(x3),x3.shape)

#         x4=self.layer4(x3)
#         # [b,c]
#         # print('layer4:',K.int_shape(x4),x4.shape)
#         print('begin avg')
#         # x=self.avgpool(x)
#         # print('output0:',x.shape)
#         # # x = layers.Flatten()(x)
#         # print('output0:',x.shape)
#         # x=self.fc(x)
#         # print('output1:',x.shape)

#         # block_shape = K.int_shape(x4)
#         # print('block:',block_shape)
#         # pool2 = AveragePooling2D(pool_size=(1, 1),strides=(1, 1))(x4)
#         pool2 = self.avgpool(x4)
#         # print('pools',K.int_shape(pool2),pool2.shape)
#         # flatten1 = Flatten()(pool2)
#         # print('flatten',K.int_shape(flatten1),flatten1.shape)
#         x = Dense(units=10,activation="softmax")(pool2)
#         # print('out',K.int_shape(x),x.shape)
#         return x

#     def build_resblock(self,filter_num,blocks,strides=1):
#         res_blocks= Sequential()
#         # may down sample
#         res_blocks.add(BasicBlock(filter_num,strides))
#         # just down sample one time
#         for pre in range(1,blocks):
#             res_blocks.add(BasicBlock(filter_num,strides))
#         return res_blocks
# def resnet18(input,num_classes):
#     res = ResNet([2,2,2,2],num_classes)

#     return  res.call(input)

# class ResnetBuilder(object):
#     # @staticmethod
#     def build(self,input , numclass , block_fn, repetitions):
#         """Builds a custom ResNet like architecture.
#         Args:
#             input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
#             num_outputs: The number of outputs at final softmax layer
#             block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
#                 The original paper used basic_block for layers < 50
#             repetitions: Number of repetitions of various block units.
#                 At each block unit, the number of filters are doubled and the input size is halved
#         Returns:
#             The keras `Model`.
#         """

#         input_shape = input
#         num_outputs = numclass
#         self._handle_dim_ordering()
#         block_fn = self._get_block(block_fn)

#         input = Input(shape=input_shape)
#         conv1 = self._conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
#         pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

#         block = pool1
#         filters = 64
#         for i, r in enumerate(repetitions):
#             block = self._residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
#             filters *= 2

#         # Last activation
#         block = self._bn_relu(block)

#         # Classifier block
#         block_shape = K.int_shape(block)
#         pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
#                                  strides=(1, 1))(block)

#         flatten1 = Flatten()(pool2)
#         dense = Dense(units=num_outputs,
#                       activation="softmax")(flatten1)

#         model = Model(inputs=input, outputs=dense)
#         # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#         return model

#     # @staticmethod
#     def build_resnet18(self,params):
#         return self.build(params, self.basic_block, [2, 2, 2, 2])

#     # @staticmethod
#     def build_resnet34(self,params):
#         return self.build(params, self.basic_block, [3, 4, 6, 3])

#     # @staticmethod
#     def build_resnet50(self,params):
#         return self.build(params, self.bottleneck, [3, 4, 6, 3])

#     # @staticmethod
#     def build_resnet101(self,params):
#         return self.build(params, self.bottleneck, [3, 4, 23, 3])

#     # @staticmethod
#     def build_resnet152(self,params):
#         return self.build(params, self.bottleneck, [3, 8, 36, 3])

#     def _bn_relu(self,input):
#         """Helper to build a BN -> relu block
#         """
#         norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
#         return Activation("relu")(norm)

#     def _conv_bn_relu(self,**conv_params):
#         """Helper to build a conv -> BN -> relu block
#         """
#         filters = conv_params["filters"]
#         kernel_size = conv_params["kernel_size"]
#         strides = conv_params.setdefault("strides", (1, 1))
#         kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
#         padding = conv_params.setdefault("padding", "same")
#         kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

#         def f(input):
#             conv = Conv2D(filters=filters, kernel_size=kernel_size,
#                           strides=strides, padding=padding,
#                           kernel_initializer=kernel_initializer,
#                           kernel_regularizer=kernel_regularizer)(input)
#             return self._bn_relu(conv)

#         return f


#     def _bn_relu_conv(self,**conv_params):
#         """Helper to build a BN -> relu -> conv block.
#         This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
#         """
#         filters = conv_params["filters"]
#         kernel_size = conv_params["kernel_size"]
#         strides = conv_params.setdefault("strides", (1, 1))
#         kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
#         padding = conv_params.setdefault("padding", "same")
#         kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

#         def f(input):
#             activation = self._bn_relu(input)
#             return Conv2D(filters=filters, kernel_size=kernel_size,
#                           strides=strides, padding=padding,
#                           kernel_initializer=kernel_initializer,
#                           kernel_regularizer=kernel_regularizer)(activation)

#         return f


#     def _shortcut(self,input, residual):
#         """Adds a shortcut between input and residual block and merges them with "sum"
#         """
#         # Expand channles of shortcut to match residual.
#         # Stride appropriately to match residual (width, height)
#         # Should be int if network architecture is correctly configured.
#         input_shape = K.int_shape(input)
#         residual_shape = K.int_shape(residual)
#         stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
#         stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
#         equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

#         shortcut = input
#         # 1 X 1 conv if shape is different. Else identity.
#         if stride_width > 1 or stride_height > 1 or not equal_channels:
#             shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
#                               kernel_size=(1, 1),
#                               strides=(stride_width, stride_height),
#                               padding="valid",
#                               kernel_initializer="he_normal",
#                               kernel_regularizer=l2(0.0001))(input)

#         return add([shortcut, residual])


#     def _residual_block(self,block_function, filters, repetitions, is_first_layer=False):
#         """Builds a residual block with repeating bottleneck blocks.
#         """
#         def f(input):
#             for i in range(repetitions):
#                 init_strides = (1, 1)
#                 if i == 0 and not is_first_layer:
#                     init_strides = (2, 2)
#                 input = block_function(filters=filters, init_strides=init_strides,
#                                        is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
#             return input

#         return f


#     def basic_block(self,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
#         """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
#         Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
#         """
#         def f(input):

#             if is_first_block_of_first_layer:
#                 # don't repeat bn->relu since we just did bn->relu->maxpool
#                 conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
#                                strides=init_strides,
#                                padding="same",
#                                kernel_initializer="he_normal",
#                                kernel_regularizer=l2(1e-4))(input)
#             else:
#                 conv1 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3),
#                                       strides=init_strides)(input)

#             residual = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
#             return self._shortcut(input, residual)

#         return f


#     def bottleneck(self,filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
#         """Bottleneck architecture for > 34 layer resnet.
#         Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
#         Returns:
#             A final conv layer of filters * 4
#         """
#         def f(input):

#             if is_first_block_of_first_layer:
#                 # don't repeat bn->relu since we just did bn->relu->maxpool
#                 conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
#                                   strides=init_strides,
#                                   padding="same",
#                                   kernel_initializer="he_normal",
#                                   kernel_regularizer=l2(1e-4))(input)
#             else:
#                 conv_1_1 = self._bn_relu_conv(filters=filters, kernel_size=(1, 1),
#                                          strides=init_strides)(input)

#             conv_3_3 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
#             residual = self._bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
#             return self._shortcut(input, residual)

#         return f

#     def _handle_dim_ordering(self):
#         global ROW_AXIS
#         global COL_AXIS
#         global CHANNEL_AXIS
#         if K.image_dim_ordering() == 'tf':
#             ROW_AXIS = 1
#             COL_AXIS = 2
#             CHANNEL_AXIS = 3
#         else:
#             CHANNEL_AXIS = 1
#             ROW_AXIS = 2
#             COL_AXIS = 3

#     def _get_block(self,identifier):
#         if isinstance(identifier, six.string_types):
#             res = globals().get(identifier)
#             if not res:
#                 raise ValueError('Invalid {}'.format(identifier))
#             return res
#         return identifier



def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input, depth, num_classes=10, dataset='cifar10'):
    """ResNet Version 1 Model builder [a]
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
            print('layer1:',K.int_shape(x),x.shape)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    print('layer1:',K.int_shape(x),x.shape)
    final_features = Flatten()(x)
    print('layer1:',K.int_shape(final_features),final_features.shape)
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features

def cnn_cifar10(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=64)(inputs)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(10, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits


def cnn_mnist(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=32)(inputs)
    h = conv(filters=32)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=64)(h)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(200, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(200, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(10, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits

def cnn_cifar100(inputs): 
    """
    Standard CNN architecture 
    Ref: C&W 2016 
    """
    conv = partial(Conv2D, kernel_size=3, strides=1, padding='same', 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    h = conv(filters=64)(inputs)
    h = conv(filters=64)(h)
    h = conv(filters=64)(h)
    h = MaxPooling2D()(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = conv(filters=128)(h)
    h = MaxPooling2D()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    h = Dense(256, activation='relu', kernel_initializer='he_normal')(h)
    logits = Dense(100, kernel_initializer='he_normal')(h)
    probs = Activation('softmax')(logits)

    return probs, logits