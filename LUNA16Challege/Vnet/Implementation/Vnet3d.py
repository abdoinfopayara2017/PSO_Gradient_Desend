import sys

sys.path.insert(0, 'E:/LUNA 16/PSOGD/LUNA16Challege/Vnet')

from layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add,
                         weight_xavier_init, bias_variable, save_images)
import tensorflow as tf
import numpy as np
import cv2
import os
import random



def conv_bn_relu_drop(x, W, B,pre_activations,activations):
    conv = conv3d(x, W) + B
    pre_activations.append(conv.copy())    
    conv = tf.nn.relu(conv)
    activations.append(conv.copy())
    return conv



def down_sampling(x, W, B ,pre_activations,activations):
    
    conv = conv3d(x, W, 2) + B
    pre_activations.append(conv.copy())    
    conv = tf.nn.relu(conv)    
    activations.append(conv.copy())
    return conv


def deconv_relu(x, W,B,pre_activations,activations,samefeture=False ):
    conv = deconv3d(x, W, samefeture, True) + B
    pre_activations.append(conv.copy())
    conv = tf.nn.relu(conv)
    activations.append(conv.copy())
    return conv


def conv_sigmod(x, W,B ,pre_activations,activations):
    conv = conv3d(x, W) + B
    pre_activations.append(conv.copy())
    conv = tf.nn.sigmoid(conv)
    activations.append(conv.copy())
    return conv

# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size  
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

def cost(Y_gt, Y_pred):
        Z, H, W, C = Y_gt.get_shape().as_list()[1:]
        smooth = 1e-5
        pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
        true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
        loss = -tf.reduce_mean(intersection / denominator)
        return loss

def derivative_cost(dice,Y_gt,Y_pred):
    smooth = 1e-5
    dY_pred = (2* Y_gt - dice)/(np.sum(Y_pred)+np.sum(Y_gt)+smooth)
    return dY_pred

def derivative_sigmoid(X) :
    X = tf.nn.sigmoid(X) * (1 - tf.nn.sigmoid(X))

def _create_conv_net(X, image_z, image_width, image_height, image_channel,position,drop):
    pre_activations = []
    activations = []
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    
    activations.append(inputX)
    layer0 = conv_bn_relu_drop(x=inputX,W=position[0],B=position[1],pre_activations=pre_activations,
                               activations=activations)
    layer1 = conv_bn_relu_drop(x=layer0, W=position[2],B=position[3],pre_activations=pre_activations,
                               activations=activations)
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1,W=position[4],B=position[5],pre_activations=pre_activations,
                               activations=activations)
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, W=position[6],B=position[7],pre_activations=pre_activations,
                               activations=activations)
    layer2 = conv_bn_relu_drop(x=layer2, W=position[8],B=position[9],pre_activations=pre_activations,
                               activations=activations)
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, W=position[10],B=position[11],pre_activations=pre_activations,
                               activations=activations)
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, W=position[12],B=position[13],pre_activations=pre_activations,
                               activations=activations)
    layer3 = conv_bn_relu_drop(x=layer3, W=position[14],B=position[15],pre_activations=pre_activations,
                               activations=activations)
    layer3 = conv_bn_relu_drop(x=layer3, W=position[16],B=position[17],pre_activations=pre_activations,
                               activations=activations)
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, W=position[18],B=position[19],pre_activations=pre_activations,
                               activations=activations)
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, W=position[20],B=position[21],pre_activations=pre_activations,
                               activations=activations)
    layer4 = conv_bn_relu_drop(x=layer4, W=position[22],B=position[23],pre_activations=pre_activations,
                               activations=activations)
    layer4 = conv_bn_relu_drop(x=layer4, W=position[24],B=position[25],pre_activations=pre_activations,
                               activations=activations)
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, W=position[26],B=position[27],pre_activations=pre_activations,
                               activations=activations)
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, W=position[28],B=position[29],pre_activations=pre_activations,
                               activations=activations)
    layer5 = conv_bn_relu_drop(x=layer5, W=position[30],B=position[31],pre_activations=pre_activations,
                               activations=activations)
    layer5 = conv_bn_relu_drop(x=layer5, W=position[32],B=position[33],pre_activations=pre_activations,
                               activations=activations)
    layer5 = resnet_Add(x1=down4, x2=layer5)

    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, W=position[34],B=position[35],pre_activations=pre_activations,
                               activations=activations)
    # layer8->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    
    layer6 = conv_bn_relu_drop(x=layer6, W=position[36],B=position[37],pre_activations=pre_activations,
                               activations=activations)
    layer6 = conv_bn_relu_drop(x=layer6, W=position[38],B=position[39],pre_activations=pre_activations,
                               activations=activations)
    layer6 = conv_bn_relu_drop(x=layer6, W=position[40],B=position[41],pre_activations=pre_activations,
                               activations=activations)
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, W=position[42],B=position[43])
    # layer8->convolution
    layer7 = crop_and_concat(layer3, deconv2)
   
    layer7 = conv_bn_relu_drop(x=layer7, W=position[44],B=position[45],pre_activations=pre_activations,
                               activations=activations)
    layer7 = conv_bn_relu_drop(x=layer7, W=position[46],B=position[47],pre_activations=pre_activations,
                               activations=activations)
    layer7 = conv_bn_relu_drop(x=layer7, W=position[48],B=position[49],pre_activations=pre_activations,
                               activations=activations)
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, W=position[50],B=position[51],pre_activations=pre_activations,
                               activations=activations)
    # layer8->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    
    layer8 = conv_bn_relu_drop(x=layer8, W=position[52],B=position[53],pre_activations=pre_activations,
                               activations=activations)
    layer8 = conv_bn_relu_drop(x=layer8, W=position[54],B=position[55],pre_activations=pre_activations,
                               activations=activations)
    layer8 = conv_bn_relu_drop(x=layer8, W=position[56],B=position[57],pre_activations=pre_activations,
                               activations=activations)
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, W=position[58],B=position[59],pre_activations=pre_activations,
                               activations=activations)
    # layer8->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    
    layer9 = conv_bn_relu_drop(x=layer9, W=position[60],B=position[61],pre_activations=pre_activations,
                               activations=activations)
    layer9 = conv_bn_relu_drop(x=layer9, W=position[62],B=position[63],pre_activations=pre_activations,
                               activations=activations)
    layer9 = conv_bn_relu_drop(x=layer9, W=position[64],B=position[65],pre_activations=pre_activations,
                               activations=activations)
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigmod(x=layer9, W=position[68],B=position[69],pre_activations=pre_activations,
                               activations=activations)
    return output_map , pre_activations , activations

class Vnet3dModule(object):
    def __init__(self, image_height, image_width, image_depth, channels=1):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels        

    def train(self, train_images, train_lanbels,position,
               batch_size=1):
                
        index_in_epoch=random.randrange(0, train_images.shape[0]-batch_size)
        # get new batch
        batch_xs_path, batch_ys_path = _next_batch(train_images, train_lanbels, batch_size,index_in_epoch)
        batch_xs = np.empty((len(batch_xs_path), self.image_depth, self.image_height, self.image_width,
                                    self.channels))
        batch_ys = np.empty((len(batch_ys_path), self.image_depth, self.image_height, self.image_width,
                                    self.channels))
        for num in range(len(batch_xs_path)):
            index = 0
            for _ in os.listdir(batch_xs_path[num][0]):
                image = cv2.imread(batch_xs_path[num][0] + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(batch_ys_path[num][0] + "/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE)
                batch_xs[num, index, :, :, :] = np.reshape(image, (self.image_height, self.image_width,
                                                                        self.channels))
                batch_ys[num, index, :, :, :] = np.reshape(label, (self.image_height, self.image_width,
                                                                        self.channels))
                index += 1
            # Extracting images and labels from given data
        batch_xs = batch_xs.astype(np.float)
        batch_ys = batch_ys.astype(np.float)
        # Normalize from [0:255] => [0.0:1.0]
        batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
        batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
        # Make prediction
        Y_pred , pre_activation , activation =_create_conv_net(batch_xs,self.image_depth, self.image_width, self.image_height, self.channels,position)
        train_loss=cost(batch_ys,Y_pred)
        dY_pred=derivative_cost(train_loss,batch_ys,activation[-1])
        derisigmoid=derivative_sigmoid(pre_activation[-1])
        with tf.Session() as sess:
            sess.run(Y_pred)
            sess.run(train_loss)
            sess.run(dY_pred)
            sess.run(derisigmoid)
        return train_loss , dY_pred * derisigmoid , pre_activation , activation         
            


def weight_xavier_init_particule():
    # creating list
    list = []
    
    scope='layer0'
    kernal=(3, 3, 3, 1, 16)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer1'
    kernal= (3, 3, 3, 16, 16)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='down1'
    kernal= (3, 3, 3, 16, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer2_1'
    kernal= (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer2_2' 
    kernal= (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='down2' 
    kernal= (3, 3, 3, 32, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer3_1'
    kernal= (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer3_2'
    kernal= (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer3_3'
    kernal= (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='down3'
    kernal= (3, 3, 3, 64, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)


    scope='layer4_1'
    kernal= (3, 3, 3, 128, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer4_2'
    kernal= (3, 3, 3, 128, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer4_3'  
    kernal= (3, 3, 3, 128, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='down4'
    kernal= (3, 3, 3, 128, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer5_1'
    kernal= (3, 3, 3, 256, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer5_2' 
    kernal= (3, 3, 3, 256, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer5_3'  
    kernal= (3, 3, 3, 256, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='deconv1'
    kernal= (3, 3, 3, 128, 256) 
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-2]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer6_1'
    kernal=  (3, 3, 3, 256, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer6_2'
    kernal=  (3, 3, 3, 128, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer6_3'
    kernal= (3, 3, 3, 128, 128) 
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope= 'deconv2'
    kernal= (3, 3, 3, 64, 128) 
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-2]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer7_1'  
    kernal=  (3, 3, 3, 128, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer7_2'
    kernal=  (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer7_3'
    kernal= (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope= 'deconv3'  
    kernal= (3, 3, 3, 32, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-2]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer8_1'  
    kernal=  (3, 3, 3, 64, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer8_2' 
    kernal=  (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer8_3'
    kernal= (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope= 'deconv4'  
    kernal= (3, 3, 3, 16, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-2]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer9_1'  
    kernal=   (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer9_2' 
    kernal=  (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer9_3'   
    kernal= (3, 3, 3, 32, 32)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope= 'output'  
    kernal= (1, 1, 1, 32, 1)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    return list

def lunch():  
    list_weights_all_layers=weight_xavier_init_particule()
    init = tf.initialize_all_variables()    
    with tf.Session() as sess:
        sess.run(init)
        sess.run(list_weights_all_layers)   
        
    return list_weights_all_layers    
lunch()








