import sys

#sys.path.insert(0, 'D:/FELIOUNE/PSO_GD/PSO_Gradient_Desend/LUNA16Challege/Vnet')
#sys.path.insert(0, 'E:/LUNA 16/PSOGD v1/PSO_Gradient_Desend/LUNA16Challege/Vnet')

from ResNet3d.layer import (conv3d , normalizationlayer , resnet_Add , max_pool3d)
import tensorflow as tf
import numpy as np
import cv2
import os
import random

def conv_bn_relu_drop(x, W, B,pre_activations,activations,phase,image_z=None, height=None, width=None,scope=None):
    conv = conv3d(x, W) + B
    conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',scope=scope)    
    pre_activations.append(conv)    
    conv = tf.nn.relu(conv)
    activations.append(conv)
    return conv



def down_sampling(x, W, B ,pre_activations,activations,phase,image_z=None, height=None, width=None,scope=None):
    
    conv = conv3d(x, W, 2) + B
    conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',scope=scope)   
    pre_activations.append(conv)    
    conv = tf.nn.relu(conv)    
    activations.append(conv)
    return conv


def deconv_relu(x, W,B,pre_activations,activations,samefeture=False ):
    conv = deconv3d(x, W, samefeture, True) + B
    pre_activations.append(conv)
    conv = tf.nn.relu(conv)
    activations.append(conv)
    return conv


def conv_sigmod(x, W,B ,pre_activations,activations):
    conv = conv3d(x, W) + B
    pre_activations.append(conv)
    conv = tf.nn.sigmoid(conv)
    activations.append(conv)
    return conv

def full_connected_relu_drop(x, W, B, activefunction='relu', scope=None):
   
    FC = tf.matmul(x, W) + B
    if activefunction == 'relu':
        FC = tf.nn.relu(FC)
        FC = tf.nn.dropout(FC, drop)
    elif activefunction == 'softmax':
        FC = tf.nn.softmax(FC)
    return FC


# Serve data by batches
"""
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch
"""
"""
def cost(Y_gt, Y_pred):
        if(len(list(Y_gt.shape)))> 4 :
          Z, H, W, C = list(Y_gt.shape)[1:]
        else : 
          Z, H, W, C = list(Y_gt.shape)  
        smooth = 1e-5
        pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
        true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
        intersection = 2 * tf.reduce_sum(input_tensor=pred_flat * true_flat, axis=1) + smooth
        denominator = tf.reduce_sum(input_tensor=pred_flat, axis=1) + tf.reduce_sum(input_tensor=true_flat, axis=1) + smooth
        loss = -tf.reduce_mean(input_tensor=intersection / denominator)
        
        return loss
"""

def cost(Y_gt, Y_pred):
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_gt, logits=Y_pred))
   return cost

def accuracy(Y_gt, Y_pred):
        correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
        return accuracy

def derivative_cost(dice,Y_gt,Y_pred):   
   
    smooth = 1e-5   
    dY_pred = (tf.subtract(tf.multiply(2.0,Y_gt), dice))/(tf.reduce_sum(input_tensor=Y_pred) + tf.reduce_sum(input_tensor=Y_gt) + smooth)
    return dY_pred

def derivative_sigmoid(X) :
    X = tf.multiply(tf.nn.sigmoid(X) , (1 - tf.nn.sigmoid(X)))
    return X

def _create_conv_net(X, image_z, image_width, image_height, image_channel,position,phase):
    pre_activations = []
    activations = []
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # ResVGGnet model
    # layer1->convolution
    
    activations.append(inputX)
    
    layer0 = conv_bn_relu_drop(x=inputX,W=position[0],B=position[1],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer0')
    
    layer1 = conv_bn_relu_drop(x=layer0, W=position[2],B=position[3],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer1')
    
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    activations.append(layer1)
    
    # down sampling1
    down1 = max_pool3d(x=layer1, depth=True)        
    
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, W=position[4],B=position[5],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer2_1')
    
    layer2 = conv_bn_relu_drop(x=layer2, W=position[6],B=position[7],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer2_2')
    
    layer2 = resnet_Add(x1=down1, x2=layer2)
    activations.append(layer2)
    # down sampling2
    down2 = max_pool3d(x=layer2, depth=True)# layer3->convolution
    
    layer3 = conv_bn_relu_drop(x=down2, W=position[8],B=position[9],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, W=position[10],B=position[11],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    activations.append(layer3)
    # down sampling3
    down3 = max_pool3d(x=layer3, depth=True)
    
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, W=position[12],B=position[13],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, W=position[14],B=position[15],pre_activations=pre_activations,
                               activations=activations,phase=phase,scope='layer4_2')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    activations.append(layer4)
    # down sampling4
    down4 = max_pool3d(x=layer4, depth=True) # layer5->convolution
    
    layer5 = conv_bn_relu_drop(x=down4, W=position[16],B=position[17],pre_activations=pre_activations,
                               activations=activations,phase=phase, scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, W=position[18],B=position[19],pre_activations=pre_activations,
                               activations=activations,phase=phase, scope='layer5_2')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    activations.append(layer5)
    # global average pooling
    gap = tf.reduce_mean(layer5, axis=(1, 2, 3))
    
    # layer6->FC1
    layer6 = tf.reshape(gap, [-1, 256])  # shape=(?, 256)

    layer6 = full_connected_relu_drop(x=layer6, W=position[20],B=position[21], activefunction='relu',
                                      scope='fc1')
     # layer7->output
    output = full_connected_relu_drop(x=layer6, W=position[22],B=position[23], activefunction='softmax',
                                      scope='output')    
    return output 

class RestNet3dModule(object):
    def __init__(self, image_height, image_width, image_depth, channels=1, n_class=2):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels
        self.n_class = n_class
                

    def train(self, train_images , train_lanbels , position , batch_size):       
         
         #random.randrange(0, train_images.shape[0]-batch_size)
         # get new batch
         batch_xs_path, batch_ys_path = train_images, train_lanbels         
         batch_xs = np.empty((len(batch_xs_path), self.image_depth, self.image_height, self.image_width,
                                 self.channels))         
         self.phase = 1
         for num in range(len(batch_xs_path)):
          batchimage = np.reshape(np.load(batch_xs_path[num][0]),
                                        (self.image_depth, self.image_height, self.image_width, self.channels))
          batch_xs[num, :, :, :] = batchimage
         # Extracting images and labels from given data
         batch_xs = batch_xs.astype(np.float)
         batch_ys = batch_ys.astype(np.float)
         # Normalize from [0:255] => [0.0:1.0]
         batch_xs = np.multiply(batch_xs, 1.0 / 255.0)

         with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
             Y_pred =_create_conv_net(tf.convert_to_tensor(value=batch_xs)\
                                      ,self.image_depth, self.image_width, self.image_height, self.channels,position,self.phase)
             train_loss=cost(tf.convert_to_tensor(value=batch_ys),Y_pred)
             acc = accuracy(tf.convert_to_tensor(value=batch_ys),Y_pred)
             position_list = list(position)
             derivative_position = \
                            tape.gradient(train_loss,position_list)
                     
                     
                          
         return acc , derivative_position 

    def prediction(self, test_images,position,test_masks):
        test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
        test_images = test_images.astype(np.float)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        test_images=np.float32(test_images)

        test_masks = np.reshape(test_masks, (test_masks.shape[0], test_masks.shape[1], test_masks.shape[2], 1))
        test_masks = test_masks.astype(np.float)
        test_masks = np.multiply(test_masks, 1.0 / 255.0)
        test_masks=np.float32(test_masks)
        
        pred =_create_conv_net(tf.convert_to_tensor(value=test_images),\
              self.image_depth, self.image_width, self.image_height, self.channels,position,self.phase)
        train_loss=cost(tf.convert_to_tensor(value=test_masks),pred) 
        
        result = np.reshape(pred, (test_images.shape[0], test_images.shape[1], test_images.shape[2]))
        result = result.astype(np.float32) * 255.
        #result = np.clip(result, 0, 255).astype('uint8')
        return result,train_loss             

def weight_xavier_init_particule():
    # creating Tensor
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

    scope='layer2'
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

    scope='layer3' 
    kernal= (3, 3, 3, 32, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer3_1' 
    kernal= (3, 3, 3, 64, 64)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)

    scope='layer4'
    kernal= (3, 3, 3, 64, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer4_1'
    kernal= (3, 3, 3, 128, 128)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer5'
    kernal= (3, 3, 3, 128, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer5_1'
    kernal= (3, 3, 3, 256, 256)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'B')
    list.append(W)
    list.append(B)


    scope='layer6'
    kernal= (256, 512)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)

    scope='layer7'
    kernal= (512 , 2)
    W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
    B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
    list.append(W)
    list.append(B)   

    return list

def lunch():  
        
       
            list_weights_all_layers=weight_xavier_init_particule()            
                                    
            return list_weights_all_layers       
    









