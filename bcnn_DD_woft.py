'''
This file is used for the first step of the training procedure of Bilinear_CNN
where only last layer of the Bilinear_CNN (DD) model is trained. 
Two VGG16 networks are connected at the output of conv5_3 layer to form 
a Bilinear_CNN (DD) network and bilinear merging is performed on connect 
these two convolutional layers.
No finetuning is performed on the convolutional layers.
Only blinear layers are trained in this first step.
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle

import pickle 
from tflearn.data_utils import image_preloader
import h5py
import math
import logging
import random
import time
#import tensorflow as tf
from count_sketch import bilinear_pool
import scipy


os.environ["CUDA_VISIBLE_DEVICES"]="4"




def random_flip_right_to_left(image_batch):
    result = []
    for n in range(image_batch.shape[0]):
        if bool(random.getrandbits(1)):
            result.append(image_batch[n][:,::-1,:])
        else:
            result.append(image_batch[n])
    return result


def _random_blur(self, batch, sigma_max):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Random sigma
            sigma = random.uniform(0., sigma_max)
            batch[i] = \
                scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []     ## Parameters in this list will be optimized when only last layer is being trained 
        self.parameters = []                ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()                   ## Create Convolutional layers
        self.fc_layers()                    ## Create Fully connected layer
        self.weight_file = weights          
        #self.load_weights(weights, sess)


    def convlayers(self):
        
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            #images = self.imgs-mean
            images=tf.cast(self.imgs,tf.float32)*(1./255)-0.5
	    print('Adding Data Augmentation')
            

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64],  dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                   trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,   name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),  trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False,  name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                  trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]



        print('Shape of conv5_3', self.conv5_3.get_shape())
        project_size = 16000
	
        self.conv5_3_trans=tf.reshape(self.conv5_3,[-1,28*28*512])

        self.compact_pooled = bilinear_pool(self.conv5_3_trans, self.conv5_3_trans, project_size)
        print("Shape of Comapact bilinear pool",self.compact_pooled.get_shape() )

        self.phi_I = tf.divide(self.compact_pooled, 784.0)
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I), tf.sqrt(tf.abs(self.phi_I) + 1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        print('Shape of z_l2', self.z_l2.get_shape())

        # self.phi_I = tf.einsum('ijkm,ijkn->imn',self.conv5_3,self.conv5_3)
        # print('Shape of phi_I after einsum', self.phi_I.get_shape())
        ###############################################################################

        # print('Shape of conv5_3', self.conv5_3.get_shape())
        # self.phi_I = tf.einsum('ijkm,ijkn->imn',self.conv5_3,self.conv5_3)
        # print('Shape of phi_I after einsum', self.phi_I.get_shape())
        #
        #
        # self.phi_I = tf.reshape(self.phi_I,[-1,512*512])
        # print('Shape of phi_I after reshape', self.phi_I.get_shape())
        #
        # self.phi_I = tf.divide(self.phi_I,784.0)
        # print('Shape of phi_I after division', self.phi_I.get_shape())
        #
        # self.y_ssqrt = tf.multiply(tf.sign(self.phi_I),tf.sqrt(tf.abs(self.phi_I)+1e-12))
        # print('Shape of y_ssqrt', self.y_ssqrt.get_shape())
        #
        # self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        # print('Shape of z_l2', self.z_l2.get_shape())




    def fc_layers(self):

        with tf.name_scope('fc-new') as scope:

            fc3w = tf.get_variable('weights', [16000, 6], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            #fc3w = tf.get_variable('weights', [512*512, 6], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fc3b = tf.Variable(tf.constant(1.0, shape=[6], dtype=tf.float32), name='biases', trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]
            self.parameters += [fc3w, fc3b]

    def load_weights(self, sess):
        weights = np.load(self.weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            removed_layer_variables = ['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b']
            if not k in removed_layer_variables:
                print(k)
                print("",i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))

if __name__ == '__main__':

    #with tf.device('/cpu:0'):
    train_data = h5py.File('/home/goerlab/Bilinear-CNN-TensorFlow/cross_validation_add_gan/cross1/small/new_train_448.h5', 'r')
    val_data = h5py.File('/home/goerlab/Bilinear-CNN-TensorFlow/cross_validation_add_gan/cross1/small/new_val_448.h5', 'r')
    

    print('Input data read complete')

    X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']

    print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    X_val, Y_val = shuffle(X_val, Y_val)
    #print Y_train[0]
    print("Device placement on. Creating Session")
    
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess = tf.Session()
    #sess = tf.InteractiveSession()
    #with tf.device('/gpu:0'):
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 6])
    #
    # print("target:")
    # print(target)
    #print 'Creating graph'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    
    
    #with tf.device("/gpu:0"):
    print('VGG network created')
    
    
    # Defining other ops using Tensorflow
    train_total=len(X_train)
    train_nums=[1890*2,760*2,220*2*2,350*2,1100*2,370*2,120*2*10,340*2,570*2,1000*2,40*20,1000*2,2380*2]
    #coeffs=[0.5/(1.0*i/train_total) for i in train_nums]
    #coeffs=[0.5*train_total/(1.0*i) for i in train_nums]
    coeffs=[1.0,1.0,1.0,1.0,1.0,1.0]#,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    print(coeffs)
    eps = tf.constant(value=1e-10)

    softmax=tf.nn.softmax(vgg.fc3l)
    print("softmax:")
    print(softmax)
    #loss1=tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.multiply(target,tf.log(softmax)),coeffs),1))

    loss1=tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.multiply(target,tf.pow(1-softmax,2)*tf.log(softmax)),coeffs),1))
    l_pt_gamma=tf.pow(1-softmax,2)
    log_pt=tf.log(softmax)
    middle1=tf.pow(1-softmax,2)*tf.log(softmax)
    middle2=tf.multiply(target,tf.pow(1-softmax,2)*tf.log(softmax))
    loss2=tf.reduce_sum(-tf.multiply(tf.multiply(target,tf.pow(1-softmax,2)*tf.log(softmax)),coeffs),1)
    #loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
    #loss2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))


    learning_rate_wft = tf.placeholder(tf.float32, shape=[])
    learning_rate_woft = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(loss1)


    correct_prediction = tf.equal(tf.argmax(vgg.fc3l,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    vgg.load_weights(sess)

    batch_size = 1  ######queenie edited:origin=32


    print('Starting training')

    lr = 1.0
    base_lr = 1.0
    break_training_epoch = 2
    # CountSketch_h=tf.Graph.get_collection('CountSketch_Reshape_0/h:0')
    # CountSketch_s=tf.Graph.get_collection('CountSketch_Reshape_0/s:0')
    #parameters_list=tf.Graph.get_collection()
    #print(parameters_list)
    global_parameters_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print(global_parameters_list)
    CountSketch_h=global_parameters_list[27]
    CountSketch_s=global_parameters_list[28]
    local_parameters_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    print(local_parameters_list)
    model_parameters_list=tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    print(model_parameters_list)
    trainable_parameters_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(trainable_parameters_list)
    #for epoch in range(100):
    for epoch in range(91):   ###queenie edited, origin is as above#############

        if epoch==break_training_epoch:
            last_layer_weights = []
            for v in vgg.parameters:
                print(v)
                if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    print('Printing Trainable Variables :', sess.run(v).shape)
                    last_layer_weights.append(sess.run(v))
            last_layer_weights.append(sess.run(CountSketch_h))
            last_layer_weights.append(sess.run(CountSketch_s))
	    
            np.savez('last_layers_epoch_20_VIDI_debug_epoch_02.npz',last_layer_weights)
            print("Last layer weights saved")
            break

        avg_cost = 0.
        #total_batch = int(6000/batch_size)
        total_batch=int(len(X_train)/batch_size) ####queenie edited, origin is as above###
        X_train, Y_train = shuffle(X_train, Y_train)
        

        
        # Uncomment following section if you want to break training at a particular epoch





        for i in range(total_batch-1):
            print("i:%d" %(i))
            batch_xs, batch_ys = X_train[i*batch_size:i*batch_size+batch_size], Y_train[i*batch_size:i*batch_size+batch_size]
            batch_xs = random_flip_right_to_left(batch_xs)

            #if epoch <= finetune_step:
            start = time.time()
            _,count_h,count_s=sess.run([optimizer,CountSketch_h,CountSketch_s], feed_dict={imgs: batch_xs, target: batch_ys})
            #print(count_h)
            #print(count_s)

            if i%20==0:
                print('Last layer training, time to run optimizer for batch size:', batch_size,'is --> ',time.time()-start,'seconds')


                target_out,softmax_out,l_pt_gamma_out,log_pt_out,cost1,cost2,count_h,count_s = sess.run([target,softmax,l_pt_gamma,log_pt,loss1,loss2,CountSketch_h,CountSketch_s], feed_dict={imgs: batch_xs, target: batch_ys})
                #print(count_h)
                #print(count_s)
                #if i % 20 == 0:
                #print ('Learning rate: ', (str(lr)))
                #if epoch <= finetune_step:
                print("Target:")
                print(target_out)
                print("Softmax")
                print(softmax_out)
                print("1-pt_gamma:")
                print(l_pt_gamma_out)
                print("log_pt:")
                print(log_pt_out)
                print("cost2:")
                print(cost2)
                print ("cost1:")
                print(cost1)
                print("Training last layer of BCNN_DD")

                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,"Loss1:", str(cost1))
                print("Training Accuracy -->", sess.run(accuracy,feed_dict={imgs: batch_xs, target: batch_ys}))
        print("Train end")
        val_batch_size = 10
        total_val_count = len(X_val)
        correct_val_count = 0
        val_loss = 0.0
        total_val_batch = int(total_val_count/val_batch_size)

        print("Validation Start:")
        for i in range(total_val_batch):
            batch_val_x, batch_val_y = X_val[i*val_batch_size:i*val_batch_size+val_batch_size], Y_val[i*val_batch_size:i*val_batch_size+val_batch_size]
            val_loss += sess.run(loss1, feed_dict={imgs: batch_val_x, target: batch_val_y})


            pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_val_x, target: batch_val_y})
            correct_val_count+=pred


        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, total_val_count)
        print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*total_val_count))
        print("##############################")

        

    # test_data = h5py.File('/home/goerlab/Bilinear-CNN-TensorFlow/train_test_small/new_test_224.h5', 'r')
    # X_test, Y_test = test_data['X'], test_data['Y']
    # total_test_count = len(X_test)
    # correct_test_count = 0
    # test_batch_size = 10
    # total_test_batch = int(total_test_count/test_batch_size)
    # for i in range(total_test_batch):
    #     batch_test_x, batch_test_y = X_test[i*test_batch_size:i*test_batch_size+test_batch_size], Y_test[i*test_batch_size:i*test_batch_size+test_batch_size]
    #
    #     pred = sess.run(num_correct_preds, feed_dict = {imgs: batch_test_x, target: batch_test_y})
    #     correct_test_count+=pred
    #
    # print("##############################")
    # print("correct_test_count, total_test_count", correct_test_count, total_test_count)
    # print("Test Data Accuracy -->", 100.0*correct_test_count/(1.0*total_test_count))
    # print("##############################")



