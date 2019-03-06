import os
import numpy as np
import tensorflow as tf
import cv2
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.reshape(224,224,1)
    img = np.concatenate((img, img, img), -1)
    img = img / 255.0

    resized_img = img.reshape(1,224,224,3)
    return resized_img
def load_data():
    img=[]
    label=[]
    dir = './'
    with open("train.txt") as f:
        line = f.readline()
        while line:
            Item = line.split()
            try:
                resized_img = load_img(os.path.join(dir, Item[0]))
            except OSError:
                continue
            img.append(resized_img)
            label.append(Item[1])
            #imgs.append(resized_img)    # [1, height, width, depth] * n
            if len(img) == 200:        # only use 400 imgs to reduce my memory load
                break
            #print(line.strip())
            print(len(img))
            line = f.readline()
    return img, label
class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss
    
    def predict(self, paths):
        for i, path in enumerate(paths):
            x = load_img(path)
            x = np.array(x)
            x = x.reshape([1,224,224,3])
            length = self.sess.run(self.out, {self.tfx: x})
            if (length < 0):
                length = -length
        return length
    def save(self, path='./model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)
def train():
    imgs, labels = load_data()

    xs = np.array(imgs)
    ys = np.array(labels)
    ys = ys.reshape([-1,1])
    print(xs.shape, ys.shape)
    
    vgg = Vgg16(vgg16_npy_path='./vgg16.npy')
    print('Net built')
    for i in range(300):
        b_idx = np.random.randint(0, len(xs), 60)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)

    vgg.save('./model/transfer_learn')      # save learned fc layers
    
vgg = Vgg16(vgg16_npy_path='./vgg16.npy',
                restore_from='./model/transfer_learn')
def eval(ct_path):
    ct =vgg.predict([ct_path])
    return ct



