#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import time

def get_train_test(data_array):
    num_data_points = len(data_array)
    train_size = 9*num_data_points // 10
    test_size = num_data_points // 10

    # splitting train and test data in a 9:1 ratio
    train = data_array[:train_size]
    test = data_array[num_data_points-test_size:]
    train_features = train[:,:40]
    train_target = train[:,40:]
    test_features = test[:,:40]
    test_target = test[:,40:]
    return train_features, train_target, test_features, test_target

def multilayer_perceptron(x, weights, biases):
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

def get_epoch(x, y, n):
    input_size = x.shape[0]
    number_batches = input_size // n
    extra_examples = input_size % n
    batches = {}
    batch_indices = np.arange(input_size)
    np.random.shuffle(batch_indices)
    for i in range(number_batches):
        temp_indices = batch_indices[n*i:n*(i+1)]
        temp_x = []
        temp_y = []
        for j in temp_indices:
            temp_x.append(x[j])
            temp_y.append(y[j])
        batches[i] = [np.asarray(temp_x), np.asarray(temp_y)]
    if extra_examples != 0:
        extra_indices = batch_indices[input_size-extra_examples:input_size]
        temp_x = []
        temp_y = []
        for k in extra_indices:
            temp_x.append(x[k])
            temp_y.append(y[k])
        batches[i+1] = [np.asarray(temp_x), np.asarray(temp_y)]
    return batches

def main():
    # reading data into numpy array
    file_path = 'https://raw.githubusercontent.com/michaelneuder/smart_scroller/master/data/synthetic_data.csv'
    data = pd.read_csv(file_path, header = None)
    data_array = data.values
    input_size = 40
    output_size = 1

    # getting test - train data in a 9:1 ratio
    train_features, train_target, test_features, test_target = get_train_test(data_array)

    # reducing data size for testing
    num_points = 40
    train_features, train_target, test_features, test_target = train_features[:40], train_target[:40], test_features[:40], test_target[:40]

    # start of the tensorflow session
    sess = tf.Session()
    init = tf.global_variables_initializer()

    # declaring our placeholders
    features = tf.placeholder(tf.float32, shape=[None, input_size])
    target = tf.placeholder(tf.float32, shape=[None, output_size])

    # variables to be learned single layer fully connected network
    weights = {
        'out': tf.Variable((1/(input_size))*tf.random_normal([input_size, output_size]))
    }
    biases = {
        'out': tf.Variable((1/(input_size))*tf.random_normal([output_size]))
    }

    # activation
    learning_rate = .01
    epochs = 100000
    batch_size = 40

    # model
    prediction = multilayer_perceptron(features, weights, biases)

    # loss and optimization
    cost = tf.reduce_mean(tf.square(tf.subtract(prediction, target)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        global_step = 0
        print("starting training ... ")
        for epoch_count in range(epochs):
            epoch = get_epoch(train_features, train_target, batch_size)
            print('beginning epoch {} ...'.format(epoch_count))
            for batch in range(len(epoch)):
                train_features_batch, train_target_batch = np.asarray(epoch[batch][0]), np.asarray(epoch[batch][1])
                sess.run(optimizer, feed_dict={features : train_features_batch, target : train_target_batch})
                loss = sess.run(cost, feed_dict={features : train_features_batch, target : train_target_batch})
                print('  -  traning global step {}, error : {:.4f}'.format(global_step, loss))
                global_step += 1
        print('optimization finished!')
        loss = sess.run(cost, feed_dict={features : test_features, target : test_target})
        print('total error = {:.4f}'.format(loss))

if __name__ == '__main__':
    main()
