import time
from ProcessDataset import *
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt
start = time.time()


images_train_orig, labels_train_orig, images_dev_orig, labels_dev_orig = importFiles()

images_train = images_train_orig/255
labels_train = (labels_train_orig).reshape(-1,1)
images_dev = images_dev_orig/255
labels_dev = (labels_dev_orig).reshape(-1,1)
imsize = images_train.shape[1:3] # image size
dsize = images_train.shape[0]    # num of train images
dsize_dev = images_dev.shape[0]  # num of dev images
print("Shapes")
print("Images train:", images_train.shape)
print("Labels train:", labels_train.shape)
print("Images dev:", images_dev.shape)
print("Labels dev:", labels_dev.shape)
print("imsize:", imsize)
print("dsize:", dsize)
print("dsize_dev:", dsize_dev)


##############################################
#            Building the model              #
##############################################
print("** Building the model **")

def model_builder(cparams, hparams):
    # Builds the models placeholders for inputs and outputs, parameter
    # variables and feed forward computation graph.
    # Argument: cparams - constant params, meaning, input/output dimensions.
    #           hparams - hyperparameters.
    # Returns: tf objects regarding X, Y, logits and costs.


    # getting win, hin, nout (inputs width and height, output # of classes)
    win = cparams["win"]
    hin = cparams["hin"]
    nout = cparams["nout"]

    # getting f1,c1,f2,c2. Stride on filters will be 1.
    f1 = hparams["f1"]  # filter size of conv 1 (5x5 in our example)
    c1 = hparams["c1"]  # num of kernels for conv 1 (48 kernels in our example)
    f2 = hparams["f2"]  # filter size of conv 2 (3x3 in our example)
    c2 = hparams["c2"]  # num of kernels for conv 2 (128 kernels in our example)

    # getting p1, p2 (pooling layer sizes, will consider size = stride)
    p1 = hparams["p1"] # 2x2 in our example
    p2 = hparams["p2"] # 2x2 in our example


    # creating placeholders for input/output
    X = tf.placeholder(dtype=tf.float32, shape=[None, win, hin, 3], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, nout], name="Y")

##  FIRST LAYER  ##

    W1 = tf.get_variable('conv_1/W', shape=(f1, f1, 3, c1), initializer=tf.glorot_normal_initializer(9)) # shape = (5, 5, 3, 48)
    b1 = tf.get_variable('conv_1/b', shape=(c1), initializer=tf.zeros_initializer())
    conv_1 = tf.nn.bias_add(tf.nn.conv2d(X, W1, (1, 1, 1, 1), 'SAME'), b1) # we use padding SAME so the size so the output remains the same size

    act_1 = tf.nn.relu(conv_1) # we used relu for the activation
    pool_1 = tf.nn.max_pool(act_1, (1, p1, p1, 1), (1, p1, p1, 1), 'VALID') # MaxPooling of 2x2



##  SECOND LAYER  ##

    W2 = tf.get_variable('conv_2/W', shape=(f2, f2, c1, c2), initializer=tf.glorot_normal_initializer(8)) # shape = (3,3,48,128)
    b2 = tf.get_variable('conv_2/b', shape=(c2), initializer=tf.zeros_initializer())
    conv_2 = tf.nn.bias_add(tf.nn.conv2d(pool_1, W2, (1, 1, 1, 1), 'SAME'), b2)
    act_2 = tf.nn.relu(conv_2)
    pool_2 = tf.nn.max_pool(act_2, (1, p2, p2, 1), (1, p2, p2, 1), 'VALID') # MaxPooling of 2x2

##############################

    keep_prob = 0.5
    h_fc1_drop = tf.nn.dropout(pool_2, keep_prob)

###########################
    # flatten the 3D tensors
    flat = tf.layers.flatten(h_fc1_drop)

    # flat = tf.layers.flatten(pool_2)


    # logits
    wpool = int(win / p1 / p2) # 128 /2 /2 = 32 (we divide twice becuse of the two poolings)
    hpool = wpool
    cpool = c2   # 128


    W3 = tf.get_variable('dense_1/W', shape=(1, wpool * hpool * cpool), initializer=tf.glorot_normal_initializer(7))
    b3 = tf.get_variable('dense_1/b', shape=(1, 1), initializer=tf.zeros_initializer())
    logits = tf.transpose(tf.matmul(W3, tf.transpose(flat)) + b3)


    # LOSS (cost)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
    reg = (tf.nn.l2_loss(W1) +
           tf.nn.l2_loss(W2) +
           tf.nn.l2_loss(W3)) / tf.cast(tf.shape(X)[0], tf.float32) # Ridge regularization
    print("LOSS ",cost)
    print("reg ",reg)

    return X, Y, logits, cost, reg



##############################################
#            Training the model              #
##############################################
print("Start training process")

def model_trainer(hparams,
                  images_train=images_train,  # Problematic if used incorrectly.
                  labels_train=labels_train,  # watch out!
                  images_dev=images_dev,
                  labels_dev=labels_dev,
                  print_every=5):
    # Trains the model and saves the best iteration.
    # Argument: hparams - hyperparameters.
    #           images_train - train set state tensor of inputs
    #           labels_train - train set state vector of outputs
    #           images_dev - dev set state tensor of inputs
    #           labels_dev - dev set state vector of outputs
    # Returns: train and dev set costs (loss)


    # number of train and dev samples
    n_train = images_train.shape[0]
    n_dev = images_dev.shape[0]
    print("n_train : ",n_train)  # 1817
    print("n_dev : ",n_dev)      # 454


    # Retrieving hyperparameters
    epochs = hparams["epochs"]
    learning_rate = hparams["learning_rate"]
    batch_size = hparams["batch_size"]
    num_batches = n_train//batch_size       # 1817 // 4   (// operator =  floor division)
    lambd = hparams["lambd"]


    # determining cparams
    cparams = {"hin": images_train.shape[1],
               "win": images_train.shape[2],
               "nout": labels_train.shape[1]}

    print("hin", images_train.shape[1])
    print("win", images_train.shape[2])
    print("nout", labels_train.shape[1])

    # building the model
    X, Y, logits, cost, reg = model_builder(cparams, hparams)


    # defining the optimizer
    cpr = cost + lambd * reg
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cpr)

    # defining the prediction
    pred = tf.sigmoid(logits)

    with tf.Session() as sess:
        # Performance variables container
        costs_train = []
        costs_dev = []
        preds_train = []
        preds_dev = []
        accs_train = []
        accs_dev = []
        cost_dev_best = float("inf")

        # enumerating the samples
        indices = np.arange(n_train)

        # Starting the global variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # iterating through epochs
        for epoch in range(epochs + 1):  # epochs+1 to facilitate print_every
            # spilts the data to mini batches and shuffle :
            indices = np.random.permutation(indices)
            splits = np.array_split(indices, num_batches)
            cost_train = 0.

            # running mini batches
            pred_train = np.zeros([n_train, 1])   # return array of zeros size 1817
            for split in splits:
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: images_train[split], Y: labels_train[split]})
                cost_train += temp_cost / num_batches
                pred_train[split] = sess.run(pred, feed_dict={X: images_train[split], Y: labels_train[split]})

            # storing train cost
            costs_train.append(cost_train)
            preds_train.append(pred_train)

            # computing dev cost
            cost_dev = 0.
            pred_dev = np.zeros([n_dev, 1])  # array of zeros zise 454

            # will divide into samples to minimize memory usage
            samples = np.array_split(np.arange(n_dev), n_dev / 16)

            for sample in samples:
                cost_dev += sess.run(cost, feed_dict={X: images_dev[sample], Y: labels_dev[sample]}) / (n_dev / 16)
                pred_dev[sample] = sess.run(pred, feed_dict={X: images_dev[sample], Y: labels_dev[sample]})
            costs_dev.append(cost_dev)
            preds_dev.append(pred_dev)

            # using numpy to calculate the accuracies
            acc_train = np.mean(np.equal(np.greater(pred_train, 0.5).astype(int), labels_train))
            acc_dev = np.mean(np.equal(np.greater(pred_dev, 0.5).astype(int), labels_dev))
            accs_train.append(acc_train)
            accs_dev.append(acc_dev)

            # saving the best parameters for later reuse
            if cost_dev < cost_dev_best:
                # saver.save(sess,"./bestfitcnn.dat")
                cost_dev_best = cost_dev

            # print output
            if not epoch % print_every:
                print("Epoch {}".format(epoch))
                print(" - loss_train: {:6.4f} - accuracy_train: {:6.4f} - loss_dev: {:6.4f} - accuracy_dev: {:6.4f}".format(cost_train,
                                                                                                      acc_train,
                                                                                                      cost_dev,
                                                                                                      acc_dev))

        return costs_train, costs_dev, accs_train, accs_dev, preds_train, preds_dev

#%%time
# tf.reset_default_graph()
# f1 = filter 1 size, c1 = number of channels (filters), p1 = size of pooling

# 
hparams = {"f1":5, "c1":48, "p1":2,
           "f2":3, "c2":128, "p2":2,
           "epochs":400,
           "learning_rate":2e-6,
           "batch_size":4,
           "lambd":0.2}

print("Hyperparameters:", hparams)
costs_train, costs_dev, accs_train, accs_dev, preds_train, preds_dev = model_trainer(hparams, print_every=1)

#  Analizing the results and prints the best epoch
best_epoch = np.argmin(costs_dev)
cost_train = costs_train[best_epoch]
cost_dev = costs_dev[best_epoch]
acc_train = accs_train[best_epoch]
acc_dev = accs_dev[best_epoch]
print("The best epoch is {}".format(best_epoch))
print(" - loss_train: {:6.4f} - accuracy_train: {:6.4f} - loss_dev: {:6.4f} - dev: {:6.4f}".format(cost_train,acc_train,cost_dev,acc_dev))

# Graph
train_curve, = plt.plot(costs_train, label = 'Train error')
test_curve,  = plt.plot(costs_dev, label = 'Dev error')
plt.legend(handles=[train_curve,test_curve])
plt.xlabel('Epochs')
plt.ylabel('Mean log loss')
plt.show()


end = time.time()
print("Total Runtime:" , (end-start)/60 , " minutes")