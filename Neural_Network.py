import tensorflow as tf
from myTensorFlow.Assignment2.Statistics import *
from myTensorFlow.Assignment2.ProcessDataset import *
import time
start = time.time()

# Extracting the train and test dataset from the hdf5 files and process them
images_train_orig, labels_train_orig, images_dev_orig, labels_dev_orig = importFiles()
images_train, labels_train = reg_reshape_snakes(images_train_orig,labels_train_orig)
images_dev, labels_dev = reg_reshape_snakes(images_dev_orig,labels_dev_orig)
imsize = images_train.shape[0]   # amount of pixels
dsize = images_train.shape[1]    # amount of images
dsize_dev = images_dev.shape[1]  # amount of dev images

# Printing dataset information
printDatasetInfoAfterProcessing(images_train_orig, labels_train_orig, images_dev_orig, labels_dev_orig)


##############################################
#            Building the model              #
##############################################
print("** Building the model **")

def model_builder(num_neurons):
# Builds the models inputs, parameters and feed forward graph.
# Argument: num_neurons - number of neurons
# Returns: tf objects regarding X, Y, costs and a dictionary containing the weights and biases
    tf.reset_default_graph()
    tf.set_random_seed(2468)
# Adding placeholders for I/O
    X = tf.placeholder(tf.float32,shape=[imsize,None],name="X")
    Y = tf.placeholder(tf.float32,shape=[1,None], name="Y")
# Gettind the parameter matrixes
    W1 = tf.get_variable("W1",[num_neurons,imsize],tf.float32,tf.contrib.layers.xavier_initializer(seed=7324))
    b1 = tf.get_variable("b1",[num_neurons,1],tf.float32,tf.zeros_initializer())

    W2 = tf.get_variable("W2",[1,num_neurons],tf.float32,tf.contrib.layers.xavier_initializer(seed=5236))
    b2 = tf.get_variable("b2",[1,1],tf.float32,tf.random_normal_initializer())

# Hidden layer feed forward
    A1 = tf.add(tf.matmul(W1,X),b1)
    Z1 = tf.nn.relu(A1)
# Output layer feed forward
    A2 = tf.add(tf.matmul(W2,Z1),b2)
# Cost function
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=A2))
# Join weights and biases in a single dictionary
    params = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return X, Y, cost, params

# Split the data into mini-batches
    split_data(indices,batch_size)


##############################################
#            Training the model              #
##############################################
print("** Start training **")

def model_trainer(images_train = images_train, labels_train = labels_train,
                  num_neurons=20, batch_size=4, epochs=10, alpha=1.e-5, lambd=5., print_every=10):
#The default learning rate (alpha) is small because we are using sum log loss instead of mean log loss
# We are using sum log loss to correctly scale costs when using batch GD.
    np.random.seed(4681)
# Call model builder and get the parameters initial guess
    X, Y, cost, params = model_builder(num_neurons)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

# *** BEFORE REIDGE REGRESSION ***
# cpr = cost plus regularization
# Will optimize cost plus regularization but will only plot the cost part
    # cpr = cost + lambd*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2))
# Defining the optimizer
    # optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cpr)
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# Initializing global variables
    init = tf.global_variables_initializer()
# Running session
    with tf.Session() as sess:
        sess.run(init)
        costs_train = []
        costs_dev = []
        cost_dev_best = float("inf")
# Indices to be shuffled
        indices = np.arange(dsize)
# splitting the indices in batch_size chunks plus a last different chunk if necessary
        splits = split_data(indices,batch_size)
# list containing the number of elements in each split
        nelements = [float(len(x)) for x in splits]
        for epoch in range(epochs+1):
# shuffling the dataset indices
            np.random.shuffle(indices)
# splitting the dataset
            splits = split_data(indices,batch_size)
            epoch_cost = []
# running the training across batches
            for split in splits:
                _,batch_cost = sess.run([optimizer,cost],feed_dict={X:images_train[:,split],Y:labels_train[:,split]})
                epoch_cost.append(batch_cost/len(split))
# computing metrics
            cost_train = np.dot(epoch_cost,nelements)/dsize
            costs_train.append(cost_train)
            cost_dev = sess.run(cost, feed_dict={X:images_dev,Y:labels_dev})
            cost_dev = cost_dev/dsize_dev
            costs_dev.append(cost_dev)
# Save parameters for best dev error (early stopping)
            if cost_dev < cost_dev_best:
                W1v,b1v,W2v,b2v = sess.run([W1,b1,W2,b2]) # "v" stands for value
                params_train = {"W1":W1v,"b1":b1v,"W2":W2v,"b2":b2v,"epoch":epoch,"cost_dev":cost_dev}
                cost_dev_best = cost_dev
# print on screen
            if epoch%print_every == 0:
                print("Epoch {}: train cost = {}, dev cost = {}".format(epoch,cost_train,cost_dev))
    return costs_train, costs_dev, params_train


##############################################
#                   MAIN                     #
##############################################


costs_train, costs_dev, params_train = model_trainer()

W1 = params_train["W1"]
b1 = params_train["b1"]
W2 = params_train["W2"]
b2 = params_train["b2"]


showErrorGraphNN(params_train , costs_train , costs_dev)

y_train_pred = predictNN(images_train,W1,b1,W2,b2)
acc = accuracy(labels_train,y_train_pred)
print("Train set accuracy:", acc)
y_dev_pred = predictNN(images_dev,W1,b1,W2,b2)
acc = accuracy(labels_dev,y_dev_pred)
print("Dev set accuracy:",acc)


def species(i):
    if i == 1:
        name = "Common Viper"
    elif i==0:
        name = "Night Snake"
    else:
        name = "Don't know"
    return name


wrong_images = images_dev_orig[(y_dev_pred != labels_dev).ravel()]
wrong_labels = labels_dev_orig[(y_dev_pred != labels_dev).ravel()]
right_images = images_dev_orig[(y_dev_pred == labels_dev).ravel()]
right_labels = labels_dev_orig[(y_dev_pred == labels_dev).ravel()]
print("Number of hits:",len(right_images))
print("Number of misses:",len(wrong_images))



# i = 16
# print("Wrongly predicted image")
# print("Should be a", species(wrong_labels[i]))
# plt.imshow(wrong_images[i])
# plt.show()
#
# i = 25
# print("Correctly predicted images")
# print("Yep, it's a", species(right_labels[i]))
# plt.imshow(right_images[i])
# plt.show()




end = time.time()
print("Total time:" , (end-start)/60)