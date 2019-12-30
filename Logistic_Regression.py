from myTensorFlow.Assignment2.Statistics import *
from myTensorFlow.Assignment2.ProcessDataset import *
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

start = time.time()

# Extracting the train and test dataset from the hdf5 files and process them

images_train_orig, labels_train_orig, images_dev_orig, labels_dev_orig = importFiles()
images_train, labels_train, images_dev, labels_dev, imsize = proccessData(images_train_orig , labels_train_orig ,images_dev_orig , labels_dev_orig)

# Printing dataset information
printDatasetInfo(images_train_orig, labels_train_orig,images_dev_orig, labels_dev_orig)
printDatasetInfoAfterProcessing(images_train, labels_train, images_dev, labels_dev)


##############################################
#            Building the model              #
##############################################
print("*** Building the model ***")

    # Create placeholder for images, call it X.
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(imsize,None))
Y = tf.placeholder(tf.float32,shape=(1,None))
    # Create weights and biases. Initialize to zero.
W = tf.get_variable("W", [1,imsize], initializer=tf.zeros_initializer())
b = tf.get_variable("b", [1],initializer=tf.zeros_initializer())
logits = tf.add(tf.matmul(W,X),b)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=logits))


##############################################
#            Training the model              #
##############################################
print("*** Start training ***")

epochs = 100
alpha = 0.001
    # Creating the optimizer
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    # Initializing global variables
init = tf.global_variables_initializer()
    # Running session
with tf.Session() as sess:
    sess.run(init)
    costs_train = []
    costs_dev = []
    print("epoch..epoch-cost..test-cost\n")
    for epoch in range(epochs):
        _,epoch_cost = sess.run([optimizer,cost],feed_dict={X:images_train,Y:labels_train})
        costs_train.append(epoch_cost)
        cost_dev = sess.run(cost,feed_dict={X:images_dev,Y:labels_dev})
        if (epoch % 50 == 0) : print(epoch,epoch_cost, cost_dev)
        costs_dev.append(cost_dev)
    Weights, Bias = sess.run([W,b])


##############################################
#            Printing error graph            #
##############################################

showErrorGraph(costs_train , costs_dev)

##############################################
#               Model accuracy               #
##############################################

y_train_pred = predict(images_train,Weights,Bias)
acc = accuracy(labels_train,y_train_pred)
print("Train set accuracy:", acc)
y_dev_pred = predict(images_dev,Weights,Bias)
acc = accuracy(labels_dev,y_dev_pred)
print("Dev set accuracy:",acc)


##############################################
#           Testing the model                #
##############################################

def species(i):
    if i == 1:
        name = "Common Viper"
    elif i==0:
        name = "Night Snake"
    else:
        name = "Don't know"
    return name

for i in range(images_train.shape[1]):
    sample = images_train[:, i]
    target = images_train[0, i]
    prediction = predict(sample, Weights, Bias)
# print("It says it's a {}, it's actually a {}".format(species(prediction),species(target)))
# print(("Good prediction!" if target == prediction else "Bad prediction!"))
# plt.imshow(images_train_orig[i])
# plt.show()

# Compute number of hits and miss's
wrong_images = images_dev_orig[(y_dev_pred != labels_dev).ravel()]
wrong_labels = labels_dev_orig[(y_dev_pred != labels_dev).ravel()]
right_images = images_dev_orig[(y_dev_pred == labels_dev).ravel()]
right_labels = labels_dev_orig[(y_dev_pred == labels_dev).ravel()]
print("Number of hits:",len(right_images))
print("Number of misses:",len(wrong_images))

end = time.time()
print("Total Runtime:" , (end-start)/60 , " minutes")