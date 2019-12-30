import h5py as h5
import numpy as np



def importFiles():
# Extracting the train dataset from the h5 file.
    f = h5.File("/Volumes/Home/train_set.hdf5", 'r')
    print("Stored data:", list(f.keys()))
# Storing the original data "permanently".
    images_train_orig = f["images_train"][()]
    labels_train_orig = f["labels_train"][()]
    f.close()
# Extracting the dev dataset from the h5 file.
    f = h5.File("/Volumes/Home/dev_set.hdf5", 'r')
    print("Stored data:", list(f.keys()))
# Storing the original data "permanently".
    images_dev_orig = f["images_dev"][()]
    labels_dev_orig = f["labels_dev"][()]
    f.close()
    return images_train_orig , labels_train_orig , images_dev_orig , labels_dev_orig



def proccessData(images_train_orig , labels_train_orig ,
                 images_dev_orig , labels_dev_orig):
# Copying the original data for manipulation.
    images_train = images_train_orig
    labels_train = labels_train_orig.reshape([1, len(labels_train_orig)])
    images_dev = images_dev_orig
    labels_dev = labels_dev_orig.reshape([1, len(labels_dev_orig)])
# Scaling the data
    images_train = images_train / 255
    images_train = images_train.reshape((len(images_train), -1))
    images_train = images_train.transpose()
    images_dev = images_dev / 255
    images_dev = images_dev.reshape((len(images_dev), -1))
    images_dev = images_dev.transpose()
# Shapes
    imsize = images_train.shape[0]
    return images_train , labels_train , images_dev , labels_dev , imsize


def printDatasetInfo(images_train_orig , labels_train_orig ,
                       images_dev_orig , labels_dev_orig):
    print("Train set")
    print("Images shape:", images_train_orig.shape)
    print("Images dtype:", images_train_orig.dtype)
    print("Min, max and mean pixel values:", images_train_orig.min(),"-",
      images_train_orig.max(),"-","{:5.1f}".format(images_train_orig.mean()))
    print("Labels shape:", labels_train_orig.shape)
    print("Labels dtype:", labels_train_orig.dtype)

    print("Dev set")
    print("Images shape:", images_dev_orig.shape)
    print("Images dtype:", images_dev_orig.dtype)
    print("Min, max and mean pixel values:", images_dev_orig.min(),"-",
      images_dev_orig.max(),"-","{:5.1f}".format(images_dev_orig.mean()))
    print("Labels shape:", labels_dev_orig.shape)
    print("Labels dtype:", labels_dev_orig.dtype)


def printDatasetInfoAfterProcessing(images_train , labels_train , images_dev , labels_dev):
    print("Shapes")
    print("Images train:", images_train.shape)
    print("Labels train:", labels_train.shape)
    print("Images dev:", images_dev.shape)
    print("Labels dev:", labels_dev.shape)


# for NN

def reg_reshape_snakes(images_in,labels_in):
    # Regularizing images
    images = images_in/images_in.max()
    # Reshaping/Transposing images
    images = images.reshape((len(images),-1))
    images = images.transpose()
    # Reshaping labels
    labels = labels_in.reshape([1,len(labels_in)])
    return images, labels


def split_data(indices,batch_size):
    divisible_batches = len(indices)//batch_size
    divisible_sequence = divisible_batches*batch_size
    splits = np.split(indices[:divisible_sequence],divisible_batches)
    if divisible_sequence != len(indices):
        splits.append(indices[divisible_sequence:])
    return splits