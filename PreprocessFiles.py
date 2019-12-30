
# This program is for resize all images in a folder and generate hdf5 test and train files.


from PIL import Image
import numpy as np
import os
import h5py as h5
import tensorflow as tf


def img_resize(path, size=100, save=False):
    # Reads all image files in path and resizes
    # Works if folder contains only images
    # Parameters: path: string
    #               path to read folder
    #             size: integer or (width, height) tuple
    #               pixels in each axis. Integer will create a square image
    #             save: boolean
    #               Whether to save the resized images in an "path"_small
    #               folder. If true, the new folder must be created beforehand.
    # Returns: ndarray containing the images

    directory = os.listdir(path)
    if type(size) is int:
        size = (size, size)
        # must use uint8 dtype for imshow to work correctly
    images = np.empty([len(directory), size[0], size[1], 3], dtype='uint8')

    for idx, imagename in list(enumerate(directory)):
        if imagename == '.DS_Store':   # for mac
            continue
        image = Image.open(path + "/" + imagename)
        image = image.resize((100, 100)).convert('RGB')
        if save: image.save(path + "_small/" + imagename[:-4] + ".png")
        images[idx] = np.array(image)
    return images

def shuffle_and_split(images, labels, test_ratio=0.2):
    # Shuffle all images in an [n,width,height,3] ndarray and splits it on
    # the first axis where n is the number of images in the dataset
    # Parameters: images: [n,width,height,3]
    #               ndarray containing the dataset
    #             labels: 1D numpy array containing the images labels
    #             test_ratio: float
    #               relative size of the test dataset to the whole dataset
    # Returns: images_train, images_test, labels_train, labels_test containing
    #           the images and its labels split into training and dev sets
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    images_train = images[int(test_ratio * len(images)):]
    labels_train = labels[int(test_ratio * len(images)):]
    images_dev = images[:int(test_ratio * len(images))]
    labels_dev = labels[:int(test_ratio * len(images))]
    return images_train, images_dev, labels_train, labels_dev


# This program creates a h5 files with the train-test sets from raw images
# downloaded from the web
np.random.seed(837)

    # Path to the image folders
path_commonViper = "/Volumes/Home/new/common viper"
path_nightSnake = "/Volumes/Home/new/night snake"

    # Resizing and accumulating images inside an ndarray
images_commonViper = img_resize(path_commonViper, save=False)
images_nightSnake = img_resize(path_nightSnake, save=False)


    # Creating image labels 1 for Common Viper 0 for Night Snake
    # uint8 dtype just for the sake of consistency

labels_commonViper = np.ones(len(images_commonViper), dtype='uint8')
labels_nightSnake = np.zeros(len(images_nightSnake), dtype='uint8')


    # Merging the dataset
images = np.concatenate((images_commonViper, images_nightSnake))
labels = np.concatenate((labels_commonViper, labels_nightSnake))

    # Generating the train and test split
images_train, images_dev, labels_train, labels_dev = shuffle_and_split(images, labels)



    # Creating train set file
f = h5.File("/Volumes/Home/train_set.hdf5", "w")
f.create_dataset("images_train", data=images_train)
f.create_dataset("labels_train", data=labels_train)
f.close()

    # Creating test set file
f = h5.File("/Volumes/Home/dev_set.hdf5", "w")
f.create_dataset("images_dev", data=images_dev)
f.create_dataset("labels_dev", data=labels_dev)
f.close()
