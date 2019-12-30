This project is an implementation of deep learning methods for classifing two type of snake,the Viper Snake (Daboia Palestine)
and Night Snake. Originally, we intended to classify between Viper snake and Hemorrhois Nummifer snake 
(that are commonly found in Israel).
When we tried to create the database, we couldn’t find enough images to create a sufficient repository.
So we chose the Night Snake instead of the Viper snake (they look very much alike) which was a good challenge to our model.

The code is based on the great "hermesribeiro/DeepSnakes" github code (https://github.com/hermesribeiro/DeepSnakes).
We made a few changes to adjust it to our goals, as well as made new datase files.

	Classes Description:
  PreprocessFiles:
This class creates the dataset. In this class, the images folders of the two types of snakes are recorded. The images resized to 128X128 pixels and divided to train and test files (80% for train and 20% for test). Each hdf5 file in both the train and the test contains 2 arrays: the first array contains the images, and the second array contains the labels - 0 for Night snake and 1 for Viper.

ProcessDataset:
This class has two functions: 
•	Get the hdf5 files and return the image and label arrays.  
•	Edit and reshape the image and label arrays to suit the Neural Network use.

Statistics
This class contains few functions to generate statistic and information on the model:
•	Generate percentages of accuracy for the train and test.
•	Print an error graph that shows the improvement of the train and the test. 
•	Print examples and show, for each example, whether the system was right or wrong.

Logistic_Regression
This class contains Logistic Regression model implementation.
The class uses the ProcessDataset class to input the hdf5 files, performs a train (with logistic regression) and test. For test we chose a random image that the algorithm classifies as one of the two snakes. If the algorithm fails to classify the image, a relevant message is printed to the console.
This class uses the Statistics class to print an improvement rate graph of the test and train.




Neural_Network
This class contains the Neural Network model implementation.
The class uses the ProcessDataset class to input the hdf5 files and builds a model that contains one hidden layer. The model function has a parameter called "num_neurons" that the user can change in order to determine the number of neurons in the hidden layer.
In addition, we use batch gradient descent (MB-GD) function that splits the data into mini - batches and then the data can run in sections.
We also used the Ridge regression in few of the program runs.


