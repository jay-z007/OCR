# OCR
A simple OCR to recognize Upper and Lower case English alphabets

## Dataset
It consists of upper and lower case alphabets. Each having almost 200 image samples. And Each sample is written in a different font.

## Method

### Preprocessing 
- **Binarization -** The images are binarized, i.e., the pixel intensities are converted to either 0 or 1, 
  depending on whether they are greater than or less than a particular threshold [here it is 15].
- **Zoning -** The binarized images are then processed for feature extraction. It is done by zoning technique.
  Where in the images are divided into small blocks [like a grid] and then the number of one's in each block is calculated.
  
  For example, 
  In this case, the images are divided into 16 blocks [4x4 grid].
  The list of number of one's from these blocks forms the feature vector for the neural network.
  
### Training the Neural network
- Network configuration [1 hidden layer]
  - Input nodes - 16
  - Hidden nodes - 30
  - Output nodes - 26
  
**Note -** The number of nodes in input layer should be equal to the size of a feature vector.

- Training
  The data is divided into Training [90%] and Testing sets [10%].
  The network is trained on the training set and its accuracy is measure on the Testing sets.
  
### Results
- The results are logged in the zoning_log.txt file
Highest Accuracy achieved by tweaking the parameters is 82.36%
