
import random
from numpy import *
from neural_network import *
from helper_functions import *
from init import *
import text

X = text.data
Y = text.target
X_len = len(X)

####################
#
# Get the feature vector from the training samples
#
####################
feature_vector_list = []
for i in range(X_len):
	feature_vector_list.append(text.zoning(X[i], 4, 4))


####################
#
#	Split the data into training and testing samples
#
####################
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(feature_vector_list, Y, test_size = .1)
len_Y_train = len(Y_train)


####################
#
#	Read the weights and bias and initialize the Neural Network
#
####################
hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias = read_weights("weights.txt")

my_classifier = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes, hidden_layer_weights=hidden_layer_weights, 
	hidden_layer_bias=hidden_layer_bias, output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias)

alpha = my_classifier.LEARNING_RATE


####################
#
#	Convert the output label to a binary array format
#
####################
bin_Y_train = [[0]*26 for i in range(len_Y_train)]
for i in range(len_Y_train):
	index = text.convert_char_to_int(Y_train[i]) - text.convert_char_to_int('a')
	bin_Y_train[i][index] = 1


####################
#
# Training the network
#
####################
epochs = 1
for i in range(epochs):
	print i
	my_classifier.train(X_train, bin_Y_train)


####################
#
#	Predict the output for the testing samples
#
####################
bin_predictions = my_classifier.predict(X_test)
predictions = []

for label in bin_predictions:
	new_label = []
	index = -1

	for i in label:
		new_label.append(int(round(i)))
	for j in range(len(new_label)):
		if new_label[j] == 1:
			index = j
			break
	# print new_label
	char = text.convert_int_to_char(index)
	predictions.append(char)
#print predictions


####################
#
# print prediction v/s target and calculate the accuracy
#
####################
for i in range(len(predictions)):
	print "prediction = ",predictions[i], "target output = ",Y_test[i]
print len(predictions)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print accuracy


####################
#
# write the weights and bias to a file
#
####################
i2h = my_classifier.get_input_to_hidden_weights()
h2o = my_classifier.get_hidden_to_output_weights()
hb = my_classifier.hidden_layer.bias
ob = my_classifier.output_layer.bias

write_weights("weights.txt", i2h, h2o, hb, ob)

log("zoning_log.txt", "\n["+str(epochs)+", "+str(alpha)+"] : "+str(accuracy))

###########################
