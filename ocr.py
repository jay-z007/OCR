
import random
from numpy import *
from neural_network import *
import text

X = text.data
Y = text.target
X_len = len(X)
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1)

num_input_nodes = 16
num_hidden_nodes = 30
num_output_nodes = 26

hidden_layer_weights = []
output_layer_weights = []
hidden_layer_bias = 0.35
output_layer_bias = 0.6
# hidden_layer_weights=[round(random.random()*random.randint(-1, 2), 3) for i in range(num_input_nodes*num_hidden_nodes)]
# output_layer_weights=[round(random.random()*random.randint(-1, 2), 3) for i in range(num_hidden_nodes*num_output_nodes)]

# with open("weights.txt") as file:

# 	for i in range(num_input_nodes):
# 		for h in range(num_hidden_nodes):
# 			hidden_layer_weights.append(float(file.readline()))

# 	hidden_layer_bias = float(file.readline())

# 	for h in range(num_hidden_nodes):
# 		for o in range(num_output_nodes):
# 			output_layer_weights.append(float(file.readline()))

# 	output_layer_bias = float(file.readline())

len_Y_train = len(Y_train)

my_classifier = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes, hidden_layer_weights=hidden_layer_weights, 
	hidden_layer_bias=hidden_layer_bias, output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias)
#my_classifier = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes)
bin_Y_train = [[0]*26 for i in range(len_Y_train)]

for i in range(len_Y_train):
	index = text.convert_char_to_int(Y_train[i]) - text.convert_char_to_int('a')
	bin_Y_train[i][index] = 1
	# print Y_train[i], bin_Y_train[i]


for i in range(10):
	print i
	my_classifier.train(X_train, bin_Y_train)

##########################

bin_predictions = my_classifier.predict(X_test)

predictions = []

for label in bin_predictions:
	index = label.index(1)
	char = text.convert_int_to_char(index)
	predictions.append(char)

# #print predictions
for i in range(len(predictions)):
	print "prediction = ",predictions[i], "actual output = ",Y_test[i]
print len(predictions)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)

i2h = my_classifier.get_input_to_hidden_weights()
h2o = my_classifier.get_hidden_to_output_weights()

with open("weights.txt", 'w') as file:
	for i in i2h:
		for h in i:
			file.write("%f\n"%h)

	hb = my_classifier.hidden_layer.bias
	file.write("%f\n"%hb)

	for h in h2o:
		for o in h:
			file.write("%f\n"%o)

	ob = my_classifier.output_layer.bias
	file.write("%f\n"%ob)

###########################

# for i in range(height):
# 	for j in range(width)
# 	print img_matrix[i*width + j]," "

# ' '.join('{0:08b}'.format(ord(x), 'b') for x in image_data)
#print "image_data = ", repr(image_data)

# img = Image.open('./training/lower/A/0.gif')
