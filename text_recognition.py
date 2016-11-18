
import random
from numpy import *
from neural_network import *
import text

X = text.data
Y = text.target
X_len = len(X)
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')

#from sklearn.svm import SVC
#my_classifier = SVC()

# not available in sklearn v17
# from sklearn.neural_network import MLPClassifier
# my_classifier = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(50))

# not good result
# from sklearn.ensemble import AdaBoostClassifier
# my_classifier = AdaBoostClassifier(n_estimators=50)

num_input_nodes = 400
num_hidden_nodes = 50
num_output_nodes = 7

hidden_layer_weights = []
output_layer_weights = []
hidden_layer_bias = 0.35
output_layer_bias = 0.6
# hidden_layer_weights=[round(random.random()*random.randint(-1, 2), 3) for i in range(num_input_nodes*num_hidden_nodes)]
# output_layer_weights=[round(random.random()*random.randint(-1, 2), 3) for i in range(num_hidden_nodes*num_output_nodes)]

with open("weights.txt") as file:

	for i in range(num_input_nodes):
		for h in range(num_hidden_nodes):
			hidden_layer_weights.append(float(file.readline()))

	hidden_layer_bias = float(file.readline())

	for h in range(num_hidden_nodes):
		for o in range(num_output_nodes):
			output_layer_weights.append(float(file.readline()))

	output_layer_bias = float(file.readline())


my_classifier = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes, hidden_layer_weights=hidden_layer_weights, 
	hidden_layer_bias=hidden_layer_bias, output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias)
#my_classifier = NeuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes)
bin_Y_train = []

for label in Y_train:
	bin = text.convert_char_to_bin(label)
	arr = [int(i) for i in str(bin)[2:]]
	bin_Y_train.append(arr)

for i in range(10):
	print i
	my_classifier.train(X_train, bin_Y_train)

#my_classifier.fit(X_train, Y_train)
##########################

bin_predictions = my_classifier.predict(X_test)

predictions = []

for label in bin_predictions:
	new_label = []
	for i in label:
		new_label.append(str(int(round(i))))
	bin = ''.join(new_label)
	bin = "0b"+bin
	char = text.convert_bin_to_char(bin)
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
