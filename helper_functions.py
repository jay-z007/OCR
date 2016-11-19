import os.path
from init import *

####################
#
#	Used to print the result to a log file
#
####################
def log(fname, text):
	with open(fname, "a+") as file:
		file.write(text)


####################
#
#	Used to read the weights from a file and return it to the caller
#
####################
def read_weights(fname):
	hidden_layer_weights = []
	output_layer_weights = []
	hidden_layer_bias = 0.35
	output_layer_bias = 0.6

	if os.path.exists(fname):
		with open(fname) as file:
			for i in range(num_input_nodes):
				for h in range(num_hidden_nodes):
					hidden_layer_weights.append(float(file.readline()))

			hidden_layer_bias = float(file.readline())

			for h in range(num_hidden_nodes):
				for o in range(num_output_nodes):
					output_layer_weights.append(float(file.readline()))

			output_layer_bias = float(file.readline())
		
	return hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias


####################
#
#	Used to write the updated weughts to a file 
#
####################
def write_weights(fname, i2h, h2o, hb, ob):
	with open(fname, 'w') as file:
		for i in i2h:
			for h in i:
				file.write("%f\n"%h)

		file.write("%f\n"%hb)

		for h in h2o:
			for o in h:
				file.write("%f\n"%o)

		file.write("%f\n"%ob)
		