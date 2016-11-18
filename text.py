
from PIL import Image
import os.path
import binascii

num = 200
alphabet = [
	['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
	['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
]

data = []
target = []

def print_char(fname, pixels, n):
	print ""
	print ""
	print fname
	print " n - ", n
	print "Raw image"
	print "********************"

	for i in range(height):
		print pixels[i]

	print "********************"

def get_name(char, n, case):
	prefix = "./training/lower/"
	if case == 1:
		prefix = "./training/upper/"
	suffix = ".gif"
	fname = prefix+char+"/"+str(n)+suffix
	return fname

def binarization(pixels, size):
	for x in range(size):			
			if pixels[x] >= 15:
				pixels[x] = 1
			else:
				pixels[x] = 0

	return pixels

def zoning(pixels, zone_num_x, zone_num_y):
	height = len(pixels)
	width = len(pixels[0])

	zone_size_x = width/zone_num_x
	zone_size_y = height/zone_num_y

	feature_vector = []

	for y in range(zone_num_y):
		for x in range(zone_num_x):
			count = read_block(pixels, x, y, zone_size_x, zone_size_y)
			feature_vector.append(float(count)/float(zone_size_x*zone_size_y))

	return feature_vector

	# for i in range(4):
	# 	print feature_vector[i*4:(i+1)*4]


def read_block(pixels, x, y, zone_size_x, zone_size_y):
	count = 0
	for j in range(y*zone_size_y, (y+1)*zone_size_y):
		for i in range(x*zone_size_x, (x+1)*zone_size_x):
			if pixels[i][j] == 1:
				count = count + 1
	return count

def convert_char_to_bin(char):
	return bin(int(binascii.hexlify(char), 16))

def convert_char_to_int(char):
	return int(binascii.hexlify(char), 16)

def convert_bin_to_char(bin):
	n = int(bin, 2)
	return binascii.unhexlify('%x' %n)

def convert_int_to_char(index):
	if index == -1:
		return ' '
	else:
		return alphabet[0][index]

for index in range(0, 26):
	im = []
	for n in range(num):
		
		char = str(alphabet[1][index])
		fname = get_name(char, n, 0)

		if os.path.exists(fname):
			im.append(Image.open(fname))
		else:
			continue

		pixels = list(im[n].getdata())
		width, height = im[n].size

		pixels = binarization(pixels, width*height)
		
		pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
		#print_char(fname, pixels, n)

		im[n].close()

		#img_matrix = []

		# for x in range(width):
		# 	for y in range(height):
		# 		curr_pixel = pixels[x][y]
		# 		if curr_pixel >= 20:
		# 			curr_pixel = 1
		# 		else:
		# 			curr_pixel = 0
		# 		img_matrix.append(curr_pixel)

		data.append(pixels)
		target.append(alphabet[0][index])

# pixels = [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# zoning(pixels, 4, 4)