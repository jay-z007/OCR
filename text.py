
from PIL import Image
import os.path
import binascii

num = 200
alphabet = [
	['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
	['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
]

prefix = "./training/lower/"
suffix = ".gif"
data = []
target = []


for index in range(0, 2):
	im = []
	for n in range(num):
		fname = prefix+str(alphabet[1][index])+"/"+str(n)+suffix
		
		if os.path.exists(fname):
			im.append(Image.open(fname))
		else:
			continue

		pixels = list(im[n].getdata())
		width, height = im[n].size
		pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]

		im[n].close()

		# print ""
		# print ""
		# print fname
		# print "index - ", index, " n - ", n
		# print "Raw image"
		# print "********************"

		# for i in range(height):
		# 	print pixels[i]

		#print "********************"

		img_matrix = []
		row = []


		for x in range(width):
			for y in range(height):
				curr_pixel = pixels[x][y]
				if curr_pixel >= 20:
					curr_pixel = 1
				else:
					curr_pixel = 0
				img_matrix.append(curr_pixel)

		#print img_matrix
		#row = [img_matrix, alphabet[0][index]]
		data.append(img_matrix)
		target.append(alphabet[0][index])

def convert_char_to_bin(char):
	return bin(int(binascii.hexlify(char), 16))

def convert_char_to_int(char):
	return int(binascii.hexlify(char), 16)

def convert_bin_to_char(bin):
	n = int(bin, 2)
	return binascii.unhexlify('%x' %n)

def convert_int_to_char(index):
	return alphabet[0][index]
