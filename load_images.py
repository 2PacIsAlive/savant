#!usr/bin/env python

class Images():
	def __init__(self):
		self.images, self.batches = self.unpickle("images/data_batch_1")
	
	def unpickle(self, file):
		import cPickle
		fo = open(file, 'rb')
	   	dict = cPickle.load(fo)
	   	fo.close()
		fo = open("images/batches.meta", 'rb')
		batches = cPickle.load(fo)
		fo.close()
	   	return dict, batches

	def saveImage(self, index):
		#import matplotlib.pyplot as plt
		#import matplotlib.image as mpimg
		import glob
		filename = "images/saved/"+self.batches["label_names"][self.images["labels"][index]]+"_"+str(index)+".jpeg"
		if filename not in glob.glob("images/saved/*"):
			import numpy as np
			#for key in self.images.keys():
			#	print key
			#print self.images["labels"][index]
			#print self.images["batch_label"][self.images["labels"][index]]
			im = self.images["data"][index].reshape(3,32,32).transpose(1,2,0)
			#plt.imsave("images/saved/"+index"im)
			from PIL import Image
			im = Image.fromarray(im)
			im.save(filename)
