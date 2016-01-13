#usr/bin/env python

import glob
from load_images import Images
from neural_network import Neuron, NeuralNetwork

class Savant():
	def __init__(self):
		self.vision = Images()
		self.knowledgeBase = []

	def loadKB(self):
		self.knowledgeBase = [x[14:] for x in glob.glob("knowledgeBase/*")]
		print "current known commands:"
		for knowledge in self.knowledgeBase:
			print knowledge.replace("_"," ")		#["train vision"] #need to load these from a file

	def learnAction(self,action):
		if raw_input("I have never done this before, should I attempt to learn it now? (y/n): ") == "y":
			actionSpecs = open("knowledgeBase/"+action.replace(" ","_"),"w")
			inputLayerUnits = input("How many input units do I need? ")
			hiddenLayerUnits = input("How many hidden units do I need? ")
			outputLayerUnits = input("How many output units do I need? ")
			actionSpecs.write(str(inputLayerUnits))
			actionSpecs.write("\n")
			actionSpecs.write(str(hiddenLayerUnits))
			actionSpecs.write("\n")
			actionSpecs.write(str(outputLayerUnits))
			actionSpecs.close()
			net = NeuralNetwork(action)
			print "making neural network..."
			net.makeNetwork(inputLayerUnits,hiddenLayerUnits,outputLayerUnits)
			print "initializing random weights..."
			net.initializeRandomWeights()
			print "generating input activations..."
			net.genInputActivations()
			try:
				print "training..."
				net.train()
			except KeyboardInterrupt:
				net.saveWeights()	
				print "weights saved"
				return
			except RuntimeError:
				print "Training failed."
				net.saveWeights()
				print "weights saved"
				return
			print "Success!"
			net.saveWeights()
			print "weights saved"
			self.knowledgeBase.append(action)
			print "testing..."
			net.test()
			return
		else:
			print "Ok, maybe later."		
			return

	def performAction(self,action):
		actionSpecs = open("knowledgeBase/"+action.replace(" ","_"),"r")
		inputLayerUnits = actionSpecs.readline()
		hiddenLayerUnits = actionSpecs.readline()
		outputLayerUnits = actionSpecs.readline()
		actionSpecs.close()
		net = NeuralNetwork(action)
		print "making neural network..."
		net.makeNetwork(int(inputLayerUnits),int(hiddenLayerUnits),int(outputLayerUnits))
		print "generating input activations..."
                net.genInputActivations()
                print "loading weights..."
		net.loadWeights("weights/"+action.replace(" ","_"))
		if raw_input("Continue training? (y/n): ") == "y":
                        try:
                                print "training..."
                                net.train()
                        except KeyboardInterrupt:
                                net.saveWeights()
                                print "weights saved"
                        except RuntimeError:
                                print "Training failed."
                                net.saveWeights()
                                print "weights saved"
                #need to do minus/plus phase here

	def trainVision(self):
		net = NeuralNetwork("vision")
		net.makeNetwork(1024,25,2)
		if raw_input("Continue Previous Training? (y/n): ") == "y":
			print "loading weights..."	
			net.loadWeights("weights/vision_2")	
		else:
			print "initializing random weights..."
			net.initializeRandomWeights()
		print "generating input activations..."
		net.genInputActivations()
		try:
			print "training..."
			net.train()
		except KeyboardInterrupt:
			net.saveWeights()	
			print "weights saved"
	
	def loadVision(self):
		#index = input("Enter image index: ")
		for index in range(len(self.vision.images["data"])-1):
			self.vision.saveImage(index)

def main():
	try:
		savant.loadKB()
		action = raw_input("What should I do right now? ")
		if action.replace(" ","_") in savant.knowledgeBase:
			savant.performAction(action)
			#savant.trainVision()
		else:
			savant.learnAction(action)
		return main()
	except KeyboardInterrupt:
		if raw_input("Quit? (y/n): ") == "y":
			return
		else:
			return main()

if __name__=="__main__":
	savant = Savant()
	#savant.loadVision()
	main()
