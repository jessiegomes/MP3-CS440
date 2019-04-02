import numpy as np

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model. 

		This function will initialize a feature_dim weight vector,
		for each class. 

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS] 
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		"""

		self.w = np.zeros((feature_dim+1,num_class))

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		"""
		METHOD:
		use np.dot to find max product as prediction. check correctness of prediction
		and update weights based on validity
		"""
		# list of 10 empty lists
		all_dots = [ [], [], [], [], [], [], [], [], [], [] ]
		image_idx = -1
		# walk through each image in train and find max dot prod
		for curr_image in train_set:
			image_idx += 1
			for i in range(10):
				all_dots[i] = np.dot(self.w[0:784, i], curr_image)

			prediction = np.argmax(all_dots)

			if train_label[image_idx] != prediction:
				# update prediction and train label
				# bias is 1
				self.w[784, prediction] -= 1		
				self.w[784, train_label[image_idx]] += 1

				self.w[0:784, prediction] -= curr_image
				self.w[0:784, train_label[image_idx]] += curr_image

		pass

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset. 
			The accuracy is computed as the average of correctness 
			by comparing between predicted label and true label. 
			
		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value 
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""    

		# YOUR CODE HERE
		accuracy = 0 
		pred_label = np.zeros((len(test_set)))
		# list of 10 empty lists
		all_dots = [ [], [], [], [], [], [], [], [], [], [] ]
		image_idx = -1
		# walk through each image in train and find max dot prod
		for curr_image in test_set:
			image_idx += 1
			for i in range(10):
				all_dots[i] = np.dot(self.w[0:784, i], curr_image)

			prediction = np.argmax(all_dots)
			pred_label[image_idx] = prediction

			if test_label[image_idx] == prediction:
				# update accuracy
				accuracy += 1
		
		print(accuracy/len(test_label))
		pass
		return accuracy/len(test_label), pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters 
		""" 

		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		""" Load the trained model parameters 
		""" 

		self.w = np.load(weight_file)

