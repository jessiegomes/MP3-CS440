import numpy as np
import matplotlib.pyplot as plt

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model. 

		This function will initialize prior and likelihood, where 
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of 
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		    num_value(int): number of possible values for each pixel 
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))
		self.highest_posterior_prob = np.array([float('-inf'), float('-inf'), float('-inf'),float('-inf'),float('-inf'),float('-inf'),float('-inf'),float('-inf'),float('-inf'),float('-inf')])
		self.highest_pic_index = np.zeros(num_class)
		self.lowest_posterior_prob = np.array([float('inf'), float('inf'), float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf')])
		self.lowest_pic_index = np.zeros(num_class)
	
	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset. 
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE
		for picture in range(0, len(train_set)):
			for pix_index, pixel in enumerate(train_set[picture]):
				self.likelihood[pix_index][pixel][train_label[picture]] += 1

		K = 1
		for c in range(0, self.num_class):
			self.prior[c] = np.sum(train_label==c)/float(train_label.shape[0]) 
			self.likelihood[:,:,c] = (self.likelihood[:,:,c] + K) / ((K*self.num_value) + (self.prior[c] * len(train_set)))

		# print(self.likelihood)
		# print(self.likelihood[230][201][0])
		# print(self.likelihood[234][245])


	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.  
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
		pred_label = np.zeros((len(test_set)))
		accuracy = 0
		for idx, image in enumerate(test_set):
			class_probs = np.zeros(self.num_class)
			for category in range(0, self.num_class):
				curr_prob = 0
				for pix_index in range(0, len(test_set[image])):
					curr_prob += np.log(self.likelihood[pix_index][image[pix_index]][category])
				curr_prob += np.log(self.prior[category])
				class_probs[category] = curr_prob
			pred_label[idx] = np.argmax(class_probs)
			if pred_label[idx] == test_label[idx]:
				accuracy += 1
			if class_probs[test_label[idx]] < self.lowest_posterior_prob[test_label[idx]]:
				self.lowest_posterior_prob[test_label[idx]] = class_probs[test_label[idx]]
				self.lowest_pic_index[test_label[idx]] = idx
			if class_probs[test_label[idx]] > self.highest_posterior_prob[test_label[idx]]:
				self.highest_posterior_prob[test_label[idx]] = class_probs[test_label[idx]]
				self.highest_pic_index[test_label[idx]] = idx

		accuracy /= len(test_set)
		print("Highest", self.highest_pic_index)
		print("Lowest", self.lowest_pic_index)
		return accuracy, pred_label

	def display_highest_posterior_prob_images(self, test_set, test_label):
		for i in self.highest_pic_index:
			pic = test_set[int(i)]
			pic = np.reshape(pic, (28,28))
			plt.imshow(pic, cmap='gray')
			plt.show()

	def display_lowest_posterior_prob_images(self, test_set, test_label):
		for i in self.lowest_pic_index:
			pic = test_set[int(i)]
			pic = np.reshape(pic, (28,28))
			plt.imshow(pic, cmap='gray')
			plt.show()
	
	def save_model(self, prior, likelihood):
		""" Save the trained model parameters 
		"""    

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters 
		""" 

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
	    """
	    Get the feature likelihoods for high intensity pixels for each of the classes,
	        by sum the probabilities of the top 128 intensities at each pixel location,
	        sum k<-128:255 P(F_i = k | c).
	        This helps generate visualization of trained likelihood images. 
	    
	    Args:
	        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	            (# of features/pixels per image, # of possible values per pixel, # of class)
	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	            (# of features/pixels per image, # of class)
	    """
	    # YOUR CODE HERE
	    
	    feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))
	    for c in range(0, self.num_class):
		    feature_likelihoods[:,c] = np.sum(self.likelihood[:,128:256,c], axis=1)

	    return feature_likelihoods
