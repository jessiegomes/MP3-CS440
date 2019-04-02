# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

        self.laplace = 0.5
        self.num_class = 14
        self.prior = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.likelihood = []
        self.bigram_likelihood = []
        self.classes = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
        self.top_feature_words = []

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        for i in range(0, self.num_class): 
            diction = {}
            self.likelihood.append(diction)
            self.bigram_likelihood.append(diction)

        for idx, sentence in enumerate(train_set):
            self.prior[train_label[idx] - 1] += 1
            for word in sentence:
                for group in range(0, self.num_class):
                    if word not in self.likelihood[group]:
                        self.likelihood[group][word] = 0
                if word not in self.likelihood[train_label[idx] - 1]:
                    self.likelihood[train_label[idx] - 1][word] = 1
                else:
                    self.likelihood[train_label[idx] - 1][word] += 1

        for idx, sentence in enumerate(train_set):
            for i in range(0, len(sentence)):
                for j in range(i+1, len(sentence)):
                    init_tup = (sentence[i], sentence[j])
                    sort_tup = sorted(init_tup)
                    tup = (sort_tup[0], sort_tup[1])
                    for group in range(0, self.num_class):
                        if tup not in self.bigram_likelihood[group]:
                            self.bigram_likelihood[group][tup] = 0
                    if tup not in self.bigram_likelihood[train_label[idx] - 1]:
                        self.bigram_likelihood[train_label[idx] - 1][tup] = 1
                    else:
                        self.bigram_likelihood[train_label[idx] - 1][tup] += 1
        
        self.prior = [i/len(train_set) for i in self.prior]

        for idx, class_dict in enumerate(self.likelihood):
            for key in class_dict:
                class_dict[key] = (class_dict[key] + self.laplace)/((self.laplace*sum(class_dict.values())) + (self.prior[idx]*len(train_set)))

        for idx, class_dict in enumerate(self.bigram_likelihood):
            for key in class_dict:
                class_dict[key] = (class_dict[key] + self.laplace)/((self.laplace*sum(class_dict.values())) + (self.prior[idx]*len(train_set)))

    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        for idx, sentence in enumerate(x_set):
            class_probs = []
            uni_probs = []
            bi_probs = []
            for category in range(0, self.num_class):
                curr_prob = 0
                mix_prob = 0
                for word in range(0, len(sentence)):
                    if word in self.likelihood[category]:
                        curr_prob += math.log(self.likelihood[category][word])
                    for word2 in range(word+1, len(sentence)):
                        init_tup = (sentence[word], sentence[word2])
                        sort_tup = sorted(init_tup)
                        tup = (sort_tup[0], sort_tup[1])
                        if tup in self.bigram_likelihood[category]:
                            mix_prob += math.log(self.bigram_likelihood[category][tup])
                curr_prob += math.log(self.prior[category])
                mix_prob += math.log(self.prior[category])
                uni_probs.append(curr_prob)
                bi_probs.append(mix_prob)
                class_probs.append((uni_probs[category]*(1-self.lambda_mixture)) + (bi_probs[category]*(self.lambda_mixture)))
            result.append(class_probs.index(max(class_probs)) + 1)
            if dev_label[idx] == result[idx]:
                accuracy += 1
        # print(self.top_feature_words)
        accuracy /= len(x_set)
        return accuracy, result

    def bigram_fit(self, train_set, train_label):
        for i in range(0, self.num_class): 
            diction = {}
            self.likelihood.append(diction)

        for idx, sentence in enumerate(train_set):
            self.prior[train_label[idx] - 1] += 1
            for i in range(0, len(sentence) - 1):
                for j in range(i+1, len(sentence)):
                    init_tup = (sentence[i], sentence[j])
                    sort_tup = sorted(init_tup)
                    tup = (sort_tup[0], sort_tup[1])
                    for group in range(0, self.num_class):
                        if tup not in self.likelihood[group]:
                            self.likelihood[group][tup] = 0
                    if tup not in self.likelihood[train_label[idx] - 1]:
                        self.likelihood[train_label[idx] - 1][tup] = 1
                    else:
                        self.likelihood[train_label[idx] - 1][tup] += 1
            
        self.prior = [i/len(train_set) for i in self.prior]

        for idx, class_dict in enumerate(self.likelihood):
            for key in class_dict:
                class_dict[key] = (class_dict[key] + self.laplace)/((self.laplace*len(class_dict.keys())) + (self.prior[idx]*len(train_set)))

    def bigram_predict(self, x_set, dev_label,lambda_mix=0.0):
        accuracy = 0.0
        result = []

        # TODO: Write your code here
        for idx, sentence in enumerate(x_set):
            class_probs = []
            for category in range(0, self.num_class):
                curr_prob = 0
                for i in range(0, len(sentence) - 1):
                    for j in range(i+1, len(sentence)):
                        init_tup = (sentence[i], sentence[j])
                        sort_tup = sorted(init_tup)
                        tup = (sort_tup[0], sort_tup[1])
                        if tup in self.likelihood[category]:
                            curr_prob += math.log(self.likelihood[category][tup])
                curr_prob += math.log(self.prior[category])
                class_probs.append(curr_prob)
            result.append(class_probs.index(max(class_probs)) + 1)
            if dev_label[idx] == result[idx]:
                accuracy += 1
        # print(self.top_feature_words)
        accuracy /= len(x_set)
        return accuracy, result


