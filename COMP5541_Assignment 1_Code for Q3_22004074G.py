from numpy import dtype
from sklearn.datasets import load_iris
import numpy as np

#define class of Naive Bayes Classifier
class NBC:
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes

    def fit(self, Xtrain, ytrain, epsilon=1e-6): #training dataset
        self.epsilon = epsilon
        self.classes_list = np.unique(ytrain)  # find the unique element from the 3 classes of iris
        num_classes=len(self.classes_list)
        num_features = len(self.feature_types)
        # find the mean, variance and prior probability
        self.mean = np.zeros((num_classes, num_features))
        self.variance = np.zeros((num_classes,num_features))
        self.prior = np.zeros(num_classes, dtype=np.float64) #Frequency for each classes, a vector of two(1,0)

        for i, c in enumerate(self.classes_list):
        # populate every data from xtrain and corresponding to each class in each iteration
            X_i = Xtrain[ytrain==c] #compare matrix with ytrain and get the indices
            #calculate the mean to the row of index in each iteration in every columns
            self.mean[i, : ] = np.mean(X_i, axis=0) #applying the mean of every row
            self.variance[i, : ] = np.var(X_i, axis=0) #Compute the variance of Xtrain
            #Get the size of the matrix and calculate for the prior probaility
            self.prior[i] = X_i.shape[0] / float(N) #returns a tuple with each index having the number of corresponding elements
        return self.mean, self.variance, self.prior

    def prediction(self, Xtest):
        prediction = []
        for i in range(len(Xtest)):
            # find the class y with maximum probability
            posterior = np.argmax(self.gaussian(Xtest)[i]) #return vector of 4 features
            prediction.append(posterior)
        return prediction

    # Helper function to get the conditional distribution for Gaussian pdf in prediction
    def gaussian(self, Xtest):
        class_prob = []
        for i in range(len(self.classes_list)): #Evaluate the value of each classes in each literation
            mean = self.mean[i, : ]  # mean for the feature x
            variance = self.variance[i, : ]  # variance for the feature x
            # Gaussian PDF equation
            numerator = np.exp(- (Xtest - mean) ** 2 / (2 * variance))
            denominator = np.sqrt(2 * np.pi * variance)
            pdf = numerator / denominator # conditional probability formula
            class_conditional = np.sum(np.log(pdf), axis=1) #sum of class conditional probability
            prior = np.log(self.prior[i]) #grab the 1st index
            class_prob.append(prior + class_conditional) #obtain the class, given the predictors
        return np.transpose(np.array(class_prob))


#LOAD DATASET
iris = load_iris()
X, y = iris['data'], iris['target']
N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

'''implementation'''
nbc = NBC(feature_types=['r', 'r', ' r', 'r'], num_classes=3)
nbc.fit(Xtrain, ytrain)
yhat = nbc.prediction(Xtest)
test_accuracy = np.mean(yhat == ytest)
format_float = "{0:.2f}%".format(test_accuracy*100)
print("Mean Accuracy:",format_float)
