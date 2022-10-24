# Naive-Bayes-Classifier
Implement NBC on the Iris dataset
shuffle the dataset, put 20% aside for testing.
N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

For training data Xtrain, ytrain and test data Xtest, ytest enable to run:
nbc.fit(Xtrain, ytrain)
yhat = nbc.predict(Xtest)
test accuracy = np.mean(yhat == ytest)
