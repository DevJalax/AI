from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron

# Dataset definition
# make classification method generates random n-class classification problem
#X = generated samples
#y = integer label for class memberhip of each sample

X,y = make_classification(n_samples=1000,n_features = 10, n_informative = 10 , n_redundant = 0 , random_state = 1)

#model definition

model = Perceptron(eta0=0.0001)

#model evaluation method

cv = RepeatedStratifiedKFold(n_splits = 10 , n_repeats = 3 , random_state = 1)

#grid definition

grid = dict()
grid['max_iter'] = [1,10,100,1000,10000]

#Search

search = GridSearchCV(model,grid,scoring='accuracy',cv=cv,n_jobs=-1)
results = search.fit(X,y)

#Summary/ Results

print('Mean Accuracy: %.3f' % results.best_score_)

print('Config: %s' % results.best_params_)

means = results.cv_results_['mean_test_score']

params = results.cv_results_['params']

for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
