
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

ridge = RidgeClassifier(alpha = 2, solver = "sag")
logit = LogisticRegression(solver = "sag")
perceptron = Perceptron(n_iter = 50)
passiveAggressive = PassiveAggressiveClassifier(n_iter = 20, loss = 'hinge')
knn = KNeighborsClassifier(n_neighbors = 5)
nearestCentroid = NearestCentroid()
L1SVC = LinearSVC(loss = 'squared_hinge', penalty = 'l1', dual = False)
L2SVC = LinearSVC(loss = 'squared_hinge', penalty = 'l2', dual = False)
L1SGD = SGDClassifier(alpha = .0001, n_iter = 10, penalty = 'L1')
L2SGD = SGDClassifier(alpha = .0001, n_iter = 10, penalty = 'L2')
elasticNet = SGDClassifier(alpha = .0001, n_iter = 175, penalty = "elasticnet")
MNB = MultinomialNB(alpha = .01)
BNB = BernoulliNB(binarize = .01, alpha = .01)
pipeline = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty = "l1", dual = False, tol = 1e-3))),
  ('classification', LinearSVC())
])
randomForest = RandomForestClassifier(n_estimators = 100)
adaBoost = AdaBoostClassifier(n_estimators = 5)
bagging = BaggingClassifier(n_estimators=23)

classifiers = [ridge, logit, perceptron, passiveAggressive, knn, nearestCentroid, L1SVC, L2SVC, L1SGD, L2SGD, elasticNet, MNB, BNB, pipeline, randomForest, adaBoost, bagging]
classifier_names = ['ridge', 'logit', 'perceptron', 'passiveAggressive', 'knn', 'nearestCentroid', 'L1SVC', 'L2SVC', 'L1SGD', 'L2SGD', 'elasticNet', 'MNB', 'BNB', 'pipeline', 'randomForest', 'adaBoost', 'bagging']

'''
try to save all the classification model to pkl using first set of training data: % by 
def save(self, path):
    with open(path, 'wb') as f:
        pickle.dump(self.params, f, -1)
    return None
    
'''