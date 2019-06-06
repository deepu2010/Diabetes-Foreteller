import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
url = 'https://s3-us-west-1.amazonaws.com/personaldeepakbucket/My+files/diabetes.csv'
data = pd.read_csv(url)
# remove the Outlier of skin thickness

max_skinthickness = data.SkinThickness.max()
data = data[data.SkinThickness!=max_skinthickness]

# create a helper function
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]

    # run the function
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:   
    replace_zero(data, col, 'Outcome')   
    
#Partioning X and Y
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)


#Importing all ML models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score


from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
params = {'n_estimators':1200, 'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}
gbc = GradientBoostingClassifier(**params)

n_estimators = 10
clf = gbc

# split training set into training and testing set
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.3, random_state=100)

X_train_cv = X_train_cv.reset_index(drop=True, inplace=False)
y_train_cv = y_train_cv.reset_index(drop=True, inplace=False)


clf.fit(X_train_cv,y_train_cv)
#score = f1_score(y_train, clf.predict(X_train), pos_label = 1)
acc = clf.score(X_test_cv, y_test_cv)
    
n_estimators = params['n_estimators']
x = np.arange(n_estimators) + 1

""" The following part code was stole from sklearn """
def heldout_score(clf, X_test_cv, y_test_cv):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test_cv)):
        score[i] = clf.loss_(y_test_cv, y_pred)
    return score


def cv_estimate(n_splits=10):
    cv = KFold(n_splits=n_splits)
    cv_clf = clf
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train_cv):
        cv_clf.fit(X_train_cv.iloc[train], y_train_cv[train])
        val_scores += heldout_score(cv_clf, X_train_cv.iloc[test], y_train_cv[test])
    val_scores /= n_splits
    return val_scores


# Estimate best n_estimator using cross-validation
cv_score = cv_estimate(3)

# Compute best n_estimator for test data
test_score = heldout_score(clf, X_test_cv, y_test_cv)

# negative cumulative sum of oob improvements
cumsum = -np.cumsum(clf.oob_improvement_)

# min loss according to OOB
oob_best_iter = x[np.argmin(cumsum)]

# min loss according to test (normalize such that first loss is 0)
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]

# min loss according to cv (normalize such that first loss is 0)
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]


# helper functions
def train_clf(clf, X_train, y_train):
    
    return clf.fit(X_train, y_train)
    
def pred_clf(clf, features, target):
    
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label = 1)

def train_predict(clf, X_train, y_train, X_test, y_test):
    
    train_clf(clf, X_train, y_train)
    
    print("F1 score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)))
    print("F1 score for testing set is: {:.4f}\n".format(pred_clf(clf, X_test, y_test)))
    
    
from sklearn.metrics import accuracy_score

params = {'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}
gbc = GradientBoostingClassifier(n_estimators=290, **params)
clf_ = gbc.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
print('Accuracy is {}'.format(accuracy_score(y_test,y_pred )))
train_predict(gbc, X_train, y_train, X_test, y_test)


#pickling my model
from sklearn.externals import joblib
joblib.dump(gbc, 'model1.pkl')
print("Model dumped!")
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Model columns dumped")