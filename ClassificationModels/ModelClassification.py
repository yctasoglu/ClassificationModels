from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from lightgbm import LGBMClassifier

class ModelClassification:

    self.models = {
        "adaboost": AdaBoostClassifier(),
        "catboost": CatBoostClassifier(),
        "kernelsvm": SVC(),
        "knn": KNeighborsClassifier(),
        "lgbm": LGBMClassifier(),
        "logistic": LogisticRegression(),
        "nativebayes": GaussianNB(),
        "randomforest": RandomForestClassifier(),
        "sgd": SGDClassifier(),
        "xgboost": XGBClassifier()
    }    

    def __init__(self, X, y, cv):
        self.__X = X
        self.__y = y
        self.__cv = cv

    def __init__(self, X_train, X_test, y_train, y_test):
        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test   

    def model_creation(self, model_name, grid_values):
        if grid_values != None:
            self.__clf = GridSearchCV(self.models[model_name], param_grid = grid_values, scoring = 'accuracy')
        else:
            self.__clf = self.models[model_name]

    def predict(self):
        self.__clf.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__clf.predict(self.__X_test)

    def cross_validation(self):
        self.__scores = cross_val_score(self.__clf, self.__X, self.__y, self.__cv, scoring='accuracy')
    
    def calc_accuracy(self):
        return {
            "accuracy_score" : accuracy_score(self.__y_test,self.__y_pred),
            "precision_score" : precision_score(self.__y_test,self.__y_pred),
            "recal_score" : recall_score(self.__y_test,self.__y_pred),
            "f1_score" : f1_score(self.__y_test,self.__y_pred),
            "kfold_mean" : self.__scores.mean(),
            "kfold_std" : self.__scores.std()
        }
    
    def get_y_pred(self):
        self.__y_pred
        
