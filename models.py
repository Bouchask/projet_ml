from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from data_preprocissing import Data
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np 
class Models :
    def __init__(self,url,target_name):
        self.obj_data = Data(url,target_name)
        self.lr = LogisticRegression(penalty="l2",max_iter=100,solver="lbfgs",C=0.1)
        self.rf = RandomForestClassifier(n_estimators=100)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.dtree = DecisionTreeClassifier()
        self.svm = SVC(C=0.1, kernel="rbf", gamma="scale",probability=True)
        self.X_train,self.X_test,self.Y_train,self.Y_test = self.obj_data.split_data()
    def create_ensemble_models(self):
        voting = VotingClassifier(estimators=[
            ("lr",self.lr),
            ("dtree" , self.dtree),
            ("knn" , self.knn),
            ("rf" , self.rf),
            ("svc" , self.svm)
        ],
        voting="soft")
        return voting
    def create_gridSearch(self):
        param_grid = {"lr__C"  : [0.1 , 0.01 , 0.12 , 0.14 , 1] ,
                      "rf__n_estimators" : [100 ,50 ,60 ,40],
                      "knn__n_neighbors" : [5,4,3],
                      "svc__C" : [0.1 , 0.01 , 0.12 , 0.14 , 1]}
        voting = self.create_ensemble_models()
        grid = GridSearchCV(estimator=voting , param_grid=param_grid , cv=5 , scoring="accuracy" , n_jobs=-1)
        return grid
    def train_models(self):
        grid = self.create_gridSearch()
        grid.fit(self.X_train , self.Y_train)
        return grid.score(self.X_test , self.Y_test)
