from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from data_preprocissing import Data 
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np 
class Models :
    def __init__(self,url,target_name):
        self.obj_data = Data(url,target_name)
        self.lr = LogisticRegression(penalty="l2",max_iter=100,solver="lbfgs")
        self.rf = RandomForestClassifier(n_estimators=100)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.dtree = DecisionTreeClassifier()
        self.svm = SVC(C=0.1, kernel="rbf", gamma="scale")
        self.X_train,self.X_test,self.Y_train,self.Y_test = self.obj_data.split_data()
    def create_ensemble_models(self):
        voting = VotingClassifier(estimators=[
            ("lr",self.lr),
            ("dtree" , self.dtree),
            ("knn" , self.knn),
            ("rf" , self.rf),
            ("svc" , self.svm)
        ],
        voting="hard")
        return voting
    def train_models(self):
        voting = self.create_ensemble_models()
        voting.fit(self.X_train , self.Y_train)
        return voting.score(self.X_test , self.Y_test)
