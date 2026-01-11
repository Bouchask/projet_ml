from sklearn.ensemble import VotingClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from data_preprocissing import Data 
class Models :
    def __init__(self,url,target_name):
        self.lr = LogisticRegression(max_iter=200)
        self.rf = RandomForestClassifier(n_estimators=100)
        self.knn = KNeighborsClassifier(n_neighbors=4)
        self.svc = SVC(kernel ="rbf")
        self.nB = GaussianNB()
        self.gbc = GradientBoostingClassifier(n_estimators=100)
        self.dcT = DecisionTreeClassifier(max_depth=4)
        self.obj_data = Data(url,target_name)
    def create_ensemble(self):
        self.voting_model = VotingClassifier(estimators= [
            ("lr",self.lr) ,
            ("knn",self.knn),
            ("rf" , self.rf),
            ("svc" , self.svc),
            ("nB" , self.nB),
            ("gbc" , self.gbc),
            ("dct" , self.dcT)
        ] , 
        voting="hard")
    def train_models(self):
        self.create_ensemble()
        param_grid = {
                         "lr__C": [0.01, 0.1, 1, 10],

                         "rf__n_estimators": [100, 200],
                         "rf__max_depth": [None, 5, 10],

                         "knn__n_neighbors": [3, 5, 7],

                         "svc__C": [0.1, 1, 10],
                         "svc__kernel": ["rbf","linear"],

                         "nB__var_smoothing": [1e-9, 1e-8],

                         "gbc__learning_rate": [0.01, 0.1],

                         "dct__max_depth": [3, 5, 7]
                        }
        grid = GridSearchCV( self.voting_model ,param_grid=param_grid, cv=5, n_jobs=-1,scoring="accuracy")  
        x_train,x_test ,y_train,y_test = self.obj_data.split_data()
        grid.fit(x_train,y_train)
        self.best_model = grid.best_estimator_
        print("Best params:", grid.best_params_)
        print("Best CV score:", grid.best_score_)

                        

