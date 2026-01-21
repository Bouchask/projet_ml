from sklearn.model_selection import GridSearchCV,StratifiedKFold
class Optimizer:
    def __init__(self,pipelines , param_grids):
        self.pipelines = pipelines
        self.param_grids = param_grids
        self.Kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def run(self , X_train , Y_train):
        result = {}
        for name,pipe in self.pipelines.items():
            gs = GridSearchCV(pipe,self.param_grids[name],cv=self.Kfolder,n_jobs=-1,return_train_score=True,scoring="f1_macro")
            gs.fit(X_train,Y_train)
            result[name] = {
                "best_params":gs.best_params_,
                "best_score":gs.best_score_,
                "best_model":gs.best_estimator_
            }
        return result