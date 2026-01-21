from data_preprocissing import Data
from models import Models
from optimization import Optimizer
from evaluate import Evaluator
from param_grids import PARAM_GRIDS

data = Data(r"C:\Users\VoxxF\Desktop\Projet\projet_ml\data\TrafficTwoMonth.csv", "Traffic Situation")
X_train, X_test, y_train, y_test = data.split_data()

models = Models()
pipelines = models.create_all_pipelines()

optimizer = Optimizer(pipelines, PARAM_GRIDS)
results = optimizer.run(X_train, y_train)

evaluator = Evaluator(X_test, y_test)
evaluator.plot(results)
