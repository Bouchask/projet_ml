from models import Models

model = Models(
    "/home/yahya-bouchak/projet_ml/data/TrafficTwoMonth.csv",
    "Traffic Situation"
)

model.train_models()
