from models import Models

model = Models(
    "/home/yahya-bouchak/projet_ml/data/TrafficTwoMonth.csv",
    "Traffic Situation"
)

score = model.train_models()
print(score)
