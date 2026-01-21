from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import os
class Evaluator:
    def __init__(self, X_test,Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
    def evaluate(self,list_result_pipelines):
        result = {}
        for name,pipe in list_result_pipelines.items():
            Y_pred = pipe["best_model"].predict(self.X_test)
            result[name] = {
                "accuracy":accuracy_score(self.Y_test,Y_pred),
                "precision":precision_score(self.Y_test,Y_pred,average="macro"),
                "recall":recall_score(self.Y_test,Y_pred,average="macro"),
                "f1":f1_score(self.Y_test,Y_pred,average="macro"),
                "confusion_matrix":confusion_matrix(self.Y_test,Y_pred),
                "classification_report":classification_report(self.Y_test,Y_pred)
            }
        return result
    def plot(self, list_result_pipelines):
    # create folder for plots
        if not os.path.exists("plots"):
            os.makedirs("plots")

    # get evaluation results
        results = self.evaluate(list_result_pipelines)

    # --------- 1) FEATURE IMPORTANCE (if available) ----------
        for name, pipe in list_result_pipelines.items():
             model = pipe["best_model"]

        # check if model has feature_importances_
             if hasattr(model, "feature_importances_"):
                plt.figure(figsize=(12, 6))
                plt.plot(model.feature_importances_)
                plt.xlabel("Feature index")
                plt.ylabel("Importance")
                plt.title(f"Feature Importance - {name}")
                plt.tight_layout()
                plt.savefig(f"plots/feature_importance_{name}.png")
                plt.close()

    # --------- 2) CONFUSION MATRIX PLOTS ----------
        for name, res in results.items():
            cm = res["confusion_matrix"]

            plt.figure(figsize=(6, 5))
            plt.imshow(cm)
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()

            for i in range(cm.shape[0]):
               for j in range(cm.shape[1]):
                   plt.text(j, i, cm[i, j], ha="center", va="center")

            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix_{name}.png")
            plt.close()

    # --------- 3) SAVE METRICS TO FILE ----------
        if not os.path.exists("results"):
            os.makedirs("results")

        with open("results/metrics.txt", "w") as f:
            for name, res in results.items():
                f.write(f"Model: {name}\n")
                f.write(f"Accuracy: {res['accuracy']}\n")
                f.write(f"Precision: {res['precision']}\n")
                f.write(f"Recall: {res['recall']}\n")
                f.write(f"F1-score: {res['f1']}\n")
                f.write("\n")
                f.write(res["classification_report"])
                f.write("\n" + "-"*50 + "\n")

    
