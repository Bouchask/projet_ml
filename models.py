import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier, 
    VotingClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from data_preprocissing import Data

class Models:
    def __init__(self, url: str):
        self.obj_data = Data(url)
        self.data = self.obj_data.upload_data()
        
        # Standardisation
        data = self.obj_data.standard_data(self.data)
        data = pd.DataFrame(data)
        
        # --- CORRECTION ICI ---
        # On doit supprimer la Target (Traffic Situation) MAIS AUSSI
        # les colonnes non numériques "Date" et "Time" qui provoquent l'erreur.
        columns_to_drop = ["Traffic Situation", "Date", "Time"]
        
        # On vérifie quelles colonnes existent vraiment pour éviter une erreur si l'une manque
        existing_cols_to_drop = [c for c in columns_to_drop if c in data.columns]
        
        if "Traffic Situation" in data.columns:
            self.y = data["Traffic Situation"] # Garde le Y complet pour plus tard si besoin
            data_x = data.drop(labels=existing_cols_to_drop, axis=1)
        else:
            raise ValueError("La colonne 'Traffic Situation' est introuvable.")

        # Split des données
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.obj_data.split_data(data_x, self.y)
        
        self.best_model = None

    def creation_pipeline_classification(self):
        # 1. Initialisation des modèles
        clf_lr   = LogisticRegression(max_iter=1000, random_state=42)
        clf_rf   = RandomForestClassifier(random_state=42)
        clf_svc  = SVC(probability=True, random_state=42)
        clf_gb   = GradientBoostingClassifier(random_state=42)
        clf_mlp  = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)
        clf_ada  = AdaBoostClassifier(algorithm='SAMME', random_state=42)
        clf_et   = ExtraTreesClassifier(random_state=42)
        clf_hgb  = HistGradientBoostingClassifier(random_state=42)

        # 2. Ensemble
        estimators_list = [
            ('lr', clf_lr),
            ('rf', clf_rf),
            ('svc', clf_svc),
            ('gb', clf_gb),
            ('mlp', clf_mlp),
            ('ada', clf_ada),
            ('et', clf_et),
            ('hgb', clf_hgb)
        ]
        
        voting_clf = VotingClassifier(estimators=estimators_list, voting='soft')

        # 3. Pipeline
        main_pipeline = Pipeline([
            ("voting", voting_clf)
        ])

        # 4. Param Grid
        param_grid = {
            'voting__lr__C': [1, 10],
            'voting__rf__n_estimators': [50, 100],
            'voting__svc__C': [1, 10],
            'voting__gb__learning_rate': [0.1],
            'voting__mlp__activation': ['relu'],
            'voting__mlp__alpha': [0.001],
        }

        # 5. Validation Croisée avec PredefinedSplit
        X_combined = np.vstack((self.X_train, self.X_val))
        y_combined = np.hstack((self.y_train, self.y_val))

        split_index = [-1] * len(self.X_train) + [0] * len(self.X_val)
        ps = PredefinedSplit(test_fold=split_index)

        print("Lancement de l'optimisation (GridSearch)...")
        grid = GridSearchCV(
            main_pipeline, 
            param_grid, 
            cv=ps, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_combined, y_combined)

        print(f"Meilleurs paramètres : {grid.best_params_}")
        print(f"Meilleur score val : {grid.best_score_:.4f}")

        self.best_model = grid.best_estimator_
        return self.best_model

    def save_model(self, filename="best_voting_model.pkl"):
        if self.best_model:
            joblib.dump(self.best_model, filename)
            print(f"Modèle sauvegardé : {filename}")
        else:
            print("Aucun modèle entraîné.")