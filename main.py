import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importation de votre classe depuis le fichier models.py
from models import Models

if __name__ == "__main__":
    # 1. Configuration du chemin du fichier CSV
    # Assurez-vous que ce fichier existe dans votre dossier
    csv_path = "/home/yahya-bouchak/projet_ml/data/data_model.csv"

    print(f"--- Démarrage du projet avec : {csv_path} ---")

    try:
        # 2. Initialisation : Chargement et Préparation des données
        # Cela appelle __init__ de votre classe Models
        app = Models(csv_path)
        
        print(f"Données chargées avec succès.")
        print(f"Taille Train : {app.X_train.shape[0]} lignes")
        print(f"Taille Val   : {app.X_val.shape[0]} lignes")
        print(f"Taille Test  : {app.X_test.shape[0]} lignes")

        # 3. Entraînement et Recherche des Hyperparamètres
        print("\n--- Lancement du GridSearch sur le VotingClassifier ---")
        print("Veuillez patienter, cela peut prendre du temps selon votre CPU...")
        
        # Cette fonction retourne le meilleur modèle trouvé (best_estimator_)
        best_model = app.creation_pipeline_classification()

        # 4. Prédiction sur les données de TEST (jamais vues par le modèle)
        print("\n--- Évaluation finale sur le Test Set ---")
        y_pred = best_model.predict(app.X_test)

        # 5. Calcul des métriques
        accuracy = accuracy_score(app.y_test, y_pred)
        print(f"Précision (Accuracy) : {accuracy:.2%}")
        
        print("\nRapport de Classification détaillé :")
        print(classification_report(app.y_test, y_pred))

        # 6. Sauvegarde du Modèle
        app.save_model("traffic_model_final.pkl")

        # 7. Visualisation de la Matrice de Confusion
        # C'est important pour voir quelles classes sont confondues par le voting
        cm = confusion_matrix(app.y_test, y_pred)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de Confusion (Voting Classifier)\nAccuracy: {accuracy:.2%}')
        plt.ylabel('Vrai Label')
        plt.xlabel('Label Prédit')
        plt.show()

    except FileNotFoundError:
        print(f"ERREUR CRITIQUE : Le fichier '{csv_path}' est introuvable.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")