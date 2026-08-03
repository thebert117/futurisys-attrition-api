"""
Point d'entrée pour lancer l'entraînement du modèle.

Ce fichier existe uniquement pour que ml/train_model.py soit toujours importé
comme un module normal (jamais exécuté directement comme __main__). C'est
nécessaire pour que le FunctionTransformer utilisé dans le pipeline (qui
référence add_engineered_features) puisse être rechargé correctement par
joblib, y compris depuis un autre contexte (API, tests, etc.).

Usage : python train.py
"""

from ml.train_model import main

if __name__ == "__main__":
    main()