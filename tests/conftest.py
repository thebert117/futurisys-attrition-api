"""
Configuration partagée des tests.

Force une clé API dédiée aux tests (différente de la vraie clé de production
dans .env), définie AVANT que l'application soit importée par les modules
de test, pour que app/security.py la lise dès son chargement.
"""

import os

os.environ["API_KEY"] = "test-secret-key-12345"