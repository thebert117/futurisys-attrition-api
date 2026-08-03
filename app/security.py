"""
Authentification de l'API par clé API (header X-API-Key).

Choix volontairement simple pour ce POC : une seule clé secrète partagée,
plutôt qu'un système de comptes utilisateurs. Adapté à un usage où l'API
est consommée par un service interne (SIRH, dashboard RH), pas par des
utilisateurs individuels ayant besoin de droits différenciés.
"""

import os

from dotenv import load_dotenv
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

load_dotenv()

API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)) -> None:
    """Dépendance FastAPI : vérifie que la clé API fournie est valide.

    Levée d'une 401 si absente, 403 si incorrecte — distinction utile pour
    le débogage côté consommateur de l'API.
    """
    if not API_KEY:
        # Sécurité : si la clé n'est pas configurée côté serveur, on refuse
        # tout accès plutôt que de désactiver silencieusement la protection.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clé API non configurée côté serveur.",
        )
    if provided_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante. Fournissez un en-tête X-API-Key.",
        )
    if provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide.",
        )