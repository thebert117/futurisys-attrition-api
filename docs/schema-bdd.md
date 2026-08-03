# Schéma de la base de données

Ce diagramme est écrit en [Mermaid](https://mermaid.js.org/) — GitHub le restitue automatiquement sous forme de diagramme visuel lors de l'affichage de ce fichier sur le dépôt.

```mermaid
erDiagram
    EMPLOYEES {
        int id_employee PK
        int age
        string genre
        int revenu_mensuel
        string statut_marital
        string departement
        string poste
        int nombre_experiences_precedentes
        int annee_experience_totale
        int annees_dans_l_entreprise
        int annees_dans_le_poste_actuel
        int satisfaction_employee_environnement
        int note_evaluation_precedente
        int niveau_hierarchique_poste
        int satisfaction_employee_nature_travail
        int satisfaction_employee_equipe
        int satisfaction_employee_equilibre_pro_perso
        int note_evaluation_actuelle
        string heure_supplementaires
        string augmentation_salaire_precedente
        int nombre_participation_pee
        int nb_formations_suivies
        int distance_domicile_travail
        int niveau_education
        string domaine_etude
        string frequence_deplacement
        int annees_depuis_la_derniere_promotion
        int annees_sous_responsable_actuel
        string a_quitte_l_entreprise
        int target_attrition
    }

    PREDICTION_LOGS {
        int id PK
        datetime created_at
        int age
        string genre
        int revenu_mensuel
        string statut_marital
        string departement
        string poste
        int nombre_experiences_precedentes
        int annee_experience_totale
        int annees_dans_l_entreprise
        int annees_dans_le_poste_actuel
        int satisfaction_employee_environnement
        int note_evaluation_precedente
        int niveau_hierarchique_poste
        int satisfaction_employee_nature_travail
        int satisfaction_employee_equipe
        int satisfaction_employee_equilibre_pro_perso
        int note_evaluation_actuelle
        string heure_supplementaires
        string augmentation_salaire_precedente
        int nombre_participation_pee
        int nb_formations_suivies
        int distance_domicile_travail
        int niveau_education
        string domaine_etude
        string frequence_deplacement
        int annees_depuis_la_derniere_promotion
        int annees_sous_responsable_actuel
        boolean risque_depart
        float probabilite_depart
    }
```

## Notes de conception

**Pas de clé étrangère entre les deux tables.** `EMPLOYEES` est une donnée de référence (le dataset RH nettoyé du projet 4). `PREDICTION_LOGS` est un journal d'utilisation de l'API : une prédiction peut concerner un profil hypothétique qui n'existe pas dans `EMPLOYEES` (simulation, nouveau candidat). Les relier par clé étrangère contraindrait artificiellement un usage qui n'a pas vocation à être limité aux salariés déjà connus.

**Schéma volontairement dénormalisé** (pas de 3NF stricte, pas de tables de dimension séparées pour `departement`/`poste`). Justification détaillée dans le [README](../README.md#base-de-données) : volume très faible (1470 lignes, 3 départements, 9 postes), validation déjà assurée en amont par Pydantic, `PREDICTION_LOGS` suit le pattern standard d'un journal d'événements (souvent dénormalisé même dans des systèmes matures).

**Contraintes** : toutes les colonnes sont `NOT NULL` sauf `a_quitte_l_entreprise` et `target_attrition` dans `EMPLOYEES` (colonnes exclues des features du modèle mais conservées pour la traçabilité historique). `id_employee` (EMPLOYEES) et `id` (PREDICTION_LOGS) sont les clés primaires, `created_at` est généré automatiquement à l'insertion.