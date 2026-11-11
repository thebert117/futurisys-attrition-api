-- Requêtes d'interrogation de la base futurisys_attrition
-- Usage : psql -U postgres -h localhost -d futurisys_attrition -f db/requetes.sql

-- 1. Volume de chaque table
SELECT 'employees' AS table_name, COUNT(*) AS nb_lignes FROM employees
UNION ALL
SELECT 'prediction_logs', COUNT(*) FROM prediction_logs;

-- 2. Taux d'attrition réel dans le dataset d'origine
SELECT target_attrition,
       COUNT(*) AS effectif,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pourcentage
FROM employees
GROUP BY target_attrition
ORDER BY target_attrition;

-- 3. Taux de départ par département
SELECT departement,
       COUNT(*) AS effectif,
       ROUND(100.0 * AVG(target_attrition), 1) AS taux_depart_pct
FROM employees
GROUP BY departement
ORDER BY taux_depart_pct DESC;

-- 4. Les 10 dernières prédictions enregistrées (entrée clé + sortie du modèle)
SELECT id, created_at, age, departement, heure_supplementaires,
       risque_depart, probabilite_depart
FROM prediction_logs
ORDER BY created_at DESC
LIMIT 10;

-- 5. Répartition des prédictions : à risque ou non, probabilité moyenne
SELECT risque_depart,
       COUNT(*) AS nb_predictions,
       ROUND(AVG(probabilite_depart)::numeric, 3) AS proba_moyenne
FROM prediction_logs
GROUP BY risque_depart;

-- 6. Volume de prédictions par jour (base d'un suivi de dérive)
SELECT DATE(created_at) AS jour, COUNT(*) AS nb_predictions
FROM prediction_logs
GROUP BY jour
ORDER BY jour;