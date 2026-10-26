# Classification des races de chiens

Le notebook `dog_breed_classifier.ipynb` utilise les images de `train/`, leurs races
dans `labels.csv`, les images de `test/` et le format de `sample_submission.csv`.
Il associe un extracteur visuel EfficientNet B0 préentraîné à un classifieur
multiclasse entraîné sur ces données. Il fonctionne sur CPU et utilise CUDA si disponible.

## Exécuter

1. Placer les quatre éléments de données à côté du notebook.
2. Ouvrir le notebook avec Jupyter dans l'environnement Python du projet.
3. Au besoin, installer les dépendances avec `%pip install -r requirements.txt`,
   puis redémarrer le noyau.
4. Exécuter les cellules dans l'ordre, ou utiliser « Restart Kernel and Run All Cells ».

Le premier passage télécharge environ 21 Mo de poids depuis le serveur officiel
PyTorch. Les caractéristiques visuelles sont calculées une seule fois et mises en
cache. Leur extraction peut prendre plusieurs minutes sur CPU.

Pour exécuter tout le notebook depuis un terminal :

```powershell
python scripts/execute_notebook.py dog_breed_classifier.ipynb --in-process
```

Ce lanceur conserve les sorties et sauvegarde le notebook après chaque cellule,
y compris en cas d'erreur. Il fonctionne sans ouvrir de serveur Jupyter.

## Résultats

- `submission.csv` : une ligne par image de test et une probabilité pour chacune
  des races, dans l'ordre exact du fichier exemple.
- `artifacts/dog_breed_model.pth` : modèle complet, ordre des races et paramètres.
- `artifacts/validation_metrics.json` : accuracy top 1/top 5 et log loss.
- `artifacts/validation_predictions.csv` : erreurs et prédictions de validation.
- `artifacts/features/` : caractéristiques en cache pour les prochaines exécutions.

La validation utilise une séparation stratifiée de 20 % des images étiquetées.
Elle sert aussi à choisir la régularisation ; c'est une validation interne.
Le classifieur final est ensuite réentraîné sur toutes les images étiquetées.
Les images de test n'ont pas de races connues : on ne peut pas mesurer leur accuracy.

Sur cette séparation de 2 045 images de validation (graine 42), le modèle retenu
avec `C=0.1` obtient **88,41 % d'accuracy top 1**, **99,17 % de top 5** et une
**log loss de 0,3893**. Ces mesures sont conservées dans le notebook exécuté et
dans le fichier de métriques.

Pour prédire une nouvelle photo, exécuter les imports puis la section 8 du notebook,
et appeler `predict_breed("chemin/vers/mon_chien.jpg")`.
Le modèle rechargé n'a pas besoin de télécharger à nouveau les poids.

Les données et les fichiers générés sont ignorés par Git pour éviter d'y ajouter
les images, les caches et les modèles volumineux.
