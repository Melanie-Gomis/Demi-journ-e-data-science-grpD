# Demi-journee-data-science-grpD

Nous avons participer à un défis Kaggle dans le cadre de notre 1ere année de master.
Le but était de prédire au mieux la classe de la variable DIFF.

Pour cela, nous avons d'abord fait une exploration des données pour les visualiser et décider de la stratégie à adopter.

Nous nous sommes ensuite divisé le travail pour maximiser notre temps.

Ainsi une rapide exploration des données a été réalisé par Arnaud M (1ere partie de script_reg_log_datat_science.R ), et une ACP par Maximiliano R (code commenté disponible dans ACP_demi_journee.Rmd).

Pour pouvoir prédire la classe de DIFF, nous avons réaliser plusieurs modèles de classification :
- KNN par Enzo N (disponible dans KNN_train.R)
- Régression logistique (disponible dans script_reg_log_datat_science.R)
- SVM à marge souple, avec toutes les valeurs par Mélanie G (disponilble dans SVM.Rmd) et sans les valeur par Tojolalaina R ()
- SVM à noyau par Mélanie G (disponible dans "SVM à noyau.Rmd")

Ainsi, nous avons entraînner et évaluer nos données sur 90% du jeu de donnée "farms_train.csv". Et nous avons comparé tout nos modèle sur les 10% restant. 
Grâce à cette démarche, nous avons conclue que le meilleur modèle était celui de SVM à marge souple comprennant toutes les données.


Malheuresement, la limite de temps ne nous a pas permis de faire tout ce que nous aurions voulue faire. 
Voici une liste non-exhaustive de ce que nous aurions fait si le temps nous l'aurait permis : 
- Analyser mieux les valeurs abérrantes
- Tester un modèle basé sur les arbres de décision.
- Faire les modèles de régression logistique et SVM directement sur les dimensions de l'ACP.
- Explorer plus profondément les données.

