# === 1. Librairies ===
library(class)
library(caret)
library(dplyr)
library(ggplot2)
library(MLmetrics)

# === 2. Charger dataset ===
df <- read.csv("C:/Users/nguye/Desktop/Enzo/Cours/M1 MIASHS/Semestre 1/Classification supervisée et non supervisée/Demi-journée data science/Demi-journee-data-science-grpD/farms_train.csv")
df$DIFF <- factor(df$DIFF)

# === 3. Visualisation simple (2 features) ===
# exemple avec R2 et R7
ggplot(df, aes(x = R2, y = R7, color = DIFF)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(title = "Répartition des fermes selon R2 et R7",
       x = "R2", y = "R7")

# === 4. Préparer les données ===
X <- scale(df[, -1])  # toutes les colonnes sauf DIFF
y <- df$DIFF

# === 5. Validation croisée pour trouver le k optimal ===
set.seed(42)

# Paramètres de validation croisée
k_folds <- 5
taille.appr <- 50
cv_f1_scores <- matrix(0, nrow = taille.appr, ncol = k_folds)

# Créer les plis pour la validation croisée
folds <- createFolds(y, k = k_folds, list = TRUE, returnTrain = FALSE)

# Boucle de validation croisée
for (k in 1:taille.appr) {
  for (fold in 1:k_folds) {
    # Indices de test pour ce pli
    test_indices <- folds[[fold]]
    train_indices <- setdiff(1:nrow(X), test_indices)
    
    # Données d'entraînement et de test pour ce pli
    X_train_fold <- X[train_indices, ]
    y_train_fold <- y[train_indices]
    X_test_fold <- X[test_indices, ]
    y_test_fold <- y[test_indices]
    
    # Prédiction KNN
    pred_fold <- knn(train = X_train_fold, test = X_test_fold, 
                     cl = y_train_fold, k = k)
    
    # Calculer F1-score pour ce pli
    cv_f1_scores[k, fold] <- F1_Score(y_pred = pred_fold, y_true = y_test_fold, 
                                      positive = levels(df$DIFF)[1])
  }
}

# Calculer la moyenne et l'écart-type des F1-scores pour chaque k
mean_f1_scores <- rowMeans(cv_f1_scores)
sd_f1_scores <- apply(cv_f1_scores, 1, sd)

# === 6. Visualisation des résultats de validation croisée ===
cv_results <- data.frame(
  k = 1:taille.appr,
  F1_mean = mean_f1_scores,
  F1_sd = sd_f1_scores,
  F1_lower = mean_f1_scores - sd_f1_scores,
  F1_upper = mean_f1_scores + sd_f1_scores
)

ggplot(cv_results, aes(x = k, y = F1_mean)) +
  geom_ribbon(aes(ymin = F1_lower, ymax = F1_upper), alpha = 0.3, fill = "lightblue") +
  geom_line(color = "dodgerblue", size = 1) +
  geom_point(color = "darkorange", size = 2) +
  theme_minimal() +
  labs(title = "F1-score moyen avec validation croisée (5 plis)",
       subtitle = "La zone colorée représente ± 1 écart-type",
       x = "k (nombre de voisins)", 
       y = "F1-score moyen")

# === 7. k optimal ===
best_k <- which.max(mean_f1_scores)
cat("Meilleur k =", best_k, "avec F1 moyen =", round(max(mean_f1_scores), 4), 
    "± écart-type =", round(sd_f1_scores[best_k], 4), "\n")

# === 8. Évaluation finale avec split train/test ===
set.seed(42)
Index <- createDataPartition(df$DIFF, p = 0.7, list = FALSE)

X_train <- X[1:360, ]
y_train <- y[1:360]
X_test <- X[361:401, ]
y_test <- y[361:401]

# Prédiction avec le k optimal
y_pred <- knn(train = X_train, test = X_test, cl = y_train, k = best_k, prob = TRUE)

# === 9. Matrice de confusion avec labels TP, TN, FP, FN ===
conf_mat <- table(Pred = y_pred, Real = y_test)
print("Matrice de confusion brute:")
print(conf_mat)

# Supposons que la classe positive est le niveau 1 de DIFF
positive_class <- levels(df$DIFF)[1]
negative_class <- levels(df$DIFF)[2]

# Créer une matrice de confusion avec labels explicites
conf_mat_labeled <- matrix(0, nrow = 2, ncol = 2)
rownames(conf_mat_labeled) <- c(paste("Pred", positive_class), paste("Pred", negative_class))
colnames(conf_mat_labeled) <- c(paste("Real", positive_class), paste("Real", negative_class))

# Remplir la matrice
if (positive_class %in% rownames(conf_mat) && positive_class %in% colnames(conf_mat)) {
  TP <- conf_mat[positive_class, positive_class]
} else { TP <- 0 }

if (negative_class %in% rownames(conf_mat) && negative_class %in% colnames(conf_mat)) {
  TN <- conf_mat[negative_class, negative_class]
} else { TN <- 0 }

if (positive_class %in% rownames(conf_mat) && negative_class %in% colnames(conf_mat)) {
  FP <- conf_mat[positive_class, negative_class]
} else { FP <- 0 }

if (negative_class %in% rownames(conf_mat) && positive_class %in% colnames(conf_mat)) {
  FN <- conf_mat[negative_class, positive_class]
} else { FN <- 0 }

# === 10. Créer un data.frame pour la visualisation avec labels ===
conf_labels <- data.frame(
  Pred = c(paste("Pred", positive_class), paste("Pred", positive_class), 
           paste("Pred", negative_class), paste("Pred", negative_class)),
  Real = c(paste("Real", positive_class), paste("Real", negative_class),
           paste("Real", positive_class), paste("Real", negative_class)),
  Freq = c(TP, FP, FN, TN),
  Label = c(paste("TP\n", TP), paste("FP\n", FP), 
            paste("FN\n", FN), paste("TN\n", TN))
)

# === 11. Heatmap améliorée avec ggplot ===
ggplot(conf_labels, aes(x = Real, y = Pred, fill = Freq)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = Label), color = "white", size = 6) +
  scale_fill_gradient(low = "lightcoral", high = "darkred", name = "Fréquence") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.title = element_text(size = 12, face = "bold")
  ) +
  labs(title = paste("Matrice de confusion (k =", best_k, ")"),
       subtitle = "TP: Vrais Positifs, TN: Vrais Négatifs, FP: Faux Positifs, FN: Faux Négatifs",
       x = "Valeur réelle", y = "Prédiction")

# === 12. Métriques de performance ===
cat("\n=== MÉTRIQUES DE PERFORMANCE ===\n")
cat("TP (Vrais Positifs):", TP, "\n")
cat("TN (Vrais Négatifs):", TN, "\n")
cat("FP (Faux Positifs):", FP, "\n")
cat("FN (Faux Négatifs):", FN, "\n")

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_final <- 2 * (precision * recall) / (precision + recall)

cat("Exactitude (Accuracy):", round(accuracy, 4), "\n")
cat("Précision:", round(precision, 4), "\n")
cat("Rappel (Recall/Sensibilité):", round(recall, 4), "\n")
cat("F1-Score final:", round(f1_final, 4), "\n")

# === 13. Comparaison validation croisée vs test final ===
cat("\nComparaison:\n")
cat("F1-Score validation croisée:", round(max(mean_f1_scores), 4), "±", round(sd_f1_scores[best_k], 4), "\n")
cat("F1-Score test final:", round(f1_final, 4), "\n")


# Extraire la probabilité associée à la classe prédite
# prob attribué correspond à la proportion de votes pour la classe prédite
probs <- attr(y_pred_knn, "prob")

# On veut la probabilité pour la classe positive
# Si la prédiction est la classe négative, on inverse la probabilité
probs_pos <- ifelse(y_pred_knn == positive_class, probs, 1 - probs)

# Calcul de l'AUC
auc <- AUC(y_pred = probs_pos, y_true = ifelse(y_test == positive_class, 1, 0))
cat("AUC sur les 10% de test :", round(auc, 4), "\n")

# Créer un vecteur binaire pour la classe positive
y_test_bin <- ifelse(y_test == positive_class, 1, 0)

# Générer la courbe ROC
library(pROC)
# === Calcul ROC ===
roc_obj <- roc(y_test_bin, probs_pos)

# Extraire les points de la ROC
roc_df <- data.frame(
  TPR = rev(roc_obj$sensitivities),  # Recall
  FPR = rev(1 - roc_obj$specificities)
)

# Tracer avec ggplot2
ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "dodgerblue", size = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  geom_point(data = data.frame(FPR = 0, TPR = 0), aes(x = FPR, y = TPR), color = "red", size = 3) +
  theme_minimal(base_size = 14) +
  labs(
    title = paste("Courbe ROC - kNN (k =", best_k, ")"),
    subtitle = paste("AUC =", round(auc(roc_obj), 4)),
    x = "Taux de faux positifs (FPR)",
    y = "Taux de vrais positifs (TPR)"
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )