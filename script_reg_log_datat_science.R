library(tidyverse)
farms_train <- read_csv("data/farms_train.csv")
farms_test <- read_csv("data/farms_test.csv")


farms_train <- farms_train %>% 
  mutate(DIFF=factor(DIFF,levels = c(0,1),labels=c("Défaillante","Saine"))) 


farms_train %>% 
  pivot_longer(cols=-DIFF) %>% 
  mutate(DIFF=factor(DIFF,levels = c(0,1),labels=c("Défaillante","Saine"))) %>% 
  ggplot(aes(x=DIFF,y= value))+
  geom_boxplot()+
  facet_wrap("name")

farms_train %>% 
  pivot_longer(cols=-DIFF) %>% 
  mutate(DIFF=factor(DIFF,levels = c(0,1),labels=c("Défaillante","Saine"))) %>% 
  ggplot(aes(x=DIFF,y= value))+
  geom_boxplot()+
  facet_wrap("name",scales="free_y")

pairs(farms_train[2:7], 
      main = "Données exploitations agricoles - farm train", 
      pch = 21, 
      bg = c("red", "green3")[farms_train$DIFF], 
      lower.panel=NULL, 
      font.labels=0.5, cex.labels=2) 

legend(0.08, 0.43, unique(farms_train$DIFF),  fill=c("red", "green3"))

# 1 : standardiser
stdize <- function(vecteur_init){
  map_dbl(vecteur_init,function(x){
    temp <- (x - min(vecteur_init))/(max(vecteur_init)-min(vecteur_init))
    return(temp)
  })  
}

farms_train_stdize <- farms_train %>% 
  select(DIFF) %>% 
  bind_cols(
    apply(farms_train[2:7],2, stdize)
  )

# 2 : Enlever les outliers
# Flags IQR par variable (1.5*IQR par défaut)
farms_train_stdize_flags_outliers <- farms_train_stdize %>%
  mutate(across(all_of(c("R2","R7","R8","R17","R22","R32")), ~{
    q <- quantile(., c(.25,.75), na.rm = TRUE)
    i <- diff(q)
    (. < q[1] - 1.5*i) | (. > q[2] + 1.5*i)
  }, .names = "out_{.col}_iqr"))

farms_train_stdize_flags_outliers <- farms_train_stdize_flags_outliers %>% 
  mutate(
    outliers=if_any(starts_with("out_"))
  )

farms_train_stdize_flags_outliers %>% 
  group_by(outliers) %>% 
  summarise(nb=n())

farms_train_stdize_flags_outliers %>% 
  group_by(outliers,DIFF) %>% 
  summarise(nb=n()) %>% 
  mutate(part=prop.table(nb))


farms_train_stdize_sans_outliers <- farms_train_stdize_flags_outliers %>% 
  filter(!outliers)

# 3 : Déterminer tous les modèles qu'on veut tester
# Lister les modeles possibles de regression logistiques ----
# Source chatGPT pour identifier l'ensemble des combinaisons possibles:
# prédicteurs :
preds <- c("R2","R7","R8","R17","R22","R32")

# --- Générer toutes les formules (tous sous-ensembles non vides) ---
all_forms <- unlist(
  lapply(seq_along(preds), function(k) {
    combn(preds, k, FUN = function(vars) {
      as.formula(paste("DIFF ~", paste(vars, collapse = " + ")))
    }, simplify = FALSE)
  }),
  recursive = FALSE
)

length(all_forms)      # 2^6 - 1 = 63 modèles
all_forms[1:5]         # aperçu

# 3 : Validation croisée
#Entrainemnt d'un modèle sur 90% des lignes de la table sans valeurs aberrantes ----
0.9*nrow(farms_train_stdize_sans_outliers)
# apprentissage et validation sur les lignes 1 à 312
# test sur les lignes 312 à 347

farms_train_validation_sans_outliers <- farms_train_stdize_sans_outliers[1:312,] 
farms_train_test_sans_outliers <- farms_train_stdize_sans_outliers[-c(1:312),] 

library(caret)

# fits <- map(all_forms, ~glm(formula = .,data = farms_train_stdize, family= "binomial"))
ctrl <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  # calcule ROC, Sens, Spec
  savePredictions = "final"
)

# 4 : Calcul de l'AUC
farms_train_validation_sans_outliers <- farms_train_stdize_sans_outliers[1:312,] 
farms_train_test_sans_outliers <- farms_train_stdize_sans_outliers[-c(1:312),] 


# --- Entraîner tous les sous-modèles (glm binomial) ---
fits_train_validation_sans_outliers <- map(all_forms, ~
                                             train(
                                               form = .x,
                                               data = farms_train_validation_sans_outliers,
                                               method = "glm",
                                               family = "binomial",
                                               metric = "ROC",
                                               trControl = ctrl
                                             )
)


# --- Récupérer la perf CV de chaque modèle ---
perf_train_validation_sans_outliers <- map2_dfr(fits_train_validation_sans_outliers, all_forms, ~{
  tibble(
    formula = deparse(.y),
    ROC  = .x$results$ROC,
    Sens = .x$results$Sens,
    Spec = .x$results$Spec
  )
}) %>% arrange(desc(ROC))

perf_train_validation_sans_outliers

prob_pos <- predict(fits_train_validation_sans_outliers[[1]], newdata = farms_train_test_sans_outliers, type = "prob")[["Saine"]]

# --- 3) ROC + AUC (+ IC)
roc_obj <- roc(response = farms_train_test_sans_outliers$DIFF,
               predictor = prob_pos,
               levels = c("Défaillante","Saine"))  

auc_val <- auc(roc_obj)
ci_val  <- ci.auc(roc_obj)

# --- 4a) Tracé base R (pROC)
plot(roc_obj, print.auc = TRUE, legacy.axes = TRUE)



pred_class <- predict(fits_train_validation_sans_outliers[[1]], newdata = farms_test, type = "raw")

farms_test %>% 
  mutate(DIFF_pred=pred_class) %>% 
  group_by(DIFF_pred) %>% 
  summarise(nb=n())  

farms_test %>% 
  mutate(DIFF_pred=pred_class) %>% 
  ggplot(aes(y=DIFF_pred))+
  geom_bar()
