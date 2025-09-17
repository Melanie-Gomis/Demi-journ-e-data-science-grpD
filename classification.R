library(dplyr)
library(caret)
library(tidyverse)
library(FactoMineR)
library(corrplot)
library(pROC)


# Importation et aperçu des données 
data = read.csv(file="farms_train.csv",header = TRUE,sep=",",dec = ".")
tibble(data)
glimpse(data)
summary(data)

#Standardisation des données 
stdize <- function(vecteur_init){
  if(is.numeric(vecteur_init)){
    (vecteur_init - min(vecteur_init, na.rm = TRUE)) / 
      (max(vecteur_init, na.rm = TRUE) - min(vecteur_init, na.rm = TRUE))
  } else {
    vecteur_init  
  }
}
farms_train_stdize <- data %>% 
  select(DIFF) %>% 
  bind_cols(
    data[2:7] %>% map_dfc(stdize)
  )
farms_train_stdize$DIFF =  factor(farms_train_stdize$DIFF, 
                                  levels = c(0,1), 
                                  labels = c("pas_saine", "saine"))
tibble(farms_train_stdize)

#BOxplots 
farms_train_stdize %>%
  pivot_longer(cols= -DIFF, names_to = "variable", values_to = "valeur" ) %>%
  ggplot(aes(x= DIFF,y=valeur),fill=target) +
  geom_boxplot() +
  facet_wrap(~variable, scales='free_y')+
  theme_minimal()


# Gestion des valeurs aberrantes *
detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))
}
outlier_mask <- apply(farms_train_stdize %>% select(-DIFF), 2, detect_outliers)
outlier_rows <- apply(outlier_mask, 1, any)
farms_train_clean = farms_train_stdize[!outlier_rows,]

nrow(farms_train_clean)
nrow(farms_train_stdize)

#Regression logistique 

#Normalisation 
preproc <- preProcess(farms_train_clean %>% select(-DIFF), method = c("center", "scale"))
df_scaled <- predict(preproc, farms_train_clean %>% select(-DIFF))
df_scaled$DIFF <- farms_train_clean$DIFF

tibble(df_scaled)

#Regression 
set.seed(123)

#controle de validation croisée 
ctrl <- trainControl(method = "cv", number = 50, 
                     classProbs = TRUE,       
                     summaryFunction = twoClassSummary) 

#entrainement du modèle 
modele = logit_model <- train(
  DIFF ~ ., 
  data = df_scaled, 
  method = "glm", 
  family = "binomial",
  metric = "ROC",          
  trControl = ctrl
)

tibble(modele)
summary(modele)
summary(logit_model$finalModel)

#visualisation 

probs <- predict(modele, type = "prob")[, "pas_saine"]
roc_obj <- roc(df_scaled$DIFF, probs, levels = c("saine", "pas_saine"))
plot(roc_obj, col="blue", main="ROC Curve")
auc(roc_obj)

#prediction 
pred_class = predict(modele,df_scaled)
pred_class

#matrice de confusion 
confusionMatrix(pred_class,df_scaled$DIFF)

exp(coef(logit_model$finalModel))

modele$results


#######TEST#######
farms_test = read.csv(file = "farms_test.csv",header=TRUE,sep=",",dec=".")
farms_test$DIFF <- factor(farms_test$DIFF, levels = c(0,1), labels = c("saine", "pas_saine"))
df_test_scaled <- predict(preproc, test %>% select(-DIFF))



