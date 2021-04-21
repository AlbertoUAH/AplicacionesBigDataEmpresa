#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_07
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Tuneo xgboost
# Comentarios: 
#-------------------

#--- Cargo librerías
suppressPackageStartupMessages({
  library(dplyr)                # Manipulacion de datos 
  library(data.table)           # Leer y procesar ultra-rapido
  library(ggplot2)              # La librería grafica
  library(inspectdf)            # EDAs automaticos
  library(ranger)               # Fast randomForest
  library(forcats)              # Tratar variables categoricas
  library(stringi)              # Tratamiento cadenas caracteres
  library(tictoc)               # Calcular tiempos
  library(embed)                # Creacion de embeddings
  library(doParallel)           # Paralelizacion de funciones
  library(missRanger)           # Tratamiento de valores missing (mediante random forest)
  library(mice)                 # Tratamiento de valores missing (mediante regresion logistica)
  library(lubridate)            # Tratamiento de fechas
  library(gmt)                  # Calculo distancia entre dos coordenadas geograficas
  library(dataPreparation)      # Preparacion de datasets
  library(xgboost)              # XGBoost
  library(ggrepel)              # Añadir etiquetas (texto) a ggplot
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leemos el fichero con los datos completos imputados
datcompleto_imp <- fread("./data/datcompleto_imp_ap_15.csv" )
names(datcompleto_imp)[14] <- "fe_dr_year_cyear_diff"
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE )
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Para contrastar los resultados del mejor modelo hasta el momento
mejor_resultado  <- fread("./submissions/19_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_y_permit_public_meeting.csv")

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto_imp$id == 50785)

formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

#-- Mejor accuracy hasta el momento: 0.8251

#-- Probamos un xgboost por defecto (sin tunear)
vector_status_group <- ifelse(vector_status_group == "functional", 0
                               , ifelse(vector_status_group == "functional needs repair", 1
                                        , 2))
xgb.train <- xgb.DMatrix(data=as.matrix(train), label=vector_status_group)

params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree = 0.3
)

#-- Modelo 1: parametrizando el numero de iteraciones
nrounds     <- c(400, 500, 600, 700)
lista_error <- c()
for(nround in nrounds) {
  
  my_model <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = nround)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/rounds/xgboost_with_",nround,"_iters.csv")
  )

  lista_error <- c(lista_error, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
}
rm(my_model); rm(xgb_pred); rm(nround)
prediction_errors_dt <- data.table(ntrees = rep(nrounds, 2), 
                                   Accuracy = c(0.7405, 0.7701, 0.7944,0.8157, 0.8150, 0.8160, 0.8138, 0.8147),
                                   Tipo =c(rep("Train (1 - mlogloss)", 4), rep("Submission", 4)))

ggplot(prediction_errors_dt, aes(x = ntrees, y = Accuracy, colour = Tipo)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt[prediction_errors_dt$ntrees == 500, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) + ggtitle("Accuracy del modelo en funcion de nrounds")
ggsave("./charts/xgboost_nrounds.png")

# 400 iter: 0.8150; 500 iter: 0.8160; 600 iter: 0.8138; 700 iter: 0.8147

#-- Hasta ahora, hemos podido comprobar que con 500 iteraciones se obtiene el valor maximo de accuracy (0.8160)
#   ¿Y si tuneamos el resto de parametros?
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(200, 150, 80, 50),
                           eta = c(0.3, 0.4, 0.5)
)
lista_error_2 <- c()
for(i in seq(1:nrow(search_grid))) {
  my_list <- list(objective = "multi:softmax",
                  num_class = 3,
                  colsample_bytree = search_grid[i, "colsample_bytree"], 
                  max_depth = search_grid[i, "max_depth"],
                  eta = search_grid[i, "eta"])
  
  my_model <- fit_xgboost_model(my_list, xgb.train, xgb.train, nrounds = 500)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/tunning_maxdepth_eta/xgboost_with_",paste0(search_grid[i, "colsample_bytree"],
                                                                                                        "-", search_grid[i, "max_depth"],
                                                                                                        "-", search_grid[i, "eta"]),"_config.csv")
  )
  
  lista_error_2 <- c(lista_error_2, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
}
# 200 + 0.3: 0.8167
# 150 + 0.3: 0.8173
# 80 + 0.3 : 0.8157
# 50 + 0.3 : 0.8166
# 200 + 0.4: 0.8143
# 150 + 0.4: 0.8160
# 80  + 0.4: 0.8162
# 50  + 0.4: 0.8157
# 200 + 0.5: 0.8144
# 150 + 0.5: 0.8144
# 80 + 0.5 : 0.8138
# 50 + 0.5 : 0.8159

rm(my_model); rm(xgb_pred); rm(i); rm(my_list)
prediction_errors_dt_2 <- data.table(Accuracy = c(0.7980, 0.7997, 0.8032, 0.8160, 0.8152, 0.8108, 0.8154, 0.8143, 0.8143,
                                                0.8178, 0.8133, 0.8131, 0.8182, 0.8156, 0.8135, 0.8159, 0.8154, 0.8139,
                                                0.8166, 0.8157, 0.8159, 0.8157, 0.8162, 0.8138, 0.8173, 0.8160, 0.8144,
                                                0.8167, 0.8143, 0.8144),
                                   colsample_byntree = rep(0.3, 30), max_depth = c(rep(3, 3), rep(6, 3), 
                                                                                   rep(8, 3), rep(10, 3),
                                                                                   rep(15, 3), rep(20, 3),
                                                                                   rep(50, 3), rep(80, 3),
                                                                                   rep(150, 3), rep(200, 3)),
                                   eta = as.factor(rep(c(0.3, 0.4, 0.5), 10)))

ggplot(prediction_errors_dt_2, aes(x = factor(max_depth), y = Accuracy, group = eta, colour = eta)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_2[prediction_errors_dt_2$max_depth == 15, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo con nrounds = 500 + colsample_bytree = 0.3 (solo test)")
ggsave("./charts/xgboost_max_depth_ntrees.png")


search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15),
                           eta = c(0.2, 0.1, 0.05, 0.02, 0.01)
)
lista_error_3 <- c()
for(i in seq(1:nrow(search_grid))) {
  my_list <- list(objective = "multi:softmax",
                  num_class = 3,
                  colsample_bytree = search_grid[i, "colsample_bytree"], 
                  max_depth = search_grid[i, "max_depth"],
                  eta = search_grid[i, "eta"])
  
  my_model <- fit_xgboost_model(my_list, xgb.train, xgb.train, nrounds = 500)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/tunning_maxdepth_eta/new/xgboost_with_",paste0(search_grid[i, "colsample_bytree"],
                                                                                                        "-", search_grid[i, "max_depth"],
                                                                                                        "-", search_grid[i, "eta"]),"_config.csv")
  )
  
  lista_error_3 <- c(lista_error_3, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
}
rm(my_model); rm(xgb_pred); rm(i); rm(my_list)

prediction_errors_dt_3 <- data.table(Accuracy = c(0.982093, 0.959185, 0.920351, 0.855014, 0.778542,
                                                  0.8193, 0.8202, 0.8233, 0.8257, 0.8238),
                                     colsample_byntree = rep(0.3, 10), max_depth = c(rep(15, 10)),
                                     eta = rep(c(0.2, 0.1, 0.05, 0.02, 0.01), 2),
                                     Tipo = c(rep("Train (1 - mlogloss)", 5), rep("Test", 5)),
                                     col = rep("red", 10))

ggplot(prediction_errors_dt_3[prediction_errors_dt_3$Tipo == "Test", ],aes(x = factor(eta), y = Accuracy, group = col, col = col)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_3[prediction_errors_dt_3$eta <= 0.05 &
                                                   prediction_errors_dt_3$Tipo == "Test", ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) + theme(legend.position = "none") +
  ggtitle("Accuracy del modelo con nrounds = 500 + colsample_bytree = 0.3 + max_depth = 15 (solo test)")
ggsave("./charts/xgboost_eta_parameter.png")


# Modelo 4
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15),
                           eta = c(0.01, 0.02, 0.03, 0.04)
)

nrounds     <- c(500, 600, 700)
lista_error_4 <- c()
for(nround in nrounds) {
  for(i in seq(1:nrow(search_grid))) {
    my_list <- list(objective = "multi:softmax",
                    num_class = 3,
                    colsample_bytree = search_grid[i, "colsample_bytree"], 
                    max_depth = search_grid[i, "max_depth"],
                    eta = search_grid[i, "eta"])
    
    my_model <- fit_xgboost_model(my_list, xgb.train, xgb.train, nrounds = nround)
    xgb_pred <- make_predictions_xgboost(my_model, test)
    fwrite(xgb_pred, 
           file = paste0("./submissions/tunning_models/xgboost/eta_nrounds/xgboost_with_",nround,"_iters_eta_",my_list$eta,".csv")
    )
    
    lista_error_4 <- c(lista_error_4, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
  }
}
rm(my_model); rm(xgb_pred); rm(nround)

prediction_errors_dt_4 <- data.table(Accuracy = c(0.8238, 0.8257, 0.8240, 0.8228, 0.8247, 0.8260, 0.8230, 0.8228,
                                                  0.8250, 0.8253, 0.8225, 0.8225),
                                     nrounds = c(rep(500, 4), rep(600, 4), rep(700, 4)),
                                     eta = as.factor(rep(c(0.01, 0.02, 0.03, 0.04), 3)))

ggplot(prediction_errors_dt_4, aes(x = nrounds, y = Accuracy, colour = eta)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_4[prediction_errors_dt_4$eta == 0.02 &
                                                   prediction_errors_dt_4$nrounds == 600, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo con colsample_bytree = 0.3 + max_depth = 15 (solo test)")
ggsave("./charts/xgboost_nrounds_eta_parameter_tunned.png")

# Modelo 5
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           colsample_bylevel = c(0.9),
                           colsample_bynode = c(0.6, 0.5),
                           max_depth = c(15),
                           eta = c(0.02)
)

nround <- 600
lista_error_5 <- c()
for(i in seq(1:nrow(search_grid))) {
  my_list <- list(objective = "multi:softmax",
                  num_class = 3,
                  colsample_bytree = search_grid[i, "colsample_bytree"], 
                  colsample_bylevel = search_grid[i, "colsample_bylevel"],
                  colsample_bynode = search_grid[i, "colsample_bynode"],
                  max_depth = search_grid[i, "max_depth"],
                  eta = search_grid[i, "eta"])
  
  my_model <- fit_xgboost_model(my_list, xgb.train, xgb.train, nrounds = nround)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/colsample_bylevel/xgboost_with_",search_grid[i, "colsample_bynode"],"_colsample_bynode.csv")
  )
  
  lista_error_5 <- c(lista_error_5, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
}
rm(my_model); rm(xgb_pred); rm(nround)

prediction_errors_dt_5 <- data.table(Accuracy = c(0.8243, 0.8260, 0.8242, 0.8232, 0.8222),
                                     colsample_bytree = as.factor(c(0.2, 0.3, 0.4, 0.5, 0.6)))

ggplot(prediction_errors_dt_5, aes(x = colsample_bytree, y = Accuracy)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_5[prediction_errors_dt_5$colsample_bytree == 0.3, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo con nrounds = 600 + eta = 0.02 + max_depth = 15 (solo test)")
ggsave("./charts/xgboost_colsample_by_tree.png")

# Modelo 6
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15),
                           eta = c(0.02),
                           subsample  = c(0.9)
)

nround <- 600
lista_error_6 <- c()
for(i in seq(1:nrow(search_grid))) {
  my_list <- list(objective = "multi:softmax",
                  num_class = 3,
                  colsample_bytree = search_grid[i, "colsample_bytree"], 
                  max_depth = search_grid[i, "max_depth"],
                  eta = search_grid[i, "eta"],
                  subsample = search_grid[i, "subsample"]
                  )
  
  my_model <- fit_xgboost_model(my_list, xgb.train, xgb.train, nrounds = nround)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/subsample/xgboost_with_",search_grid[i, "subsample"],"_subsample_and_",search_grid[i, "colsample_bytree"],".csv")
  )
  
  lista_error_6 <- c(lista_error_6, 1 - tail(my_model$evaluation_log$val1_mlogloss, 1))
}
# subsample 0.5 and colsample 0.3: 0.8244
# subsample 0.6 and colsample 0.3: 0.8254
# subsample 0.7 and colsample 0.3: 0.8254
# subsample 0.8 and colsample 0.3: 0.8252
# subsample 0.9 and colsample 0.3: 0.8240

rm(my_model); rm(xgb_pred); rm(nround)

prediction_errors_dt_5 <- data.table(Accuracy = c(0.8244, 0.8254, 0.8254, 0.8252, 0.8240, 0.8260),
                                     subsample = as.factor(c(0.5, 0.6, 0.7, 0.8, 0.9, 1)))

ggplot(prediction_errors_dt_5, aes(x = subsample, y = Accuracy)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_5[prediction_errors_dt_5[, Accuracy] == 0.8260, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo con nrounds = 600 + eta = 0.02 + max_depth = 15 + colsample = 0.3 (solo test)")
ggsave("./charts/xgboost_subsample.png")


# colsample by level -> 0.9: 0.8265
# colsample by level -> 0.8: 0.8257
# colsample by level -> 0.7: 0.8260

prediction_errors_dt_5 <- data.table(Accuracy = c(0.8243, 0.8260, 0.8242, 0.8232, 0.8222),
                                     colsample_bytree = as.factor(c(0.2, 0.3, 0.4, 0.5, 0.6)))

ggplot(prediction_errors_dt_5, aes(x = colsample_bytree, y = Accuracy)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt_5[prediction_errors_dt_5$colsample_bytree == 0.3, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo con nrounds = 600 + eta = 0.02 + max_depth = 15 (solo test)")
ggsave("./charts/xgboost_colsample_by_tree.png")

# colsample by node -> 0.9: 0.8256
# colsample by node -> 0.8: 0.8244
# colsample by node -> 0.7: 0.8272
# colsample by node -> 0.6: 0.8261
# colsample by node -> 0.5: 0.8266


