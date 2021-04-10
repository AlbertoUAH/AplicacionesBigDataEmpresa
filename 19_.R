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

cl <- makeCluster(detectCores())
registerDoParallel(cl)

#-- Mejor accuracy hasta el momento: 0.8251

#-- Probamos un xgboost por defecto (sin tunear)
vector_status_group <- ifelse(vector_status_group == "functional", 0
                               , ifelse(vector_status_group == "functional needs repair", 1
                                        , 2))
xgb.train = xgb.DMatrix(data=as.matrix(train), label=vector_status_group)

params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree = 0.3
)

set.seed(1234)
my_model <- xgb.train(
  data   = xgb.train,
  params = params,
  watchlist=list(val1=xgb.train),
  early_stopping_rounds = 20,
  verbose = 1,
  nrounds= 500,
  nthread=4
)

xgb_pred <- predict(my_model,as.matrix(test),reshape=T)
xgb_pred <- ifelse(xgb_pred == 0, "functional", ifelse(xgb_pred == 1, "functional needs repair", "non functional"))

xgb_pred <- data.table(id = test$id, status_group = xgb_pred)

fwrite(xgb_pred, file = "./submissions/tunning_models/xgboost/26_xgboost_sin_tunear.csv")
# 0.8160 Sin tunear (ultimo mlogloss: 0.229939)

#-- Modelo 2: aumentando el numero de arboles a 600
# ¿Y si aumentamos el valor de eta? Por defecto esta a 0.3, podriamos aumentarlo antes que aumentar el numero de iteraciones
params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree = 0.3
)

my_model_2 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 600)
xgb_pred_2 <- make_predictions_xgboost(my_model_2, test)
fwrite(xgb_pred_2, file = "./submissions/tunning_models/xgboost/26_xgboost_600_rounds.csv")
# 0.8138 con 600 iteraciones (ultimo mlogloss: 0.205605)

# Aumentamos a 550?
params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree = 0.3
)

my_model_3 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 550)
xgb_pred_3 <- make_predictions_xgboost(my_model_3, test)
fwrite(xgb_pred_3, file = "./submissions/tunning_models/xgboost/26_xgboost_550_rounds.csv")
# 0.8146 con 550 iteraciones (ultimo mlogloss: 0.217445)

# Aumentar el numero de iteraciones no parece ayudar ¿Y si reducimos el numero de iteraciones?
params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree = 0.3
)
my_model_4 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 400)
xgb_pred_4 <- make_predictions_xgboost(my_model_4, test)
fwrite(xgb_pred_4, file = "./submissions/tunning_models/xgboost/26_xgboost_400_rounds.csv")
# 0.8150 con 400 iteraciones (ultimo mlogloss: 0.259503)

#-- Hasta ahora, hemos podido comprobar que con 500 iteraciones se obtiene el valor maximo de accuracy (0.8160)
#   ¿Y si tuneamos el resto de parametros?
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(20, 15, 10, 8, 6, 3),
                           eta = c(0.3, 0.4, 0.5)
)

error_final <- c()

for(fila in 1:nrow(search_grid)) {
  params = list(
      objective = "multi:softmax",
      num_class = 3,
      colsample_bytree = search_grid[fila, "colsample_bytree"],
      max_depth        = search_grid[fila, "max_depth"],
      eta              = search_grid[fila, "eta"]
    )
    
  my_model <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 700)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/tuneo/colsample_",search_grid[fila, "colsample_bytree"],
                       "_max_depth_",search_grid[fila, "max_depth"],"_eta_",search_grid[fila, "eta"],"_700_iter.csv")
         )
  
  error_final <- c(error_final, tail(my_model$evaluation_log$val1_mlogloss, 1))
}
cbind(search_grid, 1-error_final)

# Analicemos los resultados (guardado en Excel): la clave reside en max_depth 10-15. Probamos a bajar eta a 0.1 y 0.2
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15, 10),
                           eta = c(0.1, 0.2)
)

# Disminuyendo eta parece que mejora...
# Probamos a reducirlo mas...
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15),
                           eta = c(0.01, 0.02, 0.05, 0.08)
)

# Empleando 0.02 obtenemos buenos resultados ¿Y si aumentamos el numero de iteraciones?
search_grid <- expand.grid(colsample_bytree = c(0.3),
                           max_depth = c(15),
                           eta = c(0.02)
)


stopCluster(cl)



