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
  num_class = 3
)

my_model <- xgb.train(
  data   = train,
  params = params,
  watchlist=list(val1=val),
  verbose = 1,
  nrounds= 500,
  nthread=4
)

xgb_pred <- predict(my_model,as.matrix(test),reshape=T)
xgb_pred <- ifelse(xgb_pred == 0, "functional", ifelse(xgb_pred == 1, "functional needs repair", "non functional"))

xgb_pred <- data.table(id = test$id, status_group = xgb_pred)

fwrite(xgb_pred, file = "./submissions/tunning_models/xgboost/26_xgboost_sin_tunear.csv")
# 0.8118 Sin tunear (ultimo mlogloss: 0.185288)

#-- Modelo 2: aumentando eta a 0.5
# ¿Y si aumentamos el valor de eta? Por defecto esta a 0.3, podriamos aumentarlo antes que aumentar el numero de iteraciones
params = list(
  objective = "multi:softmax",
  num_class = 3,
  eta       = 0.5
)

my_model_2 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 500)
xgb_pred_2 <- make_predictions_xgboost(my_model_2, test)
fwrite(xgb_pred_2, file = "./submissions/tunning_models/xgboost/26_xgboost_eta_0.5.csv")
# 0.8094 con eta 0.5 (ultimo mlogloss: 0.111186)

# El hecho de aumentar eta a 0.5 no parece hacer que mejore el modelo
# Podriamos probar con 0.4
params = list(
  objective = "multi:softmax",
  num_class = 3,
  eta       = 0.4
)

my_model_3 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 500)
xgb_pred_3 <- make_predictions_xgboost(my_model_3, test)
fwrite(xgb_pred_3, file = "./submissions/tunning_models/xgboost/26_xgboost_eta_0.4.csv")
# 0.8110 con eta 0.4 (ultimo mlogloss: 0.142726)

# Emplear eta's demasiado altos hace que el modelo empeore, se sobreajuste ¿Y si reducimos el valor eta?
params = list(
  objective = "multi:softmax",
  num_class = 3,
  eta       = 0.1
)

my_model_4 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 500)
xgb_pred_4 <- make_predictions_xgboost(my_model_4, test)
fwrite(xgb_pred_4, file = "./submissions/tunning_models/xgboost/26_xgboost_eta_0.1.csv")
# 0.8118 con eta 0.1 (ultimo mlogloss: 0.339880)

feature_importance <- xgb.importance(feature_names = names(xgb.train), model = my_model)
xgb.ggplot.importance(
  importance_matrix = feature_importance
)
# Hay muchas variables que no son relevantes
#-- Podemos probar seleccionando variables mas relevantes (hasta fe_dr_month)
params = list(
  objective = "multi:softmax",
  num_class = 3
)
variables  <- feature_importance[feature_importance$Gain > 0.01, "Feature"]$Feature
xgb.train = xgb.DMatrix(data=as.matrix(train[,..variables]), label=vector_status_group)
my_model_5 <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 600)
xgb_pred_5 <- make_predictions_xgboost(my_model_5, test[, ..variables])
fwrite(xgb_pred_5, file = "./submissions/tunning_models/xgboost/26_xgboost_feature_importance_hasta_dr_month.csv")
# 0.8116 y 0.188664 Con 500 iteraciones
# 0.8120 y 0.161307 Con 600 iteraciones
# 0.8104 y 0.139704 Con 700 iteraciones




