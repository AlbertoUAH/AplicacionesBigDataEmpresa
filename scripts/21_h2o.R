#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_07
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Modelo autoML con h2o
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
  library(h2o)                  # AutoML
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leemos el fichero con los datos completos imputados
datcompleto_imp <- fread("./data/datcompleto_imp_ap_15.csv" )
names(datcompleto_imp)[14] <- "fe_dr_year_cyear_diff"
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE )
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto_imp$id == 50785)

formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
vector_status_group <- dattrainOrlab$status_group
train[, status_group := vector_status_group]
train[, status_group := as.factor(status_group)]

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

#-- Variable objetivo
y <- 'status_group'

#-- Resto de variables
pred <- setdiff(names(train), y)

h2o.init()

#-- Conversion train y test a objeto h2o 
train_h <- as.h2o(train)
test_h  <- as.h2o(test)

#-- Lanzamos el modelo
# Probamos con 10 modelos
aml <- h2o.automl(x = pred, y = y,
                 training_frame = train_h,
                 max_models = 10,
                 seed = 1234,
                 exclude_algos = c("DeepLearning", "StackedEnsemble"),
                 balance_classes = TRUE
)

#-- Probamos el primer modelo (lider)
leader <- aml@leader
#-- Realizamos la prediccion
prediccion    <- h2o.predict(leader, newdata = test_h)
prediccion_df <- as.data.frame(prediccion)

prediccion_df    <- prediccion_df$predict
submission       <- data.table(id = test$id, status_group = prediccion_df)
fwrite(submission, "./submissions/tunning_models/h2o/first_automl_model.csv")

#-- Vemos el resto
model_id <- as.vector(aml@leaderboard$model_id)
for(i in 1:length(model_id)) {
  aml_aux <- h2o.getModel(aml@leaderboard[i, 1])
  prediccion    <- h2o.predict(aml_aux, newdata = test_h)
  prediccion_df <- as.data.frame(prediccion)
  
  prediccion_df    <- prediccion_df$predict
  submission       <- data.table(id = test$id, status_group = prediccion_df)
  path             <- paste0("./submissions/tunning_models/h2o/",model_id[i],".csv")
  fwrite(submission, path)
}

# XGBoost_2_AutoML_20210418_094427: 0.8101
# XGBoost_1_AutoML_20210418_094427: 0.8118
# XGBoost_3_AutoML_20210418_094427: 0.8020
# GBM_5_AutoML_20210418_094427    : 0.8079
# GBM_4_AutoML_20210418_094427    : 0.8137
# GBM_3_AutoML_20210418_094427    : 0.8077
# GBM_2_AutoML_20210418_094427    : 0.8027
# DRF_1_AutoML_20210418_094427    : 0.8137
# GBM_1_AutoML_20210418_094427    : 0.7945
# GLM_1_AutoML_20210418_094427    : 0.6805



h2o.shutdown(prompt = F)

