#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_07
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Tuneo xgboost en h2o
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

#-- col_sample_rate_per_tree: indica el numero de variables a sortear en el arbol
#-- col_sample_rate_by_level: indica el numero de variables a sortear por nivel
gbm_params1 <- list(eta = c(0.02), max_depth = c(15),
                    colsample_bytree = 0.3, colsample_bylevel = 0.9,
                    colsample_bynode = 0.7)

xgboost_grid <- h2o.xgboost(x = pred, y = y,
                         training_frame = train_h,
                         ntrees = 600, seed = 1234,
                         distribution = "multinomial",
                         eta = 0.02, max_depth = 15,
                         colsample_bytree = 0.3, 
                         colsample_bylevel = 0.9,
                         colsample_bynode = 0.7)

colsample_bylevel = c(0.6)
for(i in 1) {
  xgboost_model <- h2o.getModel(xgboost_grid@model_ids[[i]])
  prediccion    <- h2o.predict(xgboost_model, newdata = test_h)
  prediccion_df <- as.data.frame(prediccion)
  
  prediccion_df    <- prediccion_df$predict
  submission       <- data.table(id = test$id, status_group = prediccion_df)
  path             <- paste0("./submissions/tunning_models/xgboost/h2o/xgboost_colsample_bylevel_",
                             colsample_bylevel[i],".csv")
  fwrite(submission, path)
}

# Empleando col_sample_rate_per_tree (hacer tabla con los resultados de XGboost)
# 0.2: 0.8230 - 0.3: 0.8237 - 0.4: 0.8253 - 0.5: 0.8248 - 0.6: 0.8169 - 1: 0.8217

# Empleando colsample_bylevel
# 0.2: 0.8193 - 0.3: 0.8193 - 0.4: 0.8216 - 0.5: 0.8214 - 0.6: 0.8193

# colsample_bynode
# 0.2: 0.8193 - 0.3: 0.8193 - 0.4: 0.8216 - 0.5: 0.8214 - 0.6: 0.8193

h2o.shutdown(prompt = FALSE)
