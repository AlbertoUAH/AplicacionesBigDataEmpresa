#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_12
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Tuneo semilla
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

#-- Probamos un xgboost por defecto (sin tunear)
vector_status_group <- ifelse(vector_status_group == "functional", 0
                              , ifelse(vector_status_group == "functional needs repair", 1
                                       , 2))
xgb.train = xgb.DMatrix(data=as.matrix(train), label=vector_status_group)

# Grid tunning
seeds <- c(1244)
for(seed in seeds) {
  params = list(
    objective = "multi:softmax",
    num_class = 3,
    colsample_bytree = 0.3,
    max_depth        = 15,
    eta              = 0.02
  )
  
  my_model <- fit_xgboost_model(params, xgb.train, xgb.train, nrounds = 600, seed = seed)
  xgb_pred <- make_predictions_xgboost(my_model, test)
  fwrite(xgb_pred, 
         file = paste0("./submissions/tunning_models/xgboost/semillas/xgboost_with_seed_",seed,".csv")
  )
}





