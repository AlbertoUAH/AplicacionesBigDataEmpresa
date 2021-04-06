#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_05
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Feature Engineering sobre valores "anomalos"
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

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto_imp$id == 50785)

#-- Nos encontrabamos valores anomalos en: construction_year, gps_height y longitude
datcompleto_imp[, construction_year := ifelse(construction_year == 0, NA, construction_year)]
datcompleto_imp[, gps_height := ifelse(gps_height == 0, NA, gps_height)]
datcompleto_imp[, longitude := ifelse(longitude == 0, NA, longitude)]

#-- Imputacion por missRanger
datcompleto_imp <- missRanger(datcompleto_imp,
                              pmm.k = 5,
                              seed = 1234,
                              maxiter = 100)

#-- Si recalculamos las variables...
date_recorded <- c(dattrainOr$date_recorded, dattestOr$date_recorded)
datcompleto_imp[, fe_dr_year_cyear_diff := year(date_recorded) - datcompleto_imp[, construction_year]]
datcompleto_imp[, fe_cyear := 2014 - datcompleto_imp[, construction_year]]
datcompleto_imp[, fe_dist := geodist(latitude, longitude, 0, 0)]

formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Con las variables imputadas                                 : 0.8167508
# Con las variables imputadas mas otras variables recalculadas: 0.8166162
my_model_21 <- fit_random_forest(formula, train)

my_sub_21   <- make_predictions(my_model_21, test)
# guardo submission
fwrite(my_sub_21, file = "./submissions/temp/21_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_mas_logicas_imputacion_vars_anomalas_mas_correcion_otras_vars.csv")
# Con las variables imputadas                                 : 0.8227
# Con las variables imputadas mas otras variables recalculadas: 0.8216

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616, 0.8164478, 0.8165657, 0.8167508, 0.8168855, 0.8168855, 0.8167508), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239, 0.8226, 0.8222, 0.8223, 0.8251, 0.8248, 0.8227),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (primer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + hashed sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + freq. abs. sobre funder + ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward + freq. abs. cat.",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward + target encoding",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + target encoding sobre funder y ward (y lga)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + word embed sobre funder y ward (dim. 2) + freq. abs. cat.",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas",
                                      "Num + Cat (> 1 & < 40000) fe anteriores + lumping + freq. abs. sobre categoricas (salvo subv)",
                                      "Num + Cat (> 1 & < 40000) fe anteriores + lumping + freq. abs. sobre categoricas (salvo subv + red. cat.)",
                                      "Num + Cat (> 1 & < 60000) fe anteriores + lumping + freq. abs. sobre categoricas",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas + fe date_recorded",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas + imp. anomalos")),
             align = 'c')

#-- Conclusion: no merece la pena la imputacion de valores anomalos
