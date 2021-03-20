#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_17
# Inputs: Datos entrada bombas (mejor resultado concurso)
# Salida: Datos con nuevas variables (incluyendo categorias < 10000)
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
  library(stringr)              # Tratamiento cadenas caracteres
  library(tictoc)               # Calcular tiempos
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

#-- Nos traemos funder, ward, installer y scheme_name (menos de 10000 categorias)
dattrainOrlab$funder <- dattrainOr$funder
dattrainOrlab$ward <- dattrainOr$ward
dattrainOrlab$installer <- dattrainOr$installer
dattrainOrlab$scheme_name <- dattrainOr$funder

formula   <- as.formula("status_group ~ .")

dattrainOrlab$status_group <- as.factor(dattrainOrlab$status_group)
# 0.8158586
my_model_5 <- fit_random_forest(formula,
                                dattrainOrlab)

my_sub_5 <- make_predictions(my_model_5, dattestOr)
# guardo submission
fwrite(my_sub_5, file = "./submissions/05_03_mas_funder_ward_installer_scheme.csv")

knitr::kable(data.frame("Train accuracy" = c(0.8168687, 0.8100168, 0.812138, 0.8122727, 0.8134343, 0.8158586), 
                        "Data Submission" = c(0.8128, 0.8101, 0.8167, 0.8189, 0.8168, 0.8134),
                        row.names = c("Num + Cat (> 1 & < 1000) sin duplicados",
                                      "Num + Cat (> 1 & < 1000) sin duplicados imp",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua + dr_year + dr_month + abs(dr_year -cyear)", "Num + Cat (> 1 & < 1000) fe + tunning",
                                      "Num + Cat (> 1 & < 10000) + fe modelo 3")),
             align = 'c')
