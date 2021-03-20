#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_17
# Inputs: Datos entrada bombas (mejor resultado concurso)
# Salida: Datos con nuevas variables (incluyendo categorias < 2100)
#         1. Conversion a minusculas, eliminacion espacios en blanco y signos puntuacion
#            en funder y ward
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
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward (menos de 2100 categorias)
dattrainOrlab$funder <- dattrainOr$funder
dattrainOrlab$ward <- dattrainOr$ward

#-- Unimos train y test
columnas_test  <- names(dattestOr)[names(dattestOr) %in% names(dattrainOrlab)]
datcompleto <- rbind(dattrainOrlab, dattestOr[, columnas_test])

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto$id == 50785)

#--- Niveles de las categoricas.
datcat_df <- as.data.frame(datcompleto %>% select(where(is.character)))

numlev_df <- data.frame()
for (i in 1:ncol(datcat_df)) {
  col_tmp <- datcat_df[, i]
  num_lev <- length(unique(col_tmp))
  numlev_df[i, 1] <- names(datcat_df)[i]
  numlev_df[i, 2] <- num_lev
  print(numlev_df)
}
names(numlev_df) <- c('vars', 'levels')
numlev_df %>% arrange(levels)

# funder, ward
#-- Demasiadas categorias ¿Puede que algunas se repitan?
datcompleto <- as.data.table(datcompleto)

# ¿Categorias que solo se diferencian por una letra en mayuscula o minuscula? ¿O porque esten separados?
datcompleto[, fe_funder := clean_text(funder)][, fe_ward := clean_text(ward)]

#-- fe_funder    (2110 categorias) vs funder      (2141 categorias)
#   fe_ward      (2096 categorias) vs ward        (2098 categorias)
#   No existe una diferencia muy significativa

# En relacion con el resto, borramos las columnas originales salvo lga (no cambia el numero de categorias)
datcompleto[, c("ward", "funder") := NULL]

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

formula   <- as.formula("status_group~.")
# 0.8149832
my_model_5 <- fit_random_forest(formula,
                                train)

my_sub_5 <- make_predictions(my_model_5, test)
# guardo submission
fwrite(my_sub_5, file = "./submissions/05_04_mas_fe_inst_funder_scheme_ward.csv")

knitr::kable(data.frame("Train accuracy" = c(0.8168687, 0.8101178, 0.8122391, 0.8124579, 0.8122727, 0.8149832), 
                        "Data Submission" = c(0.8128, 0.8096, 0.8174, 0.8176, 0.8168, 0.8197),
                        row.names = c("Num + Cat (> 1 & < 1000) sin duplicados",
                                      "Num + Cat (> 1 & < 1000) sin duplicados imp",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua + dr_year + dr_month + abs(dr_year -cyear)", "Num + Cat (> 1 & < 1000) fe + tunning",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward")),
             align = 'c')


