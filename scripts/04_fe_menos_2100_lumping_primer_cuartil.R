#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_17
# Inputs: Datos 01_fe_menos_2100.R
# Salida: Datos con nuevas variables (incluyendo categorias < 2100 depuradas) + lumping
#         1. Realizar una transformacion lumping sobre las variables funder y ward (sobre el primer cuartil de las proporciones)
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

#-- ¿Y si corregimos todas las variables categoricas?
datcompleto <- as.data.table(datcompleto)

fe_cat <- data.table()
for (column in numlev_df$vars) {
  new_column <- paste0("fe_", column)
  fe_cat[, new_column] <- sapply(datcompleto[, ..column], clean_text)
}

numlev_dt <- data.frame()
for (i in 1:ncol(fe_cat)) {
  col_tmp <- fe_cat[, ..i]
  num_lev <- nrow(unique(col_tmp))
  numlev_dt[i, 1] <- names(fe_cat)[i]
  numlev_dt[i, 2] <- num_lev
  print(numlev_dt)
}
names(numlev_dt) <- c('vars', 'levels')
numlev_dt %>% arrange(levels)


# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
datcompleto[, fe_funder := clean_text(funder)][, fe_ward := clean_text(ward)]

#-- fe_funder
#-- Aplicamos lumping sobre el tercer cuartil (25 % de categorias con una proporcion menor a 2e-04)
summary(c(prop.table(table(datcompleto[, fe_funder]))))
datcompleto[, fe_funder := fct_lump_prop(datcompleto[,fe_funder], 2e-05, other_level = "other")]
datcompleto$fe_funder <- as.character(datcompleto$fe_funder)
# Pasamos de 2110 a 999 categorias
datcompleto[, funder := NULL]

#-- fe_ward
#-- Aplicamos lumping sobre el tercer cuartil (25 % de categorias con una proporcion menor a 2e-04)
summary(c(prop.table(table(datcompleto[, fe_ward]))))
datcompleto[, fe_ward := fct_lump_prop(datcompleto[,fe_ward], 2e-04, other_level = "other")]
datcompleto$fe_ward <- as.character(datcompleto$fe_ward)
# Pasamos de 2096 a 1483 categorias
datcompleto[, ward := NULL]

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

formula   <- as.formula("status_group~.")
# 0.8159259
my_model_8 <- fit_random_forest(formula,
                                train)

my_sub_8 <- make_predictions(my_model_8, test)

fwrite(my_sub_8, file = "./submissions/08_lumping_y_fe_sobre_funder_ward_primer_cuartil.csv")
# 0.8196

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (primer cuartil)")),
             align = 'c')

#-- Conclusion: aplicando lumping sobre la mediana de cada proporcion obtenemos el mejor valor de scoring 














