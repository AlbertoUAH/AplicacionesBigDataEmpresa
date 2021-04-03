#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_17
# Inputs: Datos 02_fe_menos_2100_lumping_mediana.R
# Salida: Datos con nuevas variables (incluyendo categorias < 2100 depuradas) + lumping
#         1. Realizar una transformacion hash sobre las variables fe_funder y fe_ward
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
  library(FeatureHashing)       # Feature Hashing
  
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
#-- Aplicamos lumping sobre la mediana (50 % de categorias con una proporcion menor a 2e-05)
summary(c(prop.table(table(datcompleto[, fe_funder]))))
datcompleto[, fe_funder := fct_lump_prop(datcompleto[,fe_funder], 2e-05, other_level = "other")]
datcompleto$fe_funder <- as.character(datcompleto$fe_funder)

datcompleto[, funder := NULL]

#-- fe_ward
#-- Aplicamos lumping sobre la mediana (50 % de categorias con una proporcion menor a 4e-04)
summary(c(prop.table(table(datcompleto[, fe_ward]))))
datcompleto[, fe_ward := fct_lump_prop(datcompleto[,fe_ward], 4e-04, other_level = "other")]
datcompleto$fe_ward <- as.character(datcompleto$fe_ward)

datcompleto[, ward := NULL]

#-- Feature hashing
# Podemos elegir el tamaño minimo (teorico) que permite reducir el ratio de colision entre valores hash
tam_minimo <- hash.size(datcompleto[, c("lga")])
mat_hash   <- hashed.model.matrix(~., datcompleto[, c("fe_funder", "fe_ward")], tam_minimo, create.mapping = TRUE)
mean(duplicated(hash.mapping(mat_hash))) # 0.9327378

# ¿Y si utilizamos 2^10?
mat_hash_2 <- hashed.model.matrix(~., datcompleto[, c("fe_funder", "fe_ward")], 2^12, create.mapping = TRUE, )
mean(duplicated(hash.mapping(mat_hash_2))) # 0.5491329 Coincide 2 ^ 10, y 0.2138728 Coincide con 2 ^ 12
rm(mat_hash)

# Sustituimos las columnas fe_funder y fe_ward por su valor hash correspondiente
vector_hash <- hash.mapping(mat_hash_2)
mat_hash_dt <- data.table("feature" = names(vector_hash), 
                          "values" = vector_hash)

# Por defecto, hashed.model.matrix añade el nombre de columna a la variable
# De forma que debemos incluirlo tambien tanto en fe_funder como en fe_ward
# para hacer coincidir las categorias
datcompleto[, fe_funder := paste0("fe_funder", fe_funder)]
datcompleto[, fe_ward := paste0("fe_ward", fe_ward)]

datcompleto[mat_hash_dt, fe_funder_hashed := i.values,  on = .(fe_funder = feature)]
datcompleto[mat_hash_dt, fe_ward_hashed := i.values,  on = .(fe_ward = feature)]

#- Numero de valores unicos en fe_funder_hashed
length(unique(datcompleto$fe_funder_hashed)) # 647 con 2 ^ 10 y 878 con 2 ^ 12
#- Numero de valores unicos en fe_ward_hashed
length(unique(datcompleto$fe_ward_hashed)) # 593 con 2 ^ 10 y 797 con 2 ^ 12

# Eliminamos las columnas fe_funder y fe_ward
datcompleto[, `:=`(fe_funder = NULL, fe_ward = NULL)]

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

formula   <- as.formula("status_group~.")
# Con 2 ^ 10 0.8162121
# Con 2 ^ 12 0.8160774
my_model_9 <- fit_random_forest(formula,
                                train)

my_sub_9 <- make_predictions(my_model_9, test)
# guardo submission
fwrite(my_sub_9, file = "./submissions/09_lumping_fe_y_hashing_sobre_funder_ward.csv")
# Con 2 ^ 12 0.8213
# Con 2 ^ 10 0.8198

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (primer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + hashed sobre funder + ward (mediana)")),
             align = 'c')

#-- Conclusion: con un valor hashing 2 ^ 12, obtenemos un score ligeramente superior al modelo lumping (mediana)
#   ¿Y si aplicamos frecuencias absolutas sobre las variables?

