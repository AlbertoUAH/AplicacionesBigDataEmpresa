#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_17
# Inputs: Datos entrada bombas (mejor resultado concurso)
# Salida: Datos con nuevas variables (incluyendo categorias < 2100) + lumping
#         1. Realizar una transformacion lumping sobre las variables funder y ward (sobre la mediana de las proporciones)
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
mean(duplicated(hash.mapping(mat_hash))) # 0.5491329

# ¿Y si utilizamos 2^10?
mat_hash_2 <- hashed.model.matrix(~., datcompleto[, c("fe_funder", "fe_ward")], 2^10, create.mapping = TRUE, )
mean(duplicated(hash.mapping(mat_hash_2))) # 0.5491329 Coincide
rm(mat_hash_2)



# Sustituimos las columnas fe_funder y fe_ward por su valor hash correspondiente
vector_hash <- hash.mapping(mat_hash)
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
length(unique(datcompleto$fe_funder_hashed)) # 647
#- Numero de valores unicos en fe_ward_hashed
length(unique(datcompleto$fe_ward_hashed)) # 593

# Eliminamos las columnas fe_funder y fe_ward
datcompleto[, `:=`(fe_funder = NULL, fe_ward = NULL)]

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

formula   <- as.formula("status_group~.")
# 0.8162121
my_model_8 <- fit_random_forest(formula,
                                train)

my_sub_8 <- make_predictions(my_model_8, test)
# guardo submission
fwrite(my_sub_8, file = "./submissions/08_05_lumping_y_hashing_sobre_funder_ward.csv")

knitr::kable(data.frame("Train accuracy" = c(0.8168687, 0.8101178, 0.8122391, 0.8124579, 0.8122727, 0.8149832, 0.8159764, 0.8146633, 0.8162121), 
                        "Data Submission" = c(0.8128, 0.8096, 0.8174, 0.8176, 0.8168, 0.8197, 0.8213, 0.8203, 0.8198),
                        row.names = c("Num + Cat (> 1 & < 1000) sin duplicados",
                                      "Num + Cat (> 1 & < 1000) sin duplicados imp",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua + dr_year + dr_month + abs(dr_year -cyear)", "Num + Cat (> 1 & < 1000) fe + tunning",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana) + hashed")),
             align = 'c')

#-- Conclusion: incluso aplicando una funcion hashing no mejora la precision del modelo

