#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 02_fe_menos_2100_lumping_mediana.R
# Salida: Datos con nuevas variables (incluyendo categorias < 2100) + lumping + transformacion fe_funder, fe_ward y lga
#         1. Reemplazar las variables fe_funder, fe_ward y lga por el promedio de aparicion con respecto a la variable objetivo 
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
  library(missRanger)           # Imputacion missings
  library(dataPreparation)      # Target encoding
  
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

# Recuperamos las columnas categoricas
datcat_df <- as.data.frame(datcompleto %>% select(fe_funder, fe_ward, lga))

#-- Imputacion de las variables categoricas mediante target encoding
datcompleto[, names(datcat_df) := lapply(.SD, as.factor), .SDcols=names(datcat_df)]
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

train[, status_group := ifelse(status_group == "functional", 0, 
                               ifelse(status_group == "functional needs repair", 1
                                      , 2))]

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

#-- Target encoding
dat_encoded <- build_target_encoding(train, 
                                     cols_to_encode = names(datcat_df),
                                     target_col = "status_group",
                                     verbose = TRUE)

train <- target_encode(train, target_encoding = dat_encoded)
test <- target_encode(test, target_encoding = dat_encoded)
sum(is.na(test)) # 21 valores missing en status_group_by_fe_funder

# Reemplazamos los valores nulos por la media
media <- mean(test[, status_group_mean_by_fe_funder], na.rm = TRUE)
setnafill(test, cols="status_group_mean_by_fe_funder", fill = media)

# Añadimos ruido a los datos train
cat_cols <- paste0('status_group_mean_by_',names(datcat_df))

train[, cat_cols] <- train[, lapply(.SD, function(x) x * rnorm(length(x), mean = 1, sd = 0.05)), .SDcols = cat_cols]

train[, status_group := ifelse(status_group == 0, "functional", 
                               ifelse(status_group == 1, "functional needs repair"
                                      , "non functional"))]
train$status_group <- as.factor(train$status_group)

#-- Modelo
formula   <- as.formula("status_group~.")
# 0.8152694 si no eliminamos las variables
my_model_13 <- fit_random_forest(formula,
                                 train)

my_sub_13 <- make_predictions(my_model_13, test)

# guardo submission
fwrite(my_sub_13, file = "./submissions/13_lumping_fe_sobre_funder_ward_target_encoding_solo_funder_ward_lga.csv")
# 0.8185 si no eliminamos las variables

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (primer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + hashed sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + freq. abs. sobre funder + ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward + freq. abs. cat.",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward + target encoding",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + target encoding sobre funder y ward (y lga)")),
             align = 'c')

#-- Conclusion: ni siquiera aplicando target encoding sobre las variables funder, ward y lga, es mejor un modelo imputado mediante freq. abs
#   ¿Y si aplicamos word embedding sobre funder y ward?




