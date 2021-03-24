#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 06_fe_menos_2100_lumping_mediana_freq_abs.R
# Salida: Datos con nuevas variables (incluyendo categorias < 2100) + lumping + transformacion fe_funder, fe_ward y lga + imputacion 
#         valores outlier
#         
#        
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
  library(doParallel)           # Calculo en paralelo
  library(missForest)           # Imputacion miss forest
  
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


#-- Imputacion de las variables categoricas por sus frecuencias absolutas
cat_cols <- names(datcompleto[, which(sapply(datcompleto, is.character)), with = FALSE])

#   Antes de imputar
freq_antes_fe <- apply(datcompleto[, ..cat_cols], 2, function(x) length(unique(x)))

for (cat_col in cat_cols) {
  datcompleto[, paste0("fe_", cat_col) := as.numeric(.N), by = cat_col]
}
names(datcompleto) <- stri_replace_all_fixed(names(datcompleto),
                                             "fe_fe_", "fe_")

for (cat_col in cat_cols) {
  datcompleto[, paste(cat_col) := NULL]
}
new_cat_cols <- paste0("fe_", stri_replace_all_fixed(cat_cols, "fe_", ""))

#-- Solo cambian funder, ward y lga en relacion al numero de categorias
freq_despues_fe <- apply(datcompleto[, ..new_cat_cols], 2, function(x) length(unique(x)))

#-- Imputacion (gps_height, longitude, construction_year)
for (column in c("construction_year", "gps_height", "longitude")) {
  new_feature <- paste0("fe_", column)
  
  datcompleto[, new_feature] <- datcompleto[, ..column]
  datcompleto[get(new_feature) == 0, (new_feature) := NA]

  datcompleto[, (column) := NULL]
}

registerDoParallel(cores = detectCores())
miss_forest_imp <- function(x) {
  miss_imp <- missForest(
    setDT(x),  
    maxiter = 10,
    ntree = 100, 
    parallelize = "variables"
  )$ximp
  miss_imp
  print(class(miss_imp))
  list("fe_construction_year" = as.double(miss_imp$fe_construction_year),
       "fe_gps_height" = as.double(miss_imp$fe_gps_height),
       "fe_longitude" = as.double(miss_imp$fe_longitude))
} 
dat_imp <- datcompleto[, miss_forest_imp(.SD), by = region_code, .SDcols = names(datcompleto)]

dat_imp_final <- cbind(datcompleto[, !c("fe_construction_year", "fe_gps_height", "fe_longitude")], dat_imp[, !"region_code"])

for (column in c("fe_construction_year", "fe_gps_height", "fe_longitude")) {
  dat_imp_final[is.na(get(column)), (column) := median(datcompleto[, get(column)], na.rm = TRUE)]
}

# Comprobamos valores missings
sum(is.na(dat_imp_final))

# Comprobamos que el orden del dataset no ha cambiado
which(dat_imp_final$id == 50785) == fila_test

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- dat_imp_final[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- dat_imp_final[c(fila_test:nrow(dat_imp_final)),]

formula   <- as.formula("status_group~.")
# 0.8102862 con district_code
# 0.8108754 con region_code (quedan 22052 valores missing)
my_model_15 <- fit_random_forest(formula,
                                 train)

my_sub_15 <- make_predictions(my_model_15, test)
# guardo submission
fwrite(my_sub_15, file = "./submissions/15_05_lumping_sobre_funder_ward_freq_abs_categoricas_num_imp_por_distrito.csv")
# 0.8173 con district_code

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8162121, 0.8154882, 0.8157071, 0.8156229, 0.8153199, 0.8154545, 0.8157576), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8213, 0.8203, 0.8198, 0.8216, 0.8226, 0.8216, 0.8227, 0.8214, 0.8199),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana) + hashed",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping (mediana) + freq. abs. sobre funder + ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas + num. imp. miss Ranger",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas + num. imp. miss Forest",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas + num. imp. mediana",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas + num. imp. media")),
             align = 'c')

#-- Conclusion: la imputacion de missings mediante la mediana no ha proporcionado una mejoria al modelo ¿Podria mejorar si realizamos la imputacion por la media?









