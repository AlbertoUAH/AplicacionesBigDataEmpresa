#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos entrada bombas (mejor resultado concurso)
# Salida: Datos con nuevas variables (incluyendo categorias < 2100) + transformacion fe_funder y fe_ward
#         1. Realizar una transformacion sobre las variables fe_funder y fe_ward
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

#-- Imputacion de fe_funder y fe_ward por frecuencia absoluta
#-  fe_funder
datcompleto[, fe_funder_freq := as.numeric(.N), by = fe_funder]
length(unique(datcompleto[, fe_funder_freq])) # 169 vs 999 fe_funder

#- fe_ward
datcompleto[, fe_ward_freq := as.numeric(.N), by = fe_ward]
length(unique(datcompleto[, fe_ward_freq]))  # 129 vs 904 fe_ward

# Eliminamos las columnas originales
datcompleto[, `:=`(fe_funder = NULL, fe_ward = NULL)]

#-- extraction_type_class - extraction_type_group - extraction_type (45.08 % gravity) en este orden de importancia
datcompleto[, fe_extraction := paste(extraction_type_class, extraction_type_group, extraction_type, sep = "_+_")]
summary(c(prop.table(table(datcompleto[, fe_extraction])))) # Eliminamos el 25 % de las categorias (proporcion menor a 1.5e-03)
datcompleto[, fe_extraction := fct_lump_prop(datcompleto[,fe_extraction], 
                                                  1.5e-03, other_level = "other")] # 14 categorias

#- Eliminamos las columnas originales
datcompleto[, `:=`(extraction_type = NULL, extraction_type_class = NULL, extraction_type_group = NULL)]

#-- source - source_type - source_class (en source y source_type hay un 28.55 % spring) en este orden de importancia
datcompleto[, fe_source := paste(source, source_type, source_class, sep = "_+_")]
summary(c(prop.table(table(datcompleto[, fe_source])))) # other_+_other_+_unknown y unknown_+_other_+_unknown ¿Las podriamos agrupar?
datcompleto[, fe_source := fct_lump_prop(datcompleto[,fe_source], 
                                                  0.01, other_level = "other")] # 9 categorias

#- Eliminamos las columnas originales
datcompleto[, `:=`(source = NULL, source_type = NULL, source_class = NULL)]

#-- scheme_management - management - management_group (en management y scheme_management hay un 68.19 y 61.94 % de WWC) en este orden de importancia
datcompleto[, fe_management := paste(scheme_management, management, management_group, sep = "_+_")]
summary(c(prop.table(table(datcompleto[, fe_management])))) # ¿Y si eliminamos la mitad de las categorias?
datcompleto[, fe_management := fct_lump_prop(datcompleto[,fe_management], 
                                         3e-4, other_level = "other")] # 46 categorias

#- Eliminamos las columnas originales
datcompleto[, `:=`(scheme_management = NULL, management = NULL, management_group = NULL)]

#-- waterpoint_type y waterpoint_type_group (hay un 58.29 y 48.02 % communal standpipe)
datcompleto[, fe_waterpoint := paste(waterpoint_type, waterpoint_type_group, sep = "_+_")]
summary(c(prop.table(table(datcompleto[, fe_waterpoint])))) # ¿Y si agrupamos other_+_other y dam_+_dam?
datcompleto[, fe_waterpoint := ifelse(fe_waterpoint %in% c("other_+_other", "dam_+_dam"), 
                                      "other",
                                      fe_waterpoint)]

datcompleto[, `:=`(waterpoint_type = NULL, waterpoint_type_group = NULL)]

#-- Modelo
# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

formula   <- as.formula("status_group~.")
# 0.8155219
my_model_11 <- fit_random_forest(formula,
                                train)

my_sub_11 <- make_predictions(my_model_11, test)
# guardo submission
fwrite(my_sub_11, file = "./submissions/11_05_lumping_sobre_funder_ward_freq_abs_agrupar_columnas.csv")
# 0.8197

knitr::kable(data.frame("Train accuracy" = c(0.8168687, 0.8101178, 0.8122391, 0.8124579, 
                                             0.8122727, 0.8149832, 0.8159764, 0.8146633, 
                                             0.8162121, 0.8154882, 0.8155219), 
                        "Data Submission" = c(0.8128, 0.8096, 0.8174, 0.8176, 0.8168, 
                                              0.8197, 0.8213, 0.8203, 0.8198, 0.8216,
                                              0.8197),
                        row.names = c("Num + Cat (> 1 & < 1000) sin duplicados",
                                      "Num + Cat (> 1 & < 1000) sin duplicados imp",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua",
                                      "Num + Cat (> 1 & < 1000) fe cyear + dist + cant_agua + dr_year + dr_month + abs(dr_year -cyear)", "Num + Cat (> 1 & < 1000) fe + tunning",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana) + hashed",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + freq. abs. sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + freq. abs. sobre funder + ward (mediana) + agrupacion columnas")),
             align = 'c')

#-- Conclusion: aparentemente, agrupando columnas no parece mejorar el modelo
#  ¿Todas las columnas agrupadas no son relevantes?
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_11$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/11_05_lumping_sobre_funder_ward_freq_abs_agrupar_columnas.png')

# Se ha probado a eliminar por orden de importancia y se han obtenido: 0.8191, 0.8215, 0.8201 (no parece beneficiar la agregacion de columnas)


