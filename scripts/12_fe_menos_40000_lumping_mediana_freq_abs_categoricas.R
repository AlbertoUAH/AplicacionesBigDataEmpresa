#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 11_fe_menos_10000_lumping_mediana_freq_abs_categoricas.R
# Salida: Aplicar transformacion por lumping (mediana) y frecuencias absolutas a subvillage
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
  library(embed)                # Creacion de modelos embedded
  library(doParallel)           # Paralelizacion de funciones
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward, scheme_name y subvillage (menos de 40000 categorias, a excepcion de installer)
dattrainOrlab$funder      <- dattrainOr$funder
dattrainOrlab$ward        <- dattrainOr$ward
dattrainOrlab$scheme_name <- dattrainOr$scheme_name
dattrainOrlab$subvillage  <- dattrainOr$subvillage

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
numlev_df %>% arrange(levels) # Subvillage: 21426 categorias unicas

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
}
names(numlev_dt) <- c('vars', 'levels')
numlev_dt %>% arrange(levels) # Todas las categorias se reducen en numero de variables
# scheme_name: de 2869 a 2615
# funder     : de 2141 a 2110
# ward       : de 2098 a 2096
# En el caso de subvillage, tan solo se ve reducido en 21426 - 21295 = 131 categorias

# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
cols <- c('funder', 'ward', 'scheme_name', 'subvillage')
datcompleto[ , paste0('fe_',cols) := lapply(.SD, clean_text), .SDcols = cols]
rm(cols)

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

#-- fe_scheme_name
#-- Aplicamos lumping sobre la mediana (50 % de categorias con una proporcion menor a 8.1e-05)
summary(c(prop.table(table(datcompleto[, fe_scheme_name]))))
datcompleto[, fe_scheme_name := fct_lump_prop(datcompleto[,fe_scheme_name], 8.1e-05, other_level = "other")]
datcompleto$fe_scheme_name <- as.character(datcompleto$fe_scheme_name)

datcompleto[, scheme_name := NULL]

#-- subvillage
#-- Aplicamos lumping sobre la mediana (50 % de las categorias con una proporcion menor a 3e-05)
summary(c(prop.table(table(datcompleto[, fe_subvillage]))))
datcompleto[, fe_subvillage := fct_lump_prop(datcompleto[,fe_subvillage], 3e-05, other_level = "other")]
datcompleto$fe_subvillage <- as.character(datcompleto$fe_subvillage)

datcompleto[, subvillage := NULL]
# Pasamos de 21295 a 7511

length(unique(datcompleto[, fe_subvillage]))

#-- Imputacion de las variables categoricas por sus frecuencias absolutas
cat_cols <- names(datcompleto[, which(sapply(datcompleto, is.character)), with = FALSE])[-22]

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

#-- En relacion con fe_subvillage, pasamos de 7511 a tan solo 109 categorias
freq_despues_fe <- apply(datcompleto[, ..new_cat_cols], 2, function(x) length(unique(x)))

#-- Modelo
formula   <- as.formula("status_group~.")

# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# lumping + freq. abs. 0.8165993
# solo lumping         0.816633
# solo freq. abs.      0.8164478
# variable sin imp.    0.81633
# variables sin imp t  0.8162963
my_model_16 <- fit_random_forest(formula, train)

my_sub_16 <- make_predictions(my_model_16, test)
# guardo submission
fwrite(my_sub_16, file = "./submissions/temp/16_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_y_subvillage_sin_imp_y_resto_categoricas_TOTAL.csv")
# lumping + freq.abs. 0.8219
# solo lumping        0.8221
# solo freq. abs.     0.8226
# variable sin imp.   0.8222
# variables sin imp.t 0.8220

#-- Conclusion: 
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_16$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616, 0.8164478), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239, 0.8226),
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
                                      "Num + Cat (> 1 & < 40000) fe anteriores + lumping + freq. abs. sobre categoricas (salvo subv)")),
             align = 'c')

# El hecho de incluir la variable subvillage ha provocado que la precision del modelo empeore, de 0.8239 a 0.8218
# Bien es cierto que hemos reducido el numero de categorias de mas de 20000 a tan solo 109 ¿Y si aplicamos unicamente el lumping,
# quedandonos con algo mas de 7000 categorias? Ademas, la variable subvillage se encuentra en torno a las variables mas importantes
# del modelo, por lo que podriamos estar descartando demasiadas categorias al sustituir las categorias por frecuencias

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/16_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_y_subvillage_sin_imp_y_resto_categoricas.png')













