#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 09_fe_menos_10000_lumping_mediana_freq_abs_categoricas.R
# Salida: Datos con nuevas variables (incluyendo wpt_name) + lumping + transformacion fe_funder, fe_ward, lga, installer y scheme_name
#         1. Añadir tanto subvillage como wpt_name
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
  library(embed)                # Creacion de modelos embedding
  library(doParallel)           # Paralelizacion de funciones
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward, scheme_name y wpt_name
dattrainOrlab$funder      <- dattrainOr$funder
dattrainOrlab$ward        <- dattrainOr$ward
dattrainOrlab$scheme_name <- dattrainOr$scheme_name
dattrainOrlab$wpt_name    <- dattrainOr$wpt_name


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
numlev_df %>% arrange(levels) # wpt_name: 45684 categorias unicas

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
# En el caso de subvillage, tan solo se ve reducido en 45684 - 45042 = 642 categorias

# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
cols <- c('funder', 'ward', 'scheme_name', 'wpt_name')
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

#-- fe_wpt_name
#-- Aplicamos lumping sobre la mediana (50 % de categorias con una proporcion menor a 2e-05)
summary(c(prop.table(table(datcompleto[, fe_wpt_name]))))
datcompleto[, fe_wpt_name := fct_lump_prop(datcompleto[,fe_wpt_name], 2e-05, other_level = "other")]
datcompleto$fe_wpt_name <- as.character(datcompleto$fe_wpt_name)

datcompleto[, wpt_name := NULL]

#-- Imputacion de las variables categoricas por sus frecuencias absolutas (a excepcion de wpt_name)
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

# Con solo wpt_name        : 0.8165993
# wpt_name sin depurar     : 0.8155892
# aplicando clean_text     : 0.8153367
# clean_text y lumping     : 0.8167508
my_model_18 <- fit_random_forest(formula, train)

my_sub_18 <- make_predictions(my_model_18, test)
# guardo submission
fwrite(my_sub_18, file = "./submissions/temp/18_lumping_fe_freq_abs_menos_60000_categorias_wpt_name_solo_clean_text_lumping.csv")
# Con solo wpt_name        : 0.8236
# wpt_name sin depurar     : 0.8220
# aplicando clean_text     : 0.8207
# clean_text y lumping     : 0.8227



#-- Conclusion: 
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_18$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616, 0.8164478, 0.8165657, 0.8167508), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239, 0.8226, 0.8222, 0.8223),
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
                                      "Num + Cat (> 1 & < 60000) fe anteriores + lumping + freq. abs. sobre categoricas")),
             align = 'c')

# Incluso reduciendo ligeramente el numero de cateogrias
ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/18_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_y_subvillage_y_wpt_name_sin_imp_freq_abs_y_resto_categoricas.png')

#-- Conclusion: el hecho de añadir nuevas variables como subvillage o wpt_name no han aportado mejoria alguna al modelo. No obstante,
#   ¿Y si incluimos las variables logicas permit y piublic_meeting?











