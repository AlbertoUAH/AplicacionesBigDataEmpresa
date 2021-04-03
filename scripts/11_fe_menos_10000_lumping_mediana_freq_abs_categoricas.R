#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 07_fe_menos_2100_lumping_mediana_freq_abs_categoricas.R
# Salida: Aplicar la imputacion por frecuencias absolutas a variables como scheme_name e installer
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
  library(embed)                # Creacion de modelos embebidos
  library(doParallel)           # Paralelizacion de funciones
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward, installer y scheme_name (menos de 10000 categorias)
dattrainOrlab$funder      <- dattrainOr$funder
dattrainOrlab$ward        <- dattrainOr$ward
#dattrainOrlab$installer   <- dattrainOr$installer
dattrainOrlab$scheme_name <- dattrainOr$scheme_name

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
}
names(numlev_dt) <- c('vars', 'levels')
numlev_dt %>% arrange(levels) # Todas las categorias se reducen en numero de variables
# scheme_name: de 2869 a 2615
# funder     : de 2141 a 2110
# ward       : de 2098 a 2096
# installer  : de 2411 a 2069

# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
cols <- c('funder', 'ward', 'scheme_name')
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

#-- fe_installer
#-- Aplicamos lumping sobre la mediana (50 % de categorias con una proporcion menor a 2e-05)
# summary(c(prop.table(table(datcompleto[, fe_installer]))))
# datcompleto[, fe_installer := fct_lump_prop(datcompleto[,fe_installer], 2e-05, other_level = "other")]
# datcompleto$fe_installer <- as.character(datcompleto$fe_installer)
# 
# datcompleto[, installer := NULL]
# Pasamos de 2069 a 1029 categorias

#-- fe_scheme_name
summary(c(prop.table(table(datcompleto[, fe_scheme_name]))))
datcompleto[, fe_scheme_name := fct_lump_prop(datcompleto[,fe_scheme_name], 8.1e-05, other_level = "other")]
datcompleto$fe_scheme_name <- as.character(datcompleto$fe_scheme_name)

datcompleto[, scheme_name := NULL]
# Pasamos de 2615 a 1205


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

#-- Modelo
formula   <- as.formula("status_group~.")

# Dividimos entre conjunto de entrenamiento y prueba
train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Con fe_installer y fe_scheme_name 0.8158081
# Con fe_installer                  0.8160774
# Con fe_scheme_name                0.8161616
my_model_15 <- fit_random_forest(formula, train)

my_sub_15 <- make_predictions(my_model_15, test)
# guardo submission
fwrite(my_sub_15, file = "./submissions/temp/15_lumping_fe_freq_abs_sobre_funder_ward_y_resto_categoricas_mas_scheme_name.csv")
# Con fe_installer y fe_scheme_name 0.8231
# Con fe_installer                  0.8211
# Con fe_scheme_name                0.8239

#-- Conclusion: aplicando frecuencias absolutas sobre las variables cartegoricas ha permitido mejorar considerablemente el modelo
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_15$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

# A la vista de los resultados obtenidos, incorporar las variables installer y scheme_name ha permitido mejorar el modelo
# de 0.8226 a 0.8232. No obstante, analizando el grafico de importancia podemos observar que fe_installer presenta una
# importancia ligeramente mayor a fe_scheme_name ¿Y si descartamos esta ultima variable?

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239),
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
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas")),
             align = 'c')

# Como podemos comprobar, empleando fe_scheme_name el modelo ha mejorado ligeramente, pasando de 0.8231 a 0.8239 ¿Podremos mejorar el modelo
# yendo un paso mas adelante e incorporando variables con un elevado numero de categorias como subvillage y wpt_name?
ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/15_lumping_fe_freq_abs_sobre_funder_ward_y_resto_categoricas_mas_scheme_name.png')

