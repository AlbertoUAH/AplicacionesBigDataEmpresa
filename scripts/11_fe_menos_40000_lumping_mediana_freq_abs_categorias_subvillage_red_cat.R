#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 09_fe_menos_10000_lumping_mediana_freq_abs_categoricas.R
# Salida: Datos con nuevas variables (incluyendo categorias < 40000) + lumping + transformacion fe_funder, fe_ward, lga, installer y scheme_name
#         1. Aplicar el mismo proceso de la iteracion 5 (reemplazar las variables categoricas por sus correspondientes frecuencias absolutas)
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
  library(embed)
  library(doParallel)
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward, scheme_name y subvillage (menos de 40000 categorias)
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
# En el caso de subvillage, tan solo se ve reducido en 21426 - 21295 = 131 categorias

#-- Curiosidad: hay cuidades que tan solo se diferencian por una letra. Ejemplo: Bulima, Bulima A, Bulima B ¿Podriamos agruparlas?
datcompleto[, "subvillage" := lapply(.SD, function(x) {
  stri_replace_all_regex(x, pattern = "\\s[a-zA-Z]$", replacement = "")
}), .SDcols = c("subvillage")]

length(unique(datcompleto[, subvillage])) # Pasamos a 19898 categorias

# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
cols <- c('funder', 'ward', 'scheme_name', 'subvillage')
datcompleto[ , paste0('fe_',cols) := lapply(.SD, clean_text), .SDcols = cols]
rm(cols)

length(unique(datcompleto[, fe_subvillage])) # 19781 (hemos eliminado 1645 categorias)

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

datcompleto[, subvillage := NULL]

#-- Imputacion de las variables categoricas por sus frecuencias absolutas (a excepcion de subvillage)
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

# 0.8165657
my_model_15 <- fit_random_forest(formula, train)

my_sub_15 <- make_predictions(my_model_15, test)
# guardo submission
fwrite(my_sub_15, file = "./submissions/15_13_lumping_sobre_funder_ward_installer_scheme_name_freq_abs_categoricas_mas_subvillage_freq_abs.csv")
# 0.8223

#-- Conclusion: 
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_15$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8162121, 0.8154882, 0.8157071, 0.808771, 0.7995118,
                                             0.8155556, 0.8168182, 0.8161616, 0.8160606, 0.8165657), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8213, 0.8203, 0.8198, 0.8216, 0.8226, 0.8118, 0.8056, 0.8191, 0.8232,
                                              0.8239, 0.8238, 0.8223),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana) + hashed",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping (mediana) + freq. abs. sobre funder + ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + target enc.",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + target enc. sin vars. originales",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + word emb. < 50",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping sobre funder, ward, inst. scheme_name (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping sobre funder, ward, scheme_name (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 40000) fe anteriores + subvillage + lumping sobre funder, ward, scheme_name (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 40000) fe anteriores + subvillage (red. cat) + lumping sobre funder, ward, scheme_name (mediana)")),
             align = 'c')

# Incluso reduciendo ligeramente el numero de cateogrias
ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/15_13_lumping_sobre_funder_ward_installer_scheme_name_freq_abs_categoricas_mas_subvillage_freq_abs.png')













