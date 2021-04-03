#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 09_fe_menos_10000_lumping_mediana_freq_abs_categoricas.R
# Salida: Añadidas variables logicas public_meeting y permit
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
  library(missRanger)
  library(mice)
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leo ficheros
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )
dattest <- fread(file = "./data/test_values.csv")

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Nos traemos funder, ward y scheme_name (menos de 10000 categorias)
#   Ademas de public_meeting y permit
dattrainOrlab$funder         <- dattrainOr$funder
dattrainOrlab$ward           <- dattrainOr$ward
dattrainOrlab$scheme_name    <- dattrainOr$scheme_name
dattrainOrlab$public_meeting <- dattrainOr$public_meeting
dattrainOrlab$permit         <- dattrainOr$permit


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
}
names(numlev_dt) <- c('vars', 'levels')
numlev_dt %>% arrange(levels) # Todas las categorias se reducen en numero de variables

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

#-- fe_scheme_name
summary(c(prop.table(table(datcompleto[, fe_scheme_name]))))
datcompleto[, fe_scheme_name := fct_lump_prop(datcompleto[,fe_scheme_name], 8.1e-05, other_level = "other")]
datcompleto$fe_scheme_name <- as.character(datcompleto$fe_scheme_name)

datcompleto[, scheme_name := NULL]


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

#-- Variables logicas (debemos imputarlas, dado que en el conjunto test existen missings)
sum(is.na(dattestOr$permit))
sum(is.na(dattestOr$public_meeting))


# Imputamos por missRanger
datcompleto_imp <- missRanger(datcompleto,
                            pmm.k = 5,
                            seed = 1234,
                            maxiter = 100)

# ¿Y si imputamos por mice?
# datcompleto_imp_mice <- mice(datcompleto,
#                         method = "logreg", # Regresion logistica con reemplazamiento
#                         maxit  = 5,            # Por defecto, emplea 5 iteraciones
#                         printFlag = TRUE,
#                         seed = 1234
# )
# datcompleto_imp <- as.data.table(complete(datcompleto_imp_mice))

# Comprobamos que las variables logicas estan imputadas
sum(is.na(datcompleto_imp))

# Las convertimos a variables numericas
datcompleto_imp[, public_meeting := as.numeric(public_meeting)]
datcompleto_imp[, permit := as.numeric(permit)]

# Creamos dos columnas adicionales que indiquen si la variable logica era o no NA
datcompleto_imp[, is_na_public_meeting := ifelse(is.na(datcompleto[, public_meeting]), 1, 0)]
datcompleto_imp[, is_na_permit := ifelse(is.na(datcompleto[, permit]), 1, 0)]

#-- Modelo
formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

# El hecho de si la variable era o no NA no parece ser informacion relevante
table(train[, c("is_na_public_meeting", "status_group")])
table(train[, c("is_na_permit", "status_group")])
table(train[, c("is_na_public_meeting", "is_na_permit", "status_group")])

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Empleando ambas variables logicas (+ col is na) + missRanger   : 0.8168855
# Empleando ambas variables logicas (+ col is na) + mice con boos: 0.8167845
# Empleando ambas variables logicas (+ col is na) + mice sin boos: 0.8172559
# Empleando ambas variables logicas                              : 0.8169529
# Empleando solo permit (mayor imp)                              : 0.8163973
# Empleando solo public_meeting                                  : 0.8159764

my_model_19 <- fit_random_forest(formula, train)

my_sub_19 <- make_predictions(my_model_19, test)
# guardo submission
fwrite(my_sub_19, file = "./submissions/temp/19_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_y_permit_public_meeting_CON_mice_sin_boostrap.csv")
# Empleando ambas variables logicas (+ col is na) + missRanger   : 0.8251
# Empleando ambas variables logicas (+ col is na) + mice con boos: 0.8233
# Empleando ambas variables logicas (+ col is na) + mice sin boos: 0.8240
# Empleando ambas variables logicas                              : 0.8237
# Empleando solo permit (mayor imp)                              : 0.8233
# Empleando solo public_meeting                                  : 0.8216

#-- Importancia variables
impor_df <- as.data.frame(my_model_19$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/19_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_y_permit_public_meeting.png')


















