suppressPackageStartupMessages({
  library(dplyr)          # Manipulacion de datos 
  library(data.table)     # Lectura y escritura de ficheros
  library(ggplot2)        # Representacion grafica
  library(inspectdf)      # EDAs automaticos
  library(ranger)         # randomForest (+ rapido que caret)
  library(forcats)        # Tratamiento de variables categoricas
  library(tictoc)         # Calculo de tiempo de ejecucion
  library(missRanger)     # Imputacion de valores NA
  library(knitr)          # Generacion de informes (formateo de tablas)
  library(gmt)            # Calculo de la distancia geografica
  library(stringi)        # Tratamiento de strings
  library(doParallel)           # Paralelizacion de funciones
  library(missRanger)           # Tratamiento de valores missing (mediante random forest)
  
  source("scripts/funciones.R") # Funciones propias
})

#--- Carga de ficheros train y test
dattrainOr    <- fread(file = "./data/train_values.csv", data.table = FALSE )
dattrainLabOr <- fread(file = "./data/train_labels.csv", data.table = FALSE )
dattestOr     <- fread(file = "./data/test_values.csv", data.table = FALSE  )

#-- Variables categoricas
datcat_df <- dattrainOr %>% select(where(is.character))

# Mediante un bucle for, 
numlev_df <- data.frame(
  "vars" = names(datcat_df),
  "levels" = apply(datcat_df, 2, 
                   function(x) length(unique(x)))
  
)
# Eliminamos los nombres de fila
rownames(numlev_df) <- NULL

kable(numlev_df %>% arrange(levels))

vars_gd <- numlev_df %>%
  filter(levels < 1000, levels > 1) %>% 
  select(vars)
datcat_gd <- datcat_df[ , vars_gd$vars]

#-- Variables numericas
datnum_df <- dattrainOr %>% select(where(is.numeric))
# Unificamos ambos tipos de variables...
datnumcat_df <- cbind(datnum_df, datcat_gd)

# ...Como tambien la variable objetivo
dattrainOrlab <- merge(
  datnumcat_df, dattrainLabOr,
  by.x = c('id'), by.y = c('id'),
  sort = FALSE
)

dattrainOrlab$payment_type <- NULL
dattrainOrlab$quantity_group <- NULL

# Train
dattrainOrlab$fe_cyear <- 2014 - dattrainOrlab$construction_year

# Test
dattestOr$fe_cyear <- 2014 - dattestOr$construction_year

# Train
dattrainOrlab$fe_dist <- geodist(dattrainOrlab$latitude, dattrainOrlab$longitude, 0, 0)

# Test
dattestOr$fe_dist <- geodist(dattestOr$latitude, dattestOr$longitude, 0, 0)

# Train
dattrainOrlab$fe_cant_agua <- ifelse(dattrainOrlab$population == 0,
                                     0,
                                     round(dattrainOrlab$amount_tsh /
                                             dattrainOrlab$population, 3)
)

# Test
dattestOr$fe_cant_agua <- ifelse(dattestOr$population == 0,
                                 0,
                                 round(dattestOr$amount_tsh /
                                         dattestOr$population, 3)
)

#-- date_recorded --> mes
#   Train
dattrainOrlab$fe_dr_month <- month(dattrainOr$date_recorded)

#   Test
dattestOr$fe_dr_month     <- month(dattestOr$date_recorded)

#-- date_recorded --> año date_recorded - año construction_date
#   Train
dattrainOrlab$fe_dr_year_cyear_diff <- year(dattrainOr$date_recorded) - dattrainOrlab$construction_year

#   Test
dattestOr$fe_dr_year_cyear_diff     <- year(dattestOr$date_recorded) - dattestOr$construction_year
dattrainOrlab$status_group <- NULL

datfinal <- rbind(dattrainOrlab, dattestOr[, names(dattestOr) %in% names(dattrainOrlab)])

dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

columnas_test  <- names(dattestOr)[names(dattestOr) %in% names(dattrainOrlab)]
datcompleto <- rbind(dattrainOrlab, dattestOr[, columnas_test])

fila_test <- which(datcompleto$id == 50785)

datcompleto$funder         <- c(dattrainOr$funder, dattestOr$funder)
datcompleto$ward           <- c(dattrainOr$ward, dattestOr$ward)
datcompleto$scheme_name    <- c(dattrainOr$scheme_name, dattestOr$scheme_name)
datcompleto$public_meeting <- c(dattrainOr$public_meeting, dattestOr$public_meeting)
datcompleto$permit         <- c(dattrainOr$permit, dattestOr$permit)

datcompleto <- as.data.table(datcompleto)

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

# 0.8168855
my_model <- fit_random_forest(formula, train)

my_sub <- make_predictions(my_model, test)
# 0.8251
fwrite(my_sub, file = "./submissions/final_submission.csv")



