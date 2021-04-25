suppressPackageStartupMessages({
  library(dplyr)                # Manipulacion de datos 
  library(data.table)           # Lectura y escritura de ficheros
  library(ranger)               # randomForest (+ rapido que caret)
  library(forcats)              # Tratamiento de variables categoricas
  library(tictoc)               # Calculo de tiempo de ejecucion
  library(missRanger)           # Imputacion de valores NA
  library(knitr)                # Generacion de informes (formateo de tablas)
  library(gmt)                  # Calculo de la distancia geografica
  library(stringi)              # Tratamiento de strings
  library(missRanger)           # Tratamiento de valores missing (mediante random forest)
  library(xgboost)              # XGboost
})

# -----------------------------
# ----- funciones propias -----
# -----------------------------

#-- Funcion para limpieza de textos
#   1. Conversion a minusculas
#   2. Eliminacion de espacios en blanco
#   3. Eliminacion de signos de puntuacion
clean_text <- function(text) {
  stri_trans_tolower(
    stri_replace_all_regex(
      text, 
      pattern = "[ +\\p{Punct}]", 
      replacement = ""
    )
  )
}

#-- Funcion para entrenar un modelo XGboost, en base a los parametros proporcionados
fit_xgboost_model <- function(params, train, val, nrounds, early_stopping_rounds = 20, show_log_error = TRUE, seed = 1234) {
  set.seed(seed)
  my_model <- xgb.train(
    data   = train,
    params = params,
    watchlist=list(val1=val),
    verbose = 1,
    nrounds= nrounds,
    early_stopping_rounds = early_stopping_rounds,
    nthread=4
  )
  return(my_model)
}

#-- Funcion para realizar la prediccion de un modelo XGboost
make_predictions_xgboost <- function(my_model, test) {
  xgb_pred <- predict(my_model,as.matrix(test),reshape=T)
  xgb_pred <- ifelse(xgb_pred == 0, "functional", 
                     ifelse(xgb_pred == 1, 
                            "functional needs repair", 
                            "non functional"
                            )
                     )
  
  xgb_pred <- data.table(id = test$id, status_group = xgb_pred)
  return(xgb_pred)
}

# -----------------------------
# -------- script final -------
# -----------------------------

#---------------------- Carga de ficheros train y test ---------------------
dattrainOr    <- fread(file = "./data/train_values.csv", data.table = FALSE )
dattrainLabOr <- fread(file = "./data/train_labels.csv", data.table = FALSE )
dattestOr     <- fread(file = "./data/test_values.csv", data.table  = FALSE )

#-------------------------- Variables categoricas ---------------------------
datcat_df <- dattrainOr %>% select(where(is.character))

# Mediante un bucle for... 
numlev_df <- data.frame(
  "vars" = names(datcat_df),
  "levels" = apply(datcat_df, 2, 
                   function(x) length(unique(x)))
  
)

# Eliminamos los nombres de fila
rownames(numlev_df) <- NULL

kable(numlev_df %>% arrange(levels))

# Unimos dattrainOr con la variable objetivo
dattrainOrlab <- merge(
  dattrainOr, dattrainLabOr,
  by.x = c('id'), by.y = c('id'),
  sort = FALSE
)

#-- Eliminamos las variables no empleadas
dattrainOrlab$recorded_by    <- NULL; dattestOr$recorded_by    <- NULL
dattrainOrlab$payment_type   <- NULL; dattestOr$payment_type   <- NULL
dattrainOrlab$quantity_group <- NULL; dattestOr$quantity_group <- NULL
dattrainOrlab$installer      <- NULL; dattestOr$installer      <- NULL
dattrainOrlab$wpt_name       <- NULL; dattestOr$wpt_name       <- NULL
dattrainOrlab$subvillage     <- NULL; dattestOr$subvillage     <- NULL

vector_status_group <- dattrainOrlab$status_group

dattrainOrlab$status_group <- NULL
datcompleto <- as.data.table(rbind(dattrainOrlab, dattestOr))
#-- Guardamos el indice de fila donde comienza el conjunto "test",
#   concretamente la posicion 59401
fila_test <- which(datcompleto$id == 50785)

#----------------------------- Feature Engineering ------------------------------
#-- fe_cyear: 2014 - construction_year
datcompleto$fe_cyear     <- 2014 - datcompleto$construction_year

#-- fe_dist: geodist(latitude, longitude) al (0,0)
datcompleto$fe_dist <- geodist(datcompleto$latitude, datcompleto$longitude, 0, 0)

#-- fe_cant_agua: cantidad de agua / hab.
datcompleto$fe_cant_agua <- ifelse(datcompleto$population == 0,
                                         0,
                                         round(datcompleto$amount_tsh /
                                                 datcompleto$population, 3)
)

#-- fe_dr_year_cyear_diff: año date_recorded - construction_year
datcompleto$fe_dr_year_cyear_diff <- year(datcompleto$date_recorded) - datcompleto$construction_year

#-- month: mes date_recorded
datcompleto$fe_dr_month           <- month(datcompleto$date_recorded)

#-- Eliminamos date_recorded
datcompleto$date_recorded <- NULL

#-- Limpieza de variables categoricas mediante clean_text
cols <- c('funder', 'ward', 'scheme_name')
datcompleto[ , paste0('fe_',cols) := lapply(.SD, clean_text), .SDcols = cols]
rm(cols)

#-- lumping sobre la mediana de la proporcion de aparicion
#-  fe_funder
summary(c(prop.table(table(datcompleto[, fe_funder]))))
datcompleto[, fe_funder := fct_lump_prop(datcompleto[,fe_funder], 2e-05, other_level = "other")]
datcompleto$fe_funder <- as.character(datcompleto$fe_funder)

datcompleto[, funder := NULL]

#- fe_ward
summary(c(prop.table(table(datcompleto[, fe_ward]))))
datcompleto[, fe_ward := fct_lump_prop(datcompleto[,fe_ward], 4e-04, other_level = "other")]
datcompleto$fe_ward <- as.character(datcompleto$fe_ward)

datcompleto[, ward := NULL]

#- fe_scheme_name
summary(c(prop.table(table(datcompleto[, fe_scheme_name]))))
datcompleto[, fe_scheme_name := fct_lump_prop(datcompleto[,fe_scheme_name], 8.1e-05, other_level = "other")]
datcompleto$fe_scheme_name <- as.character(datcompleto$fe_scheme_name)

datcompleto[, scheme_name := NULL]

#-- Imputacion de las variables categoricas por sus frecuencias absolutas
cat_cols <- names(datcompleto[, which(sapply(datcompleto, is.character)), with = FALSE])

#-  Antes de imputar
freq_antes_fe <- apply(datcompleto[, ..cat_cols], 2, function(x) length(unique(x)))

for (cat_col in cat_cols) {
  datcompleto[, paste0("fe_", cat_col) := as.numeric(.N), by = cat_col]
}
names(datcompleto) <- stri_replace_all_fixed(names(datcompleto),
                                             "fe_fe_", "fe_")
#-- Eliminamos las variables originales
for (cat_col in cat_cols) {
  datcompleto[, paste(cat_col) := NULL]
}

new_cat_cols <- paste0("fe_", stri_replace_all_fixed(cat_cols, "fe_", ""))

#- Despues de imputar
freq_despues_fe <- apply(datcompleto[, ..new_cat_cols], 2, function(x) length(unique(x)))

print("ANTES DE IMPUTAR")
freq_antes_fe
print("------------------------------")
freq_despues_fe

#-- Variables logicas (debemos imputarlas, dado que en el conjunto test existen missings)
sum(is.na(dattestOr$permit))
sum(is.na(dattestOr$public_meeting))

datcompleto_aux <- fread("./data/datcompleto_imp_ap_15.csv")


cols_orden <- c(names(datcompleto)[c(1:9)],  "construction_year", "fe_cyear", 
                "fe_dist", "fe_cant_agua", "fe_dr_year_cyear_diff", "fe_dr_month", 
                "public_meeting", "permit", names(datcompleto)[c(18:38)])


setcolorder(datcompleto, cols_orden)

#-- Imputamos por missRanger
datcompleto_imp <- missRanger(datcompleto,
                              pmm.k = 5,
                              seed = 1234,
                              maxiter = 100)

# Comprobamos que las variables logicas estan imputadas
sum(is.na(datcompleto_imp))

# Las convertimos a variables numericas
datcompleto_imp[, public_meeting := as.numeric(public_meeting)]
datcompleto_imp[, permit := as.numeric(permit)]

#-- Creamos dos columnas adicionales que indiquen si la variable logica era o no NA:
#   is_na_public_meeting
#   is_na_permit
datcompleto_imp[, is_na_public_meeting := ifelse(is.na(datcompleto[, public_meeting]), 1, 0)]
datcompleto_imp[, is_na_permit := ifelse(is.na(datcompleto[, permit]), 1, 0)]
#---------------------------- Fin Feature Engineering -----------------------------

#------------------------------------ Modelo --------------------------------------
formula <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]

test  <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

# Transformamos la variable objetivo en numerica
vector_status_group <- ifelse(vector_status_group == "functional", 0
                              , ifelse(vector_status_group == "functional needs repair", 1
                              , 2))
xgb.train <- xgb.DMatrix(data=as.matrix(train), label=vector_status_group)


params = list(
  objective = "multi:softmax",
  num_class = 3,
  colsample_bytree  = 0.3,
  colsample_bylevel = 0.9,
  colsample_bynode  = 0.7,
  max_depth         = 15,
  eta               = 0.02
)

for (seed in c(1234)) {
  tic()
  my_model <- fit_xgboost_model(params, train = xgb.train, val = xgb.train, nrounds = 600, seed = seed)
  toc()
  
  # accuracy = 1 - mlogloss
  accuracy <- 1 - tail(my_model$evaluation_log$val1_mlogloss, 1)
  accuracy
  
  xgb_pred <- make_predictions_xgboost(my_model, test)
  
  # Guardamos la submission
  fwrite(xgb_pred, file = paste0("./seeds/",seed,".csv"))
}
