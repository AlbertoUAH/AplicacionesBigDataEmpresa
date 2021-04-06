#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_05
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Tuneo random forest
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
  library(embed)                # Creacion de embeddings
  library(doParallel)           # Paralelizacion de funciones
  library(missRanger)           # Tratamiento de valores missing (mediante random forest)
  library(mice)                 # Tratamiento de valores missing (mediante regresion logistica)
  library(lubridate)            # Tratamiento de fechas
  library(gmt)                  # Calculo distancia entre dos coordenadas geograficas
  library(dataPreparation)      # Preparacion de datasets
  
  source("scripts/funciones.R") # Funciones propias
})

#-- Leemos el fichero con los datos completos imputados
datcompleto_imp <- fread("./data/datcompleto_imp_ap_15.csv" )
names(datcompleto_imp)[14] <- "fe_dr_year_cyear_diff"
dattrainOrlab    <- fread(file = "./data/train_values_concurso.csv", data.table = FALSE )
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE )
dattestOr        <- fread(file = "./data/test_values_concurso.csv", data.table = FALSE  )

vector_status_group <- dattrainOrlab$status_group
dattrainOrlab$status_group <- NULL

#-- Para contrastar los resultados del mejor modelo hasta el momento
mejor_resultado  <- fread("./submissions/19_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_y_permit_public_meeting.csv")

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto_imp$id == 50785)

formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

#-- Mejor accuracy hasta el momento: 0.8251

#-- Veamos hasta que numero de arboles se estabiliza el error
n_trees           <- c(500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000)
prediction_errors <- c()
for (n_tree in n_trees) {
  prediction_error  <- fit_random_forest(formula, train, num_trees = n_tree)$prediction.error
  prediction_errors <- c(prediction_errors, prediction_error)
}
rm(n_tree); rm(prediction_error)
names(prediction_errors) <- n_trees
prediction_errors_dt <- data.table(ntrees = as.numeric(names(prediction_errors)), error = prediction_errors)
ggplot(prediction_errors_dt, aes(x = ntrees, y = error, group = 1, col = "red")) + geom_line()

mtrys  <- c(3, 4, 5, 6, 7, 8)
prediction_errors_mtrys_2 <- c()
for (mtry in mtrys) {
  prediction_error  <- fit_random_forest(formula, train, num_trees = 2000, mtry = mtry)$prediction.error
  prediction_errors_mtrys_2 <- c(prediction_errors_mtrys_2, prediction_error)
}
rm(mtry)
names(prediction_errors_mtrys_2) <- mtrys

#-- Aparentemente, el error mas bajo se alcanza con 800 o 2000 arboles
#   Con 800 arboles                      : 0.8176936
#   O ntree 2000                         : 0.8176768
my_model_22 <- fit_random_forest(formula, train, num_trees = 800)
my_sub_22   <- make_predictions(my_model_22, test)

my_model_23 <- fit_random_forest(formula, train, num_trees = 800, mtry = 7)
my_sub_23   <- make_predictions(my_model_23, test)

fwrite(my_sub_22, file = "./submissions/tunning_models/random_forest/22_rf_ntree_800_mtry_original.csv")
fwrite(my_sub_23, file = "./submissions/tunning_models/random_forest/23_rf_ntree_800_mtry_7.csv")

#-- Aumentar demasiado el numero de arboles no parece mejorar demasiado el modelo
#   Reducimos el numero de arboles
n_trees           <- c(500, 550, 600, 650, 700, 750, 800, 850)
prediction_errors_menos_900 <- c()
for (n_tree in n_trees) {
  prediction_error  <- fit_random_forest(formula, train, num_trees = n_tree)$prediction.error
  prediction_errors_menos_900 <- c(prediction_errors_menos_900, prediction_error)
}
rm(n_tree); rm(prediction_error)
names(prediction_errors_menos_900) <- n_trees
prediction_errors_menos_900_dt <- data.table(ntrees = as.numeric(names(prediction_errors_menos_900)), 
                                             error = prediction_errors_menos_900)
ggplot(prediction_errors_menos_900_dt, aes(x = ntrees, y = error, group = 1, col = "red")) + geom_line()

#-- Probamos desde 550 hasta 850 arboles
cont <- 24
for (n_tree in n_trees) {
  submission_name <- paste0("./submissions/tunning_models/random_forest/",cont,"_rf_ntree_",n_tree,"_mtry_original.csv")
  my_model        <- fit_random_forest(formula, train, num_trees = n_tree)
  my_sub          <- make_predictions(my_model, test)
  cont            <- cont + 1
  fwrite(my_sub, file = submission_name)
}
rm(cont); rm(my_model); rm(my_sub)
#   Con 500 arboles                      : 0.8168855
#   Con 550 arboles                      : 0.8171212
#   Con 600 arboles                      : 0.8174411
#   Con 650 arboles                      : 0.8174579
#   Con 700 arboles                      : 0.8173906
#   Con 750 arboles                      : 0.8176263
#   Con 800 arboles                      : 0.8176936
#   Con 850 arboles                      : 0.8177778

#-- ¿Y si reducimos el numero de arboles?
cont <- 32
for (n_tree in c(300, 400)) {
  submission_name <- paste0("./submissions/tunning_models/random_forest/",cont,"_rf_ntree_",n_tree,"_mtry_original.csv")
  my_model        <- fit_random_forest(formula, train, num_trees = n_tree)
  my_sub          <- make_predictions(my_model, test)
  cont            <- cont + 1
  fwrite(my_sub, file = submission_name)
}
rm(cont); rm(my_model); rm(my_sub)
#  Con  300           : 0.816431
#  Con  400           : 0.8168519

#-- Datos test:
# ntree 300           : 0.8248
# ntree 400           : 0.8252
# ntree 500           : 0.8253 (por la semilla)
# ntree 550           : 0.8247
# ntree 600           : 0.8244
# ntree 650           : 0.8242
# ntree 700           : 0.8246
# ntree 750           : 0.8242
# ntree 800           : 0.8242
# ntree 850           : 0.8242
# ntree 1500          : 0.8246
# ntree 2000          : 0.8249

#-- Aparentemente, con 500 arboles ya obtenemos un buen resultado (0.8253) ¿Y si variamos el mtry?
#-- Por defecto es sqrt(40) ~ 6 variables
cont <- 34
mtrys  <- c(3, 4, 5, 6, 7, 8)
for (mtry in mtrys) {
  submission_name <- paste0("./submissions/tunning_models/random_forest/",cont,"_rf_ntree_",n_tree,"_mtry_",mtry,".csv")
  my_model        <- fit_random_forest(formula, train, num_trees = 500, mtry = mtry)
  my_sub          <- make_predictions(my_model, test)
  cont            <- cont + 1
  fwrite(my_sub, file = submission_name)
}
rm(mtry); rm(my_model); rm(my_sub)

#-- Train
#   ntree = 500; mtry = 3 -> 0.8112626
#   ntree = 500; mtry = 4 -> 0.815202
#   ntree = 500; mtry = 5 -> 0.8168182
#   ntree = 500; mtry = 7 -> 0.816633
#   ntree = 500; mtry = 8 -> 0.8167845

#-- Datos test:
#   ntree = 500; mtry = 3 -> 0.8196
#   ntree = 500; mtry = 4 -> 0.8213
#   ntree = 500; mtry = 5 -> 0.8239
#   ntree = 500; mtry = 6 -> 0.8251
#   ntree = 500; mtry = 7 -> 0.8241
#   ntree = 500; mtry = 8 -> 0.8241


#-- Observamos que el punto optimo es ntree = 500 y mtry = 6

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616, 0.8164478, 0.8165657, 0.8167508, 0.8168855, 0.8168855, 0.8167508,
                                             0.8168855), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239, 0.8226, 0.8222, 0.8223, 0.8251, 0.8248, 0.8227, 0.8253),
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
                                      "Num + Cat (> 1 & < 60000) fe anteriores + lumping + freq. abs. sobre categoricas",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas + fe date_recorded",
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas + imp. anomalos",
                                      "Random Forest (ntree = 500; mtry = 6)")),
             align = 'c')

