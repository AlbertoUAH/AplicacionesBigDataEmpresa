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

# El conjunto test empieza a partir de la 59401
fila_test <- which(datcompleto_imp$id == 50785)

formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

colors <- c("Train" = "darkred", "Submission" = "darkblue")

cl <- makeCluster(detectCores())
registerDoParallel(cl)

#-- Mejor accuracy hasta el momento: 0.8251

#-- Veamos hasta que numero de arboles mejora el modelo
n_trees           <- c(400, 500, 600, 700, 800, 900)
prediction_errors <- c()
for (n_tree in n_trees) {
  rf  <- fit_random_forest(formula, train, num_trees = n_tree)
  prediction_error <- rf$prediction.error
  prediction_errors <- c(prediction_errors, 1 - prediction_error)
  submission <- make_predictions(rf, test)
  fwrite(submission, paste0("./submissions/tunning_models/random_forest/ntrees/random_forest_with_",n_tree,"_ntrees.csv"))
}
rm(n_tree); rm(prediction_error); rm(rf); rm(submission)
names(prediction_errors) <- n_trees
prediction_errors_dt <- data.table(ntrees = rep(seq(400,800, 100), 2), 
                                   Accuracy = c(0.8169, 0.8169, 0.8174, 0.8174, 0.8177, 0.8252, 0.8253, 0.8241, 0.8246, 0.8244),
                                   Tipo =c(rep("Train", 5), rep("Submission", 5)))

ggplot(prediction_errors_dt, aes(x = ntrees, y = Accuracy, colour = Tipo)) + geom_point() + geom_line() + 
  geom_label_repel(data = prediction_errors_dt[prediction_errors_dt$ntrees <= 500, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) + 
  theme(axis.text = element_text(face = "bold", colour = "black", size = 11)) +
   ggtitle("Accuracy del modelo en funcion de ntree")
# Con 400-500 arboles se ha obtenido (por azar y reproducibilidad de la semilla, un score de 0.8253)
ggsave("./charts/random_forest_seleccion_ntree.png")

# Con 400-500 arboles podemos tunear mtry
prediction_errors_2 <- c()
combinaciones <- expand.grid(c(400, 500), c(9))
for (fila in 1:nrow(combinaciones)) {
  rf  <- fit_random_forest(formula, train, num_trees = combinaciones[fila, "Var1"], mtry = combinaciones[fila, "Var2"])
  prediction_error <- rf$prediction.error
  prediction_errors_2 <- c(prediction_errors_2, 1 - prediction_error)
  submission <- make_predictions(rf, test)
  fwrite(submission, paste0("./submissions/tunning_models/random_forest/mtry/random_forest_with_",combinaciones[fila, "Var1"],"_ntrees_mtry_",
                            combinaciones[fila, "Var2"],".csv"))
}
rm(combinaciones); rm(fila); rm(prediction_error); rm(rf); rm(submission)
prediction_errors_dt <- data.table(ntrees = rep(c(400, 500), 10), mtry = rep(c(4,4,5,5,6,6,7,7,8,8), 2),
                                   Accuracy = c(0.8147,0.8152,0.8167,0.8168,0.8169,0.8169,0.8164,0.8166,0.8162,0.8168, 0.8211, 0.8213, 0.8224, 0.8213, 
                                                0.8252, 0.8253, 0.8230, 0.8242, 0.8228, 0.8234),
                                   Tipo =c(rep("Train", 10), rep("Submission", 10)))

# 400 + 3: 0.8192 ; 400 + 4: 0.8211 ; 400 + 5: 0.8224 ; 400 + 6: 0.8252 ; 400 + 7:  0.8230 ; 400 + 8: 0.8228
# 500 + 3: 0.8196 ; 500 + 4: 0.8213 ; 500 + 5: 0.8213 ; 500 + 6: 0.8253 ; 500 + 7:  0.8242 ; 500 + 8: 0.8234

ggplot(prediction_errors_dt[prediction_errors_dt$ntrees == 400, ], aes(x = factor(mtry), y = Accuracy, group=Tipo, colour = Tipo)) + geom_point() + 
    geom_line() + geom_label_repel(data = prediction_errors_dt[prediction_errors_dt$mtry == 6 & prediction_errors_dt$ntrees == 400, ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) + 
  theme(axis.text = element_text(face = "bold", colour = "black", size = 11)) +
  ggtitle("Accuracy del modelo en funcion de mtry (ntree = 400)")
ggsave("./charts/random_forest_seleccion_ntree400_mtry_tunning.png")

ggplot(prediction_errors_dt[prediction_errors_dt$ntrees == 500, ], aes(x = factor(mtry), y = Accuracy, colour = Tipo, group=Tipo)) + geom_point() + 
  geom_line() + geom_label_repel(data = prediction_errors_dt[prediction_errors_dt$mtry == 6 & prediction_errors_dt$ntrees == 500, ],
                                 aes(y = Accuracy, label = Accuracy), show.legend = FALSE) + 
  theme(axis.text = element_text(face = "bold", colour = "black", size = 11)) +
  ggtitle("Accuracy del modelo en funcion de mtry (ntree = 500)")
ggsave("./charts/random_forest_seleccion_ntree500_mtry_tunning.png")

#-- Tuneo final con semillas aleatorias
prediction_errors_3 <- c()
semillas <- c(1, 120, 2500, 6000, 56000, 650000)
for (semilla in semillas) {
  rf  <- fit_random_forest(formula, train, num_trees = 500, mtry = 6, seed = semilla)
  prediction_error <- rf$prediction.error
  prediction_errors_3 <- c(prediction_errors_3, 1 - prediction_error)
  submission <- make_predictions(rf, test)
  fwrite(submission, paste0("./submissions/tunning_models/random_forest/seeds/random_forest_with_seed_",semilla,".csv"))
}
rm(semillas); rm(semilla); rm(prediction_error); rm(rf); rm(submission)

# seed: 1 - 0.8247; seed: 120: 0.8241; seed: 2500: 0.8243; seed: 6000: 0.8238;
# seed: 56000: 0.8231; seed: 56000: 0.8231; seed: 650000: 0.8239

#-- Veamos hasta que numero de arboles mejora el modelo
max_depth_vector           <- c(200, 150, 80, 50)
prediction_errors <- c()
for (max_depth in max_depth_vector) {
  rf  <- fit_random_forest(formula, train, num_trees = 500, mtry = 6, max_depth = max_depth, seed = 1234)
  prediction_error <- rf$prediction.error
  prediction_errors <- c(prediction_errors, 1 - prediction_error)
  submission <- make_predictions(rf, test)
  fwrite(submission, paste0("./submissions/tunning_models/random_forest/max_depth/random_forest_with_",max_depth,"_max_depth.csv"))
}
rm(n_tree); rm(prediction_error); rm(rf); rm(submission)

# 0.8251 con 200
# 0.8251 con 150
# 0.8253 con 80
# 0.8253 con 50
# 0.8233 con 20
# 0.8116 con 15
# 0.7671 con 10
# 0.7488 con 8
# 0.7310 con 6
# 0.6636 con 3

prediction_errors_dt <- data.table(max_depth = c(10,15,20,50,80,150,200),
                                   Accuracy = c(0.7671, 0.8116, 0.8233,
                                                0.8253, 0.8253, 0.8251, 0.8251),
                                   col = rep("red", 7))

ggplot(prediction_errors_dt, aes(x = factor(max_depth), y = Accuracy, group = col, col = col)) + geom_point() + 
  geom_line() + geom_label_repel(data = prediction_errors_dt[prediction_errors_dt$max_depth %in% c(50,80), ], aes(y = Accuracy, label = Accuracy), show.legend = FALSE) +
  ggtitle("Accuracy del modelo en funcion de max_depth (solo test)") + theme(legend.position="none",  axis.text = element_text(face = "bold", colour = "black", size = 11))
ggsave("./charts/random_forest_seleccion_max_depth.png")



