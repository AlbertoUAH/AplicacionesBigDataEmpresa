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
  library(vtreat)
  
  source("scripts/funciones.R") # Funciones propias
})

ncores <- parallel::detectCores()
parallelCluster <- parallel::makeCluster(ncores)

#-- Leo ficheros
dattrainOr       <- fread(file = "./data/train_values.csv", data.table = FALSE)
dattrainLab      <- fread(file = "./data/train_labels.csv", data.table = FALSE)
dattestOr        <- fread(file = "./data/test_values.csv", data.table = FALSE)

dattrainOrlab <- merge(
  dattrainOr, dattrainLab,
  by.x = c('id'), by.y = c('id'),
  sort = FALSE
)

target  <- "status_group"
varCols <- setdiff(names(dattrainOrlab), target)

#-- Preparacion de ambos conjuntos
# Dividimos los datos en varios clusters
tic()
# minFraction: ¿Que frecuencia minima debe tener una categoria para convertirse en columna?
# rareCount:   ¿Que frecuencia minima debe tener una categoria o categorias para ser agrupadas en torno
#              a una categoria unica?
cd <- mkCrossFrameCExperiment(dattrainOrlab,varCols,target,"non functional",
                              parallelCluster=parallelCluster, minFraction = 0.25,
                              rareCount = 10)
toc()

scoreFrame   <- cd$treatments$scoreFrame

scoretreatments  <- cd$treatments$scoreFrame
scoreFrame       <- scoretreatments[scoretreatments$recommended == TRUE, ]

train_nuevo  <- cd$crossFrame

nuevas_variables <- scoreFrame$varName[scoreFrame$sig<1/nrow(scoreFrame)]
test_nuevo       <- prepare(cd$treatments,dattestOr)

#-- Modelo
train_nuevo$status_group <- as.factor(train_nuevo$status_group)

formula   <- as.formula("status_group~.")

# Si priorizamos non functional:          0.8079293
# Si modificamos los parametros:          0.8043434
my_model_17 <- fit_random_forest(formula, train_nuevo)

my_sub_17 <- make_predictions(my_model_17, test_nuevo)

fwrite(my_sub_17, file = "./submissions/17_empleando_vtreat.csv")
# Si priorizamos functional needs repair: 0.8188
# Si modificamos los parametros:          0.8191

impor_df <- as.data.frame(my_model_17$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/17_empleando_vtreat.png')


