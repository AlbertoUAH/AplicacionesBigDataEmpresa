#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_04_04
# Inputs: Datos 15_fe_menos_10000_lumping_mediana_freq_abs_categoricas_mas_logicas.R
# Salida: Feature Engineering sobre date recorded
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

#-- Tratamiento de fechas
# A simple vista, 2004 fue el unico año en el que hubo mas bombas no funcionales que funcionales
table(year(dattrainOr$date_recorded), vector_status_group)
date_recorded <- c(dattrainOr$date_recorded, dattestOr$date_recorded)
fecha_referencia <- max(ymd(date_recorded)) # Tenemos 2013-12-03

#-- Si incluimos el año en formato ymd (y añadimos una fecha de referencia)
datcompleto_imp$fe_dr_day_count <- as.numeric(fecha_referencia - ymd(date_recorded))

#-- Si incluimos el año
# year          <- c(year(dattrainOr$date_recorded), year(dattestOr$date_recorded))
# datcompleto_imp$fe_dr_year <- year

#-- Si incluimos la fecha de referencia en meses (mayor importancia en el arbol que los años)
# datcompleto_imp$fe_dr_month_count <- time_length(interval(ymd(date_recorded), fecha_referencia), unit = "months")

#-- Si incluimos otros campos, mediante el paquete lubridate
#datcompleto_imp$fe_dr_day  <- day(date_recorded)        # Dia
#datcompleto_imp$fe_dr_wday <- wday(date_recorded)       # Dia de la semana
#datcompleto_imp$fe_dr_qday <- qday(date_recorded)       # Dia del cuatrimestre
#datcompleto_imp$fe_dr_week <- week(date_recorded)       # Semana
#datcompleto_imp$fe_dr_quarter <- quarter(date_recorded) # Cuatrimestre

#-- Si aplicamos una transformacion seno-coseno a los meses
# datcompleto_imp[, fe_dr_month_sin := sin(2*pi*fe_dr_month / 12)]
# datcompleto_imp[, fe_dr_month_cos := cos(2*pi*fe_dr_month / 12)]
# datcompleto_imp[, fe_dr_month := NULL]
# 
# ggplot(datcompleto_imp, aes(x = fe_dr_month_sin, y = fe_dr_month_cos)) + geom_point()

#-- Modelo
formula   <- as.formula("status_group~.")

train <- datcompleto_imp[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto_imp[c(fila_test:nrow(datcompleto_imp)),]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Incluyendo day_count       : 0.8168855
# Incluyendo month_count     : 0.8168687
# Incluyendo day_count y year: 0.8175421
# Incluyendo otros campos    : 0.8170202
# qday + day + day_count     : 0.8176768
# igual pero sin dr_month    : 0.8170875
# month sin cos transform    : 0.8172391
my_model_20 <- fit_random_forest(formula, train)

my_sub_20 <- make_predictions(my_model_20, test)
# guardo submission
fwrite(my_sub_20, file = "./submissions/temp/20_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_mas_logicas_fe_month_sin_cos_transformation.csv")
# Incluyendo day_count        : 0.8248
# Incluyendo month_count      : 0.8248
# Incluyendo day_count y year : 0.8244
# Incluyendo otros campos     : 0.8224
# qday y day parecen ser variables con alta importancia, ¿Y si las incluimos junto con day_count?
# qday + day + day_count      : 0.8228
# igual pero sin dr_month     : 0.8228
# con sin cos transformation  : 0.8224

#-- Importancia variables
impor_df <- as.data.frame(my_model_20$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111, 0.8161616, 0.8164478, 0.8165657, 0.8167508, 0.8168855, 0.8168855), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207, 0.8239, 0.8226, 0.8222, 0.8223, 0.8251, 0.8248),
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
                                      "Num + Cat (> 1 & < 10000) fe anteriores + lumping + freq. abs. sobre categoricas + vars. logicas + fe date_recorded")),
             align = 'c')

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/20_lumping_fe_freq_abs_sobre_funder_ward_scheme_name_resto_categoricas_mas_logicas_fe_month_sin_cos_transformation.png')

#-- Conclusion: podemos observar como añadir mas variables relacionadas con date_recorded no mejora el modelo














