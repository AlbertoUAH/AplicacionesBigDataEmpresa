#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 05_fe_menos_2100_lumping_mediana_freq_abs.R
# Salida: Datos con nuevas variables (incluyendo categorias < 2100) + lumping + transformacion fe_funder, fe_ward y lga
#         1. Creacion de un modelo embedded
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

#-- Nos traemos funder, ward (menos de 2100 categorias)
dattrainOrlab$funder <- dattrainOr$funder
dattrainOrlab$ward <- dattrainOr$ward

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
  print(numlev_dt)
}
names(numlev_dt) <- c('vars', 'levels')
numlev_dt %>% arrange(levels)


# ¿Categorias con muchas categorias pero algunas presentan pocas observaciones?
datcompleto[, fe_funder := clean_text(funder)][, fe_ward := clean_text(ward)]

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

#-- Embeddings
#   Emplearemos las variables con menos de 30 categorias diferentes

#   1. Comprobamos que las variables categoricas estan codificadas como factor
emb_cols <- colnames(datcompleto)[which(as.vector(datcompleto[,lapply(.SD, function(x) length(unique(x)))]) < 50)]
datcompleto[,(emb_cols):= lapply(.SD, as.factor), .SDcols = emb_cols]

train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

# 2. Elaboramos el modelo embedded
emb_cols_target <- c(emb_cols, "status_group")
base_recipe <- recipes::recipe(status_group ~ ., train[, ..emb_cols_target])
for(feat in emb_cols){
  base_recipe <- base_recipe %>% 
    embed::step_embed({{feat}},
                      num_terms = 2,
                      outcome = vars(status_group),
                      options = embed_control(epochs = 5, validation_split = 0.2)
    )
}
base_recipe

# 4. Creacion del modelo embedded
train_prepped <- recipes::prep(base_recipe, train[, ..emb_cols_target])
test_prepped <- recipes::bake(train_prepped, test[, ..emb_cols])

train_final  <- cbind(as.data.table(juice(train_prepped)),
                      train[, setdiff(names(train), emb_cols_target), with = FALSE]
                )

test_final  <- cbind(test_prepped,
                     test[, setdiff(names(test), emb_cols), with = FALSE]
)

#-- Modelo
formula   <- as.formula("status_group~.")

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Con 2 terminos: 0.8155556
# Con 5 terminos  0.8128451
# Con 10 terminos: 0.8092593
my_model_12 <- fit_random_forest(formula,
                                 train_final)

my_sub_12 <- make_predictions(my_model_12, test_final)
# guardo submission
fwrite(my_sub_12, file = "./submissions/12_05_lumping_sobre_funder_ward_word_embedding_2_dim.csv")
# Con 2 terminos: 0.8191
# Con 5 terminos: 0.8128
# Con 10 terminos: 0.8164

#-- Conclusion: aplicando frecuencias absolutas sobre las variables cartegoricas ha permitido mejorar considerablemente el modelo
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_12$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8162121, 0.8154882, 0.8157071, 0.808771, 0.7995118,
                                             0.8155556), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8213, 0.8203, 0.8198, 0.8216, 0.8226, 0.8118, 0.8056, 0.8191),
                        row.names = c("Mejor accuracy en el concurso",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + fe_funder + fe_ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (tercer cuartil)",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder + ward (mediana) + hashed",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping (mediana) + freq. abs. sobre funder + ward",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + freq. abs. categoricas",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + target enc.",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + target enc. sin vars. originales",
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping sobre funder y ward (mediana) + word emb. < 50")),
             align = 'c')

# A la vista de los resultados obtenidos, aplicar word embedding sobre categorias con poca representatividad mejora el modelo con respecto al target encoding
# pero sigue sin mejorar ante la codificacion por frecuencias

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/12_05_lumping_sobre_funder_ward_word_embedding_2_dim.png')













