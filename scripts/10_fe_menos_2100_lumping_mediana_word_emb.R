#-------------------
# Autor: Alberto Fernandez
# Fecha: 2021_03_21
# Inputs: Datos 07_fe_menos_2100_lumping_mediana_freq_abs_categoricas.R
# Salida: Aplicando freq abs a las variables categoricas salvo funder y ward, donde aplicaremos word embedding
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
  library(embed)                # Creacion de modelos embeddings
  library(doParallel)           # Paralelizacion de funciones
  library(recipes)              # Preprocesamiento de datasets (implementacion de modelos fe)
  
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

#-- Imputacion de las variables categoricas por sus frecuencias absolutas, salvo funder y ward
cat_cols <- names(datcompleto[, which(sapply(datcompleto, is.character)), with = FALSE])[c(-19,-20)]

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

#-- Embeddings
#-- Aplicamos word embedding sobre funder y ward

#   1. Comprobamos que las variables categoricas estan codificadas como factor
emb_cols <- c("fe_funder", "fe_ward")
datcompleto[,(emb_cols):= lapply(.SD, as.factor), .SDcols = emb_cols]

train <- datcompleto[c(1:fila_test-1),]
train$status_group <- vector_status_group
train$status_group <- as.factor(train$status_group)

test <- datcompleto[c(fila_test:nrow(datcompleto)),]

# 2. Elaboramos el modelo embedded (empezando con dimensionalidad 2)
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
train_prepped <- prep(base_recipe, train[, ..emb_cols_target])
test_prepped <-  bake(train_prepped, test[, ..emb_cols])

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

# Con 2 terminos:  0.8161111
# Con 5 terminos:  0.8142593
# Con 10 terminos: 0.8142424
my_model_14 <- fit_random_forest(formula,
                                 train_final)

my_sub_14 <- make_predictions(my_model_14, test_final)
# guardo submission
fwrite(my_sub_14, file = "./submissions/14_lumping_fe_sobre_funder_ward_y_word_embed_dim2_solo_funder_ward_resto_freq_abs.csv")
# Con 2 terminos: 0.8207
# Con 5 terminos: 0.8195
# Con 10 terminos: 0.8168

#-- Conclusion: aplicando frecuencias absolutas sobre las variables cartegoricas ha permitido mejorar considerablemente el modelo
#--- Pintar importancia de variables
impor_df <- as.data.frame(my_model_14$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

knitr::kable(data.frame("Train accuracy" = c('-', 0.8149832, 0.8159764, 0.8146633, 0.8159259, 0.8160774, 0.8154882, 0.8157071, 0.8086364,
                                             0.8152694, 0.8161111), 
                        "Data Submission" = c(0.8180, 0.8197, 0.8212, 0.8203, 0.8196, 0.8213, 0.8216, 0.8226, 0.8110,
                                              0.8185, 0.8207),
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
                                      "Num + Cat (> 1 & < 2100) fe anteriores + lumping + word embed sobre funder y ward (dim. 2) + freq. abs. cat.")),
             align = 'c')

# A la vista de los resultados obtenidos, aplicar word embedding sobre categorias con poca representatividad mejora el modelo con respecto al target encoding
# pero sigue sin mejorar ante la codificacion por frecuencias.
#-- Conclusion: nos quedamos con la codificacion de las variables categoricas por frecuencias absolutas

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/14_lumping_fe_sobre_funder_ward_y_word_embed_dim2_solo_funder_ward_resto_freq_abs.png')
