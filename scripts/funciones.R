clean_text <- function(text) {
  stri_trans_tolower(
    stri_replace_all_regex(
      text, 
      pattern = "[ +\\p{Punct}]", 
      replacement = ""
      )
    )
}

# Funcion que devuelve un modelo Random Forest entrenado
fit_random_forest <- function(formula, data, num_trees = 500, mtry = NULL, seed = 1234) {
  tic()
  my_model <- ranger( 
    formula, 
    importance = 'impurity',
    data       = data,
    num.trees = num_trees,
    mtry = mtry,
    verbose = FALSE,
    seed = seed
  )
  # **Estimacion** del error / acierto **esperado**
  success <- 1 - my_model$prediction.error
  print(success)
  toc()
  
  return(my_model)
}

make_predictions <- function(model, test_data) {
  # Prediccion
  my_pred <- predict(model, test_data)
  
  # Submission
  my_sub <- data.table(
    id = test_data[, "id"],
    status_group = my_pred$predictions
  )
  
  return(my_sub)
}







