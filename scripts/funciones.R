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
    num.trees  = num_trees,
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
  my_pred <- predict(model, test_data, type = "response")
  
  # Submission
  my_sub <- data.table(
    id = test_data[, "id"],
    status_group = my_pred$predictions
  )
  
  return(my_sub)
}

# Leave One Out Encoding
encode_leave_one_out <- function(x, y) {
  n <- length(x)
  x[is.na(x)] <- "__MISSING"
  x2 <- vapply(1:n, function(i) {
    xval <- x[i]
    yloo <- y[-i]
    xloo <- x[-i]
    yloo <- yloo[xloo == xval]
    mean(yloo, na.rm = TRUE)
  }, numeric(1))
  x2
  print("FINISHED")
}

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

make_predictions_xgboost <- function(my_model, test) {
  xgb_pred <- predict(my_model,as.matrix(test),reshape=T)
  xgb_pred <- ifelse(xgb_pred == 0, "functional", ifelse(xgb_pred == 1, "functional needs repair", "non functional"))
  
  xgb_pred <- data.table(id = test$id, status_group = xgb_pred)
  return(xgb_pred)
}



