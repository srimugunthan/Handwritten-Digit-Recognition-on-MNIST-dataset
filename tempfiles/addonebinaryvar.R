add_one_binary_var <- function() {
  #train <- data.frame(y2 = 0, train)
  (train$y2 <- rep(0, length(train$y)))
  for(i in 1:length(train$y))
  {
    if ((train$y[i] == 1)) {
      train$y2[i] <- 1
    } else {
      train$y2[i] <- 0
    }
  }
  #append(train, y2, after=0)
  #View(y2)
  
  return (train)
  
  
}