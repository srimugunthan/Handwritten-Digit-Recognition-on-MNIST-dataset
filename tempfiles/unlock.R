# e <- new.env()
# environmentIsLocked(e) # FALSE
# e$x <- 1
# lockEnvironment(e)
# environmentIsLocked(e) # TRUE
# e$x <- 2 # Can still modify existing binding
# #e$y <- 1 # Error: cannot add bindings to a locked environment
# #remove("x", envir=e) # Error: cannot remove bindings from a locked environment
# bindingIsLocked("x", e) # FALSE
# lockBinding("x", e)
# bindingIsLocked("x", e) # TRUE
# #e$x <- 1 # Error:  cannot change value of locked binding for 'x'
# 
# ee <- new.env()
# ee$xx <- 1
# ee$yy <- 1
# lockBinding("xx", ee)
# lockBinding("yy", ee)
# #ee$xx <- 2 # Error:  cannot change value of locked binding for 'xx'
# remove("xx", envir = ee) # Can still remove a locked binding from a unlocked environment.
# unlockBinding("yy", ee)
# ee$yy <- 2 # OK.
# envirs <- ls()[sapply(ls(), function(x) is.environment(get(x)))]
# 
# #sapply(envirs, function(e) 'x' %in% ls(envir=get(e)))
# 
# getEnv <- function(x) {
#   xobj <- deparse(substitute(x))
#   #gobjects <- ls(envir=.GlobalEnv)
#   gobjects <- ls(.GlobalEnv)
#   envirs <- gobjects[sapply(gobjects, function(x) is.environment(get(x)))]
#   envirs <- c('.GlobalEnv', envirs)
#   xin <- sapply(envirs, function(e) xobj %in% ls(envir=get(e)))
#   envirs[xin] 
# }
# 
# unlockBinding(train, getEnv(train))
# 
# 
# # # active bindings
# # eee <- new.env()
# # f <- local( {
# #   x <- 0
# #   function(y) {
# #     if (missing(y)) {
# #       cat("get value\n")
# #       x
# #     } else {
# #       cat("set value\n")
# #       x <<- as.numeric(y)
# #     }
# #   }
# # })
# # makeActiveBinding("num", f, eee) 
# # bindingIsActive("num", eee) # TRUE
# # eee$num
# # eee$num <- "1" # coerced to numeric
# # eee$num # 1 (numeric)
# # 
# # eee$x <- 1
# # bindingIsActive("x", eee) # FALSE
# #makeActiveBinding("x", f, eee) # Error: symbol 'x' already has a regular binding

getEnvOf <- function(what, which=rev(sys.parents())) {
  for (frame in which)
    if (exists(what, frame=frame, inherits=FALSE)) 
      return(sys.frame(frame))
  return(NULL)
}
print(getEnvOf("train"))
