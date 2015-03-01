dat=data.frame(x=c(1,2,3,4,5,6), 
               y=c(1,3,5,6,8,12))

min.RSS <- function(data, par) {
  with(data, sum((par[1] + par[2] * x - y)^2))
}

result <- optim(par = c(0, 1), min.RSS, data = dat)
print(result)
plot(y ~ x, data = dat)
abline(a = result$par[1], b = result$par[2], col = "red")
print(lm(y ~ x, data = dat))