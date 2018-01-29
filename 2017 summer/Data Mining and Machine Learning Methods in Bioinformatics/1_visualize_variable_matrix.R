# Illustrates the dataset mtcars in R
# Visualize the correlation between different variables

data(mtcars)
str(mtcars)
pairs(~ hp+mpg+disp+cyl,data=mtcars, main="Simple Scatterplot Matrix")
cor(mtcars, use="complete.obs", method="kendall") #correlation between all columns of mtcars
cor(subset(mtcars, select = c(hp,mpg,disp,cyl)),method="pearson") #Pearson correlation between selected columns of mtcars
cor(mtcars[,"mpg"],y=subset(mtcars, select = c(hp,disp,cyl)),method="pearson") # Pearson correlation between column "mpg" and other selected columns of mtcars


