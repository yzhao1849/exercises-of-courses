data("airquality")
attach(airquality) #so objects in the database can be accessed by simply giving their names
par(mfrow=c(3,1))

# Plot the effect of Solar radiation, Wind and Temperature on the amount of Ozone, using scatterplots
plot(Solar.R,Ozone,main="Effect of Solar radiation on Ozone",xlab="Solar radiation", ylab="Ozone")
plot(Wind,Ozone,main="Effect of Wind on Ozone",xlab="Wind", ylab="Ozone")
plot(Temp,Ozone,main="Effect of Temperature on Ozone",xlab="Temperature", ylab="Ozone")

# Visualize the fluctuation of the Ozone concentration as a function of the time using bar-plot
par(mfrow=c(1,1))
barplot(Ozone,main = "Everyday Ozone Amount",xlab = "Time",ylab = "Ozone",names.arg = airquality$Month)

# Compute Pearson and Spearman correlation coefficients
cor(Ozone,use = "complete.obs",y=subset(airquality, select = c(Solar.R,Wind,Temp)),method="pearson")
cor(Ozone,use = "complete.obs",y=subset(airquality, select = c(Solar.R,Wind,Temp)),method="spearman")

# Calculate correlation matrix for each month
for(i in 5:9){
  cat("Month No.",i,":\n",sep = "")
  print(cor(subset(airquality,Month==i,select = c(Solar.R,Wind,Temp)),use = "complete.obs",method="pearson"))
  cat("\n")
}

# Visualize the correlation matrices calculated above via heatmaps
library(lattice) # to use levelplot in lattice
par(mfrow=c(3,2))
new_data=subset(airquality,Month==5,select = c(Solar.R,Wind,Temp))
levelplot(cor(new_data, use = "complete.obs", method = "pearson"),at=seq(-0.6, 1.0, 0.1),main="Month No.5")
new_data=subset(airquality,Month==6,select = c(Solar.R,Wind,Temp))
levelplot(cor(new_data, use = "complete.obs", method = "pearson"),at=seq(-0.6, 1.0, 0.1),main="Month No.6")
new_data=subset(airquality,Month==7,select = c(Solar.R,Wind,Temp))
levelplot(cor(new_data, use = "complete.obs", method = "pearson"),at=seq(-0.6, 1.0, 0.1),main="Month No.7")
new_data=subset(airquality,Month==8,select = c(Solar.R,Wind,Temp))
levelplot(cor(new_data, use = "complete.obs", method = "pearson"),at=seq(-0.6, 1.0, 0.1),main="Month No.8")
new_data=subset(airquality,Month==9,select = c(Solar.R,Wind,Temp))
levelplot(cor(new_data, use = "complete.obs", method = "pearson"),at=seq(-0.6, 1.0, 0.1),main="Month No.9")


detach(airquality)


