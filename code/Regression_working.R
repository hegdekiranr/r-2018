packages <- c('here', 'coefplot', 'useful', 'glmnet', 'ggplot2', 'magrittr')
#Below  loads a walk function from purrr package without actually loading it
purrr::walk(packages, library, character.only=TRUE)

dir('data')
setwd("C:/Users/hegki001/Dropbox/machineLearning/R/odsceast2018")

lots <- readr::read_csv(
    here('data', 'manhattan_Train.csv'))

View(lots) #Tabluar view
names(lots) # Column Nmaes
valueFormula <- TotalValue ~ FireService + ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + BldgDepth + LotType + Landmark + BuiltFAR +
    Built + HistoricDistrict - 1

valueFormula
#ValueForumula is a variable in R which is a Formula Object
class(valueFormula)

# Why is lm fast in R, because it was written in Fortrain - Physicist, Chemist
value1 <- lm(valueFormula, data=lots)

coefplot(value1, sort='magnitude')
summary(value1)
#Std Error = 
# Read a CoefPlot - Large lines and Big confidence Interval
# glmnet requires matrices - so you can't give you categorical data

lotsX <- build.x(valueFormula, data=lots, contrasts = FALSE, 
                 sparse = TRUE)
# Do we want to use a Sparse Matrix or Dense Matrix - smaller & saves time
dim(lotsX)
class(lotsX)

lotsY <- build.y(valueFormula, data=lots) 
# Dont add contrats & Sparse - don't belong here
head(lotsY)
class(lotsY)

#Fitting a model called value2
value2 <- glmnet(x=lotsX, y=lotsY, family = 'gaussian')
value2$lambda
# Value is 94, which means it fits 94 different model, bigger the lambda more shrinkage,
# smaller the lambda less shrinkage
plot(value2) # didn't know what's going on
coefpath(value2)
# in LM each variable gets a coefficient - below gives 94 coefficients, bigger Lambda more 
coef(value2)
coefplot(value2, sort='magnitude', lambda=500)
coefplot(value2, sort='magnitude', lambda=2000000)
coefplot(value2, sort='magnitude', lambda=100000)
#different values of lambda led to lot of different parametrization 
# Notice there is a intercept inspite of getting rid of in the formula - because glmnet adds it
# should we scale the variables? glmnet takes care of scaling the variables
# What's the optimal lambda? Cross validation

library(animation)
cv.ani(k=10)

value3 <- cv.glmnet(x=lotsX, y=lotsY, family='gaussian', nfolds=5)

plot(value3)
#Plot gives different plot depending on the context, Top of X axis is no of models (85 variables)
#Left vertical - value will lead to best fitting model for your data - CV mean-squared Error
value3$lambda.min 
# above gives Lambda min 
# Will the results change based on seed, 2 randomness - rows are assigned in chunks, 
# Should we go with simplest or other? Start with simple
value3$lambda.1se
coefpath((value3))
coefplot(value3, sort='magnitude', lambda='lambda.min')
coefplot(value3, sort='magnitude', lambda='lambda.1se')

coefplot(value3, sort='magnitude', lambda='lambda.1se', plot=FALSE)

#Shrinkage - coefficients are shrunk towards zero so we don't have crazy effect, but some variables
# are shrunk towards zero that means they are not selected
# if your outcome is numeric - use gaussian, binary use - , outcome is survial data - coxph, 
#outcome is multinomimal , so your Y variable dictaces your family
# CV Means Squared Error, if Binomial RSE or AIC
# Can glmnet do hierachial model? Hierachial models are multi-level modes liek students in class in state
# glmnet is not hierarchial , gllm lasso can be use for Multi-level lasso
value3$cvsd

value4 <- cv.glmnet(x=lotsX, y=lotsY, family='gaussian', nfolds = 5, alpha=0)
#By setting alpha =0 we now fit cv(Cross Validated) ridge regresssion

plot(value4)
#Notice with Lasso it tends towards zero but with ridge it will have all  95 variables
# in Lasso 2 variables are highly correlated it shrinks them, in Ridge it splits them half & half
coefpath(value4)

#Most of Machine Learning is brute force fitting with tuning the coefficients & models

coefplot(value4, sort = 'magnitude', lambda = 'lambda.1se')

# R-squared is BS, ASE & BSE is abouttraining data, Mean Squared Error
value5 <- cv.glmnet(x=lotsX, y=lotsY, family='gaussian', nfolds = 5, alpha=0.5)
# Alpha = 0.5, half lasso, half ridge
coefpath(value5)

#Cross validation of alpha is not built into the package
#Chapter 23 of R boot can be used to do cross validation

lots_new <- readRDS(here('data', 'manhattan_Test.rds'))
# in Stats you call score, in regular Machine Learning you call Predict, in Deep learning you call inference

value3Preds <- predict(value3, newx = lots_new, s='lambda.1se')
#above gives error - since you are supposed to do matrix as below

lotsX_new <- build.x(valueFormula, data = lots_new, contrasts = FALSE, sparse = TRUE)
value3Preds <- predict(value3, newx = lotsX_new, s='lambda.1se')

head(value3Preds)

?glmnet
# what is R squared? its a calculator for variability but dangerous.

#glmnet gets your better prediction in better model