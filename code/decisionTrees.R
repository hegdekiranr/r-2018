# Advantages of Decision Tree, easy to understand, Robust outliers, Easy to compute
# Disadvantages of Decision Trees - if training data changes it varies too much, 
# you can overfit by having too deep of a tree which also is same as they are highly variable.
# First attempt to fixing it is by Random Forests, as they are ensamble models,
# if one model is weak other is strong they make up for each other
# After Random Forests came along, then Gradient Boosting came along, all idea of Boosting is you
# e.g 1000 rows of data, see how well you did from residuals of that model
# rows having small residuals upweight and vice versa e.g AdTech you don't have a lot of positive cases
# its a additive model, Prediction of one is sum of all he Predictions, one other BART
# GBM - Gradient Boosting Machine, GAMs -, Gap stats for Kmeans clustering, Bootstrap
# DMLC wrote SGBoost - greatest thing since glmnet, 
# Microsoft came up with light GBM, says its faster 
# Boosting does automatic variable selection

packages <- c('useful', 'coefplot', 'xgboost','here', 'magrittr', 'dygraphs','DiagrammeR')

purrr::walk(packages,library, character.only=TRUE)

manTrain <- readRDS(here('data', 'manhattan_Train.rds'))

View(manTrain)

manVal <- readRDS(here('data', 'manhattan_validate.rds'))

manTest <- readRDS(here('data','manhattan_Test.rds'))

histFormula <- HistoricDistrict ~ FireService + ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + BldgDepth + LotType + Landmark + BuiltFAR +
    Built + TotalValue - 1

manX_Train <- build.x(histFormula, data=manTrain, contrasts = FALSE, sparse = TRUE)

manY_Train <- build.y(histFormula, data = manTrain) %>% as.integer() - 1
# Basically Build.y gives us Yes/No, then  "%>% as.integer()" gives 1 & 2 and subtracting by 1 becomes 0 & 1

manX_Val <- build.x(histFormula, data=manVal, contrasts = FALSE, sparse = TRUE)

manY_Val <- build.y(histFormula, data = manVal) %>% as.integer() - 1

manX_Test <- build.x(histFormula, data=manTest, contrasts = FALSE, sparse = TRUE)

manY_Test <- build.y(histFormula, data = manTest) %>% as.integer() - 1

#Below - Data is X and Lable is Y
xgTrain <- xgb.DMatrix(data = manX_Train, label=manY_Train)
xgTrain

xgVal <- xgb.DMatrix(data = manX_Val, label=manY_Val)

xg1 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 1
)
# What is a logloss? its not saying right/wrong in traditional misclassification, 

xg2 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 1,
    watchlist = list(train=xgTrain)
)

xg2$evaluation_log

xg3 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 300,
    print_every_n = 1,
    watchlist = list(train=xgTrain)
)

xg4 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 300,
    print_every_n = 1,
    watchlist = list(train=xgTrain, validate=xgVal)
)

# if we see the Validate-logloss gets worse then its over-fitting

dygraph(xg4$evaluation_log)


xg5 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 1000,
    print_every_n = 10,
    watchlist = list(train=xgTrain, validate=xgVal)
)

dygraph(xg5$evaluation_log)

xg6 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 1000,
    print_every_n = 10,
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 70
)

dygraph(xg6$evaluation_log)


xg7 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 1000,
    print_every_n = 10,
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 70,
    xgb_model = xg3
)

#one way to visualize trees
xgb.plot.multi.trees(xg6, feature_names = colnames(manX_Train))

#Variable order
xgb.plot.importance(
    xgb.importance(
        xg6, feature_names = colnames(manX_Train)
    )
)

#below is above + %>% View
xgb.plot.importance(
    xgb.importance(
        xg6, feature_names = colnames(manX_Train)
    )
) %>% View

xg8 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gblinear', 
    eval_metric='logloss',
    nrounds = 1000,
    print_every_n = 10,
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 70
)

# When we use Linear model we can do coefficient plot below
coefplot(xg8, sort = 'magnitude')

#Boosting goes well beyond trees, even linear
#how to find names
#coefplot(xg8, newNames = colnames(manX_Train)) #### Not working

xg9 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gblinear', 
    eval_metric='logloss',
    nrounds = 1000,
    print_every_n = 10,
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 70,
    alpha = 100000, # L1 penalty called Lambda in glmnet discussion
    lamba = 200 # L2 penalty called alpha in glmnet discussion
)

coefplot(xg9, sort = 'magnitude')

dygraph(xg9$evaluation_log)

xg10 <- xgb.train(
    data = xgTrain,
    objective = 'binary:logistic',
    booster = 'gbtree',
    eval_metric='logloss',
    nrounds = 2500,
    print_every_n = 20,
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 70,
    max_depth=3, eta=0.1
)

#Whole goal of a Boosted model is create a bunch of weak learners 
#As opposed to deep Strong learners
#By reducing depth of the tree you are preventing the model for over-fitting
#How do you find the ideal depth? but there is a package got caret

#Why do Jared prefers logloss over AUC? no good reason

xg11 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric ='logloss',
    booster ='gbtree',
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 20,
    subsample = 0.5,
    nrounds = 100,
    print_every_n = 1
)

# What is special about subsample ?

xg11 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric ='logloss',
    booster ='gbtree',
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 20,
    subsample = 0.5, col_subsample=0.5,
    nrounds = 100,
    print_every_n = 1
)

# Above sounds like random forests but its not since everything is sequential

xg11 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric ='logloss',
    booster ='gbtree',
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 20,
    subsample = 0.5, col_subsample=0.5,
    nrounds = 50,
    num_parallel_tree=20,
    print_every_n = 1
)

# in Above we are now doing the boosted forest, although might need hyperparameter tuning

xg12 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric ='logloss',
    booster ='gbtree',
    watchlist = list(train=xgTrain, validate=xgVal),
    early_stopping_rounds = 20,
    subsample = 0.5, col_subsample=0.5,
    nrounds = 50,
    num_parallel_tree=20,
    print_every_n = 1,
    nthread=4
)
