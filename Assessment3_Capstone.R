rm(list=ls())

library(ISLR)
library(caret, warn.conflicts = F, quietly = T)
library(dplyr)
library(cluster, warn.conflicts = F, quietly = T) #clustering algorithms
library(factoextra, warn.conflicts = F, quietly = T) #data visualization
library(psych)
library(dbscan)


#Clustering
#capstone
data <- read.csv("FacebookSellers.csv", header = TRUE)
summary(data)
str(data)
#data Transformation
data$status_id <-as.factor(data$status_id)
data$status_type <-as.factor(data$status_type)
#delete NA columns
data <- data[,1:12]
data <- na.omit(data)

#identify duplicate rows 
which(duplicated(data))
#delete duplicated rows that are identified 
data <- data[-which(duplicated(data)),] #now observations are 6999

#### KNN Outlier
#install.packages("dbscan")

data_Outlier <- data[-c(1:3)]
#KNN parameter
k=4
#using latest version for KNN (All - TRUE)
KNN_Outlier <- kNNdist(x=data_Outlier, k =k, all = TRUE)[,k]
#No. of top outliers to be displayed
top_n <- 20
#sorting Outliers
rank_KNN_Outlier <- order(KNN_Outlier, decreasing = TRUE)
KNN_Result <- data.frame(ID = rank_KNN_Outlier, score = KNN_Outlier[rank_KNN_Outlier])
#showing top 20 outliers with ID's and scores
head(KNN_Result, top_n)

#plotting the Outliers
g0a <- ggplot() + geom_point(data=data_Outlier, mapping=aes(x=num_reactions, y= num_shares), shape = 19)
g <- g0a +
  geom_point(data = data_Outlier[rank_KNN_Outlier[1:top_n],], mapping = aes(x=num_reactions,y=num_shares), shape=19, color="red", size=2) +
  geom_text(data=data_Outlier[rank_KNN_Outlier[1:top_n],],
            mapping=aes(x=(num_reactions-0.5), y=num_shares, label=rank_KNN_Outlier[1:top_n]), size=2.5)
g

#delete Outliers from for final data
data_final <- data[-c(499,481,6707,4544,3247,4515,3893,727,6777,6609,6712,4519,6725,6749,4612,
                       6246,3852,4563, 3877,1230),]

#PCA
#show how samples are related or not to each other
#delete unwanted variables/data

#delete status_type and status_published variables for PCA
data_PCA <- data_final[-c(2:3)]

#identify duplicate rows 
which(duplicated(data_PCA))
#delete duplicated rows that are identified 
data_PCA <- data_PCA[-which(duplicated(data_PCA)),] #now observations are 6977

#make status_id row names
row.names(data_PCA) <- data_PCA$status_id
#delete status_id columns
data_PCA <-data_PCA[,-1]

#perform PCA
pca_res <- prcomp(data_PCA, center = TRUE, scale = TRUE)
pca_res
pca_res$rotation
pca_res$x
#plot
plot(pca_res$x[,1], pca_res$x[,2])

fviz_screeplot(pca_res) 
#shows the amount percentage of variance that explain by each principal component axis
#explains 60percent of the variation within the total data set.
#For example in the image shown above sharp bend is at 3. So, the number of principal axes should be 3.


data_final1 <- data[-c(1,3)]
data_final1 <- data_final1[c(1:4)]
###

#data_K is the data with only 3 variables as result from PCA
data_k <- data_final1[,-1]
#scaling data_K to normalise data
data_k <- scale(data_k)



set.seed(6) 
#Appply K means algorithm
kmeans_data <- kmeans(data_k, centers = 4, nstart = 10) 
str(kmeans_data)
fviz_cluster(kmeans_data, data = data_k)

fviz_cluster(kmeans_data, #set up plot
             data = data_k, 
             geom = "point", # only shows points and not labels 
             shape = 19,# define one shape for all clusters (a circle)
             alpha = 0)+ # make circles see-though
  geom_point(aes(colour = as.factor(kmeans_data$cluster), 
                 shape = data_final1$status_type))+ #colour by status type
  ggtitle("Comparing Clusters and Status type") #add a title

#perform kmeans & calculate ss 
total_sum_squares <- function(k){ 
  kmeans(data_k, centers = k, nstart = 10)$tot.withinss
}

#define a sequence of values for k up to 10 sequences
all_ks <- seq(1,10,1) 

#apply to all values of k
choose_k <- sapply(seq_along(all_ks), function(i){ 
  total_sum_squares(all_ks[i])
})
# dataframe for plotting
choose_k_plot <- data.frame(k = all_ks,  
                            within_cluster_variation = choose_k)
# plot
ggplot(choose_k_plot, aes(x = k, 
                          y = within_cluster_variation))+
  geom_point()+
  geom_line()+
  xlab("Number of Clusters (K)")+
  ylab("Within Cluster Variation")


#Supervised Learning/ Regression

data_model <- data_final1[,-1]
#checking for predictors correlation
cor_data <- cor(data_model,method = "pearson")
cor_data
pairs.panels(data_model)
#Correlation shows low correlation between variables therefore assume that the yare independent, Hence Naive Bayes will be use for prediction




##partition data set into train and set and randomly split it
set.seed(123) 
# Creating index for randomly splitting the dataset
ind3= createDataPartition(data_final$status_type, p=0.8, list=FALSE)
train.data <- data_final[ind3,]
test.data <- data_final[-ind3,]
# Training the model
c(nrow(train.data), nrow(test.data))
summary(data_final)
#implementing Naive Bayes
model <- train(status_type~.,
              data=train.data,
              trControl = trainControl(method = "cv", number = 5),
              method = "nb")
model
#confusion Matrix for training data
confusionMatrix(predict(model,newdata = train.data),
                train.data$status_type)
#confusion Matrix for test data
confusionMatrix(predict(model,newdata = test.data),
                test.data$status_type)
