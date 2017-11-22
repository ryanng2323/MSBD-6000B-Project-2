source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage")
library (EBImage)
library(keras)
library(rio)
library(magick)


size1 = c(64,64)

train_path<-import("train.txt")
train_img<-array(0,dim=c(nrow(train_path),size1,3))
for (i in 1:nrow(train_path)){
imgtraining<-image_load(paste(train_path[,1][i],sep=""),target_size=size1)
train_img[i,,,]<-image_to_array(imgtraining)}

test_path<-import("test.txt")
test_img<-array(0,dim=c(nrow(test_path),size1,3))
for (i in 1:nrow(test_path)){
imgtesting<-image_load(paste(test_path[,1][i],sep=""),target_size=size1)
test_img[i,,,]<-image_to_array(imgtesting)}

val_path<-import("val.txt")
val_img<-array(0,dim=c(nrow(val_path),size1,3))
for (i in 1:nrow(val_path)){
imgval<-image_load(paste(val_path[,1][i],sep=""),target_size=size1)
val_img[i,,,]<-image_to_array(imgval)}

#Model

num_classes <- max(train_path[,2]) + 1

#Train Data
x_train <- train_img
x_train <- x_train / 255
y_train <- to_categorical(train_path[,2])

#Test Data
x_test <- test_img
x_test <- x_test / 255
y_test <- array(0,dim=c(nrow(test_path),5))

#Val Data
x_val<-val_img
x_val <- x_val / 255
y_val <- to_categorical(val_path[,2])

datagen <- image_data_generator(
  rotation_range = 90,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = TRUE
)


model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3, 3),activation = "relu",
                input_shape = c(size1, 3))%>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),activation = "relu",
              input_shape = c(size1, 3))%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_dropout(rate = 0.25) %>% 
  layer_flatten()%>%
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  flow_images_from_data(x_train, y_train, datagen,batch_size = 128),
  steps_per_epoch = 50, 
  epochs = 50,  
  validation_data = list(x_val,y_val),
  validation_steps = 50,
  verbose = 2
)

pm<-predict(model,x_test)
pm1<-data.frame(apply(pm,1,which.max)-1)
export(pm1,"project2_20383422.csv",col.names=FALSE)


