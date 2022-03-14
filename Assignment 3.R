library(tensorflow)
#install_tensorflow(version = "2.0.0b1", method = "conda", envname = "r-reticulate")
library(keras)
#install_keras()

setwd("/Users/abi/Documents/GSU current/8392 - Topics in BD/Assignment 3") # make sure your working directory is correct
original_dataset_dir <- "Braille Dataset" # we will only use the labelled data
base_dir <- "Brail_learn" # to store a sebset of data that we are going to use
ls <- c("dim", "rot", "whs")
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)


for (chr in letters[1:26]){
  current_train <- file.path(train_dir, chr)
  current_valid <- file.path(validation_dir, chr)
  current_test <- file.path(test_dir, chr)
  dir.create(current_train)
  dir.create(current_valid)
  dir.create(current_test)
}

for (word in ls){
  for (i in 0:19){
    for (chr in letters[1:26]){
      fnames <- paste0(chr,"1.JPG",i, word,".jpg")
      file.copy(file.path(original_dataset_dir, fnames), file.path(train_dir, chr))
    }
  }
}

for (word in ls){
  for (i in 10:15){
    for (chr in letters[1:26]){
      fnames <- paste0(chr,"1.JPG",i, word,".jpg")
      file.copy(file.path(original_dataset_dir, fnames), file.path(validation_dir, chr))
    }
  }
}
for (word in ls){
  for (i in 16:19){
    for (chr in letters[1:26]){
      fnames <- paste0(chr,"1.JPG",i, word,".jpg")
      file.copy(file.path(original_dataset_dir, fnames), file.path(test_dir, chr))
    }
  }
}

model_v1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(2, 2), activation = "softmax",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(2, 2), activation = "softmax") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(2, 2), activation = "softmax") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(2, 2), activation = "softmax") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "softmax") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_v1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("acc")
)

summary(model_v1)


train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)


train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory
  train_datagen,              # Training data generator
  target_size = c(150, 150),  # Resizes all images to 150 × 150
  batch_size = 20,            # 20 samples in one batch
  class_mode = "categorical"       # Because we use binary_crossentropy loss,
  # we need binary labels.
)


validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

history_v1 <- model_v1 %>%
  fit(
    train_generator,
    steps_per_epoch = 78,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 10
  )

plot(history_v1)

model_v2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
    

model_v2 %>%  compile(loss = "binary_crossentropy",
                      optimizer = optimizer_rmsprop(),
                      metrics = c("acc")
                      )

history_v2 <- model_v2 %>%
  fit(train_generator,
      steps_per_epoch = 78,
      epochs = 10,
      validation_data = validation_generator,
      validation_steps = 5
      ) 

plot(history_v2)
