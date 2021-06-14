library(keras)
library(cloudml)

data_path <- gs_data_dir_local("gs://boreal-dock-314313/ada2021/data/train/")

# data_path <- "data/train/"

# Data generators
generator_train <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2,
  width_shift_range = 0.4,
  height_shift_range = 0.4,
  shear_range = 0.4,
  zoom_range = 0.4
)

generator_valid <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

# Flows
train_flow <- flow_images_from_directory(
  data_path,
  generator = generator_train,
  target_size = c(48, 48),
  batch_size = 64,
  subset = "training"
)

valid_flow <- flow_images_from_directory(
  data_path,
  generator = generator_valid,
  target_size = c(48, 48),
  batch_size = 32,
  subset = "validation"
)

# Model base
model_base <- application_resnet50(
  include_top = FALSE,
  weights = "imagenet",
  input_shape = c(48, 48, 3)
)

freeze_weights(model_base)

model <- keras_model_sequential() %>%
  model_base %>%
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(100, activation = "relu") %>%
  layer_dense(7, activation = "softmax")

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = loss_categorical_crossentropy,
  metric = metric_categorical_accuracy
)

model %>% fit_generator(
  generator = train_flow,
  steps_per_epoch = train_flow$n / train_flow$batch_size,
  epochs = 30,
  validation_data = valid_flow,
  validation_steps = valid_flow$n / valid_flow$batch_size,
  callbacks = callback_early_stopping(patience = 5)
)

###################################################################################################


model %>% save_model_hdf5("model_resnet50.rds")