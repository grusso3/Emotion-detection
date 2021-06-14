library(cloudml)
library(keras)

getwd()

# Run this to test scripts

gs_path <- "gs://boreal-dock-314313/ada2021/"
#
gs_copy(source = "data",
        destination = gs_path,
        recursive = TRUE) # Copy from local directory to gcloud bucket


### TEST RUNS
########################################################

# VGG-16

cloudml_train(
  file = "scripts/model_vgg16.R",
  master_type = "standard_gpu"
)

job_collect("cloudml_2021_06_14_080517124")
view_run("runs/cloudml_2021_06_14_080517124")

# ResNet50

cloudml_train(
  file = "scripts/model_resnet50.R",
  master_type = "standard_gpu"
)

job_collect("cloudml_2021_06_14_091037016")
view_run("runs/cloudml_2021_06_08_164831387")


# Inception-v3

cloudml_train(
  file = "scripts/model_inceptionv3.R",
  master_type = "standard_gpu"
)

job_collect("cloudml_2021_06_14_113831348")
view_run("runs/cloudml_2021_06_14_113831348")


# DenseNet169

cloudml_train(
  file = "scripts/model_densenet169.R",
  master_type = "standard_gpu"
)

job_collect("cloudml_2021_06_14_092935048")
view_run("runs/cloudml_2021_06_14_092935048")
