# igextract
This repository contains the following:\
(1) fpc_files directory: this contains coded Food Policy Council documents in a format appropriate for our analysis.\
(2) dr_DeepLearner.r: this file contains the R code that creates the data contained in the fpc_files folder, and then
	executes a series of analyses. The code is commented to indicate what is being done.


## Installing Keras for R
To run the code, you'll need to setup [Keras](https://keras.rstudio.com/) in RStudio. 

To install the Keras R package from GitHub, execute the following commands:
```
install.packages("devtools")
devtools::install_github("rstudio/keras")

library(keras)
```

Next, you should install Keras with TensorFlow backend by running the following line in the R terminal: 
```
install_keras()
```

## Other Instructions
* Install any other necessary packages.
* Change the file path if applicable.
* Run the R code and get your result displayed.