#################################
#   Model Architecture Params   #
#################################

pred_activation = "sigmoid"


#################################
#       Optimizer Params        #
#################################

loss = "binary_crossentropy"
metrics = ["accuracy", 'categorical_accuracy']


#################################
#       Dataset Details         #
#################################

n_classes = 228
input_shape = (224, 224, 3)
