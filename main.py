import tensorflow as tf
import random
import matplotlib.pyplot as plt

print('TensorFlow version', tf.__version__)

'''
Download and get the model
Below we get the VGG16 model and store it into the variable model.
include_top = False, we basically don't need the top layers of the model, i.e. the FC layers
weights = 'imagenet', we take the pretrained weights of the imagenet
input_shape (96,96,3) = input shape of the image
'''
model = tf.keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet',input_shape = (96,96,3))
model.summary()       # summary of the model gets printed in the terminal


'''
Get the Sub Model
Now we get the sub_model from the above model, i.e. the layer of the model
model.input = input as the model (VGG16)
model.get_layer(layer_name).output = output as the layer of the model VGG16
'''
def get_submodel(layer_name):
    return tf.keras.models.Model(model.input, model.get_layer(layer_name).output)   
get_submodel('block1_conv1').summary()    # summary of the sub_model gets printed in the terminal

'''
Image Visualization
Function to create random images to use them as input images.
We will use a pretrained model (VGG 16) and input random images to it.
'''



