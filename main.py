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

def create_image():
    return tf.random.uniform((96,96,3), minval = -0.5, maxval = 0.5)

'''
image = image - tf.math.reduce_min(image)
image = image / tf.math.reduce_max(image)
what this means is that if the mean is centered around zero or something like that, 
we have some values which are negative, let's say, minus one. And if that's the minimum value of our image in our image,
then we just add that to the image. And once we have all positive values, we scale this in a way so that all the values
are from 0 to 1. It helps in plotting the image
'''
def plot_image(image, title = 'random'):
    image = image - tf.math.reduce_min(image)
    image = image / tf.math.reduce_max(image)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

image  = create_image()
plot_image(image, title = 'random')

'''
Training Loop using Gradient Ascent Algorithm
f_index = filter index
'''

def Train(layer_name, f_index = None, epoch = 50):
    submodel = get_submodel(layer_name)
    num_filters = submodel.output.shape[-1]

    if f_index is None:
        f_index = random.ranint(0,num_filters - 1)
    assert num_filters > f_index, 'f_index is out of bounds'

    image = create_image()

    verbose_step = int(epoch / 10)   # to keep a track of the loss
 
    for i in range(epoch):
        with tf.GradientTape() as tape:
            tape.watch(image)               #Watch the image in gradient tape context
            output = submodel(tf.expand_dims(image,axis = 0))[:,:,:,f_index]
            # expand the image and look at the output for only a particular filter
            loss = tf.math.reduce_mean(output)    # loss for output of a particular filter
        grads = tape.gradient(loss,image)
        grads = tf.math.l2_normalize(grads) #l2 normalization
        image += grads * 10     # Gradient Ascent
    
        if (i+1) % verbose_step == 0:
            print(f'Iteration: {i+1}, Loss: {loss.numpy():.4f}')
    plot_image(image, f'{layer_name},{f_index}')