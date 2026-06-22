
import tensorflow as tf

# Get list of physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("GPU Details:", gpus)