import tensorflow as tf

# Your custom function used in Lambda
def sum_over_axis1(x):
    return tf.reduce_sum(x, axis=1)

# Enable Lambda loading
import keras
keras.config.enable_unsafe_deserialization()

# Load the model
model = tf.keras.models.load_model(
    "models/final_direction_model.keras",
    custom_objects={"sum_over_axis1": sum_over_axis1}
)

model.summary()
