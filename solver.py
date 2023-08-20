import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
#import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
    
# Load the saved model
model = keras.models.load_model("model.h5", custom_objects={'CTCLayer': CTCLayer}, compile=False)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

characters = [' ', '0', '2', '4', '8', 'A', 'D', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y'];


def preprocess_image(img_contents):
    # 1. Decode and convert to grayscale
    img = tf.io.decode_png(img_contents, channels=1)
    # 2. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 3. Resize to the desired size
    img = tf.image.resize(img, [80, 300])
    # 4. Transpose the image
    img = tf.transpose(img, perm=[1, 0, 2])
    # 5. Expand dimensions to shape (1, img_width, img_height, 1) for prediction
    img = tf.expand_dims(img, axis=0)
    
    return img

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None,
)  

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :6
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def predict_captcha(img_contents):
    # Preprocess the image
    img = preprocess_image(img_contents)

    # Predict using the model
    preds = prediction_model.predict(img)
    
    # Decode the predictions
    pred_texts = decode_batch_predictions(preds)
    print(pred_texts)
    
    # Since it's a single image, our result will be the first element
    predicted_text = pred_texts[0].replace('[UNK]', '')
    if len(predicted_text) == 5:
      predicted_text += ' '
    return predicted_text

# Example usage
#predicted_text = predict_captcha("sneed.png")
#print(predicted_text)
