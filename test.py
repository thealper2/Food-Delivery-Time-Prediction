import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("FDTmodel.h5")

age = 29
ratings = 2.9
distance = 6

test = np.array([[age, ratings, distance]])
res = model.predict(test)
print(res.item())
