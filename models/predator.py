import keras

class PredatorModel:

    def __init__(self, weights=None):

        self.weights = weights


    def build_model(self, input_size=20):  # 16 for the field observation, 4 for own coordinates and vector

        model = keras.models.Sequential(

            keras.layers.dense
        )




