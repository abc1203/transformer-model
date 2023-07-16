from tensorflow.keras.layers import Layer, LayerNormalization


class LayerNorm(Layer):
    """
    after every sublayer, perform LayerNorm(x + SubLayer(x))
    """

    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.LayerNormalization = LayerNormalization()
    
    def call(self, x, x_sublayer):
        res = self.LayerNormalization(x + x_sublayer)

        return res

