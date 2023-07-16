from tensorflow.keras.layers import Layer, Dense, ReLU


class PositionwiseFeedForwardNetwork(Layer):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2; activation function RELU is used between the 2 layers
    b1, b2 are likely residuals of some form, but they weren't specified in the paper; they will not be included in this implementation
    inner layer has dimension d_ff = 2048, input & output has dimension d_model = 512
    """

    def __init__(self, d_ff = 2048, d_model = 512, **kwargs):
        super(PositionwiseFeedForwardNetwork, self).__init__(**kwargs)
        self.d_ff = d_ff
        self.d_model = d_model

        self.W1 = Dense(d_ff)
        self.W2 = Dense(d_model)
        self.ReLU = ReLU()
    

    def call(self, x):
        # 1st layer
        res = self.W1(x)

        # activation RELU
        res = self.ReLU(res)

        # 2nd layer
        res = self.W2(res)

        return res


