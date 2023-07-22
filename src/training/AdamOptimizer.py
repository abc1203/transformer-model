from tensorflow.keras.optimizers import Adam


class AdamOptimizer(Adam):
    """
    Adam Opimtizer with beta_1 = 0.9, beta_2 = 0.98, epsilon = 10^-9, and
    learning_rate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    """
    
    def __init__(self, step_num, beta1 = 0.9, beta2 = 0.98, epsilon = 10**-9, 
            d_model = 512, warmup_steps = 4000, **kwargs):
        self.lrate = (d_model ** -0.5) * min((step_num ** -0.5), step_num * (warmup_steps ** -1.5))

        super().__init__(learning_rate=self.lrate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

