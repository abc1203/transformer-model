from tensorflow import cast, float32
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule



class AdamOptimizer(Adam):
    """
    Adam Opimtizer with beta_1 = 0.9, beta_2 = 0.98, epsilon = 10^-9, and
    learning_rate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    """
    def __init__(self, beta1 = 0.9, beta2 = 0.98, epsilon = 10**-9, **kwargs):
        super().__init__(learning_rate=LearningRate(), beta_1=beta1, beta_2=beta2, epsilon=epsilon)



class LearningRate(LearningRateSchedule):
    """
    learning_rate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    """
    def __init__(self, d_model = 512, warmup_steps = 4000, **kwargs):
        super(LearningRate, self).__init__(**kwargs)
        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step_num):
        step_num = cast(step_num, float32)
        lrate = pow(self.d_model, -0.5) * min(pow(step_num, -0.5), step_num * pow(self.warmup_steps, -1.5))

        return lrate

