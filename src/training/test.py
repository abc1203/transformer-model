import numpy as np
from tensorflow import one_hot, shape, convert_to_tensor, math, cast, float32
from time import time


mat1 = np.random.random((64, 100))
mat2 = math.logical_not(math.equal(mat1, 0))
mat2 = cast(mat2, float32)

mat1_one_hot = one_hot(mat1, depth=4000)
print(mat1_one_hot)

output_list = []

start_time = time()

for i in range(mat1_one_hot.shape[0]):
    sentence = [mat1_one_hot[i][j] * mat2[i][j] for j in range(mat1_one_hot.shape[1])]
    output_list.append(sentence)


print(convert_to_tensor(output_list))
print(math.reduce_sum(output_list) / math.reduce_sum(mat2))