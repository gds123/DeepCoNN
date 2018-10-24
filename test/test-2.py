import numpy as np
import pickle
import pandas as pd

# with open("/Users/guowang/PycharmProjects/DeepCoNN/data/glove.6B/glove.6B.50d.txt", 'r') as f:
#     a = f.readline().split()
#     print(a)
#     print(np.asarray(a[1:], dtype=np.float32))

pkl_file = open("../data/music/music.train", 'rb')
test_data = pickle.load(pkl_file)
test_data = np.array(test_data, dtype=np.int)
print(test_data)
print(len(test_data))
