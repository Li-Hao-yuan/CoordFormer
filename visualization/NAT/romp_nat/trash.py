import numpy as np

# info = {"1":2}
# print(info)
# np.savez('C:/Users/Public/Desktop/info.npz',info=info)

info_load = np.load('C:/Users/Public/Desktop/info.npz',allow_pickle=True)['info'][()]
print(np.max(info_load['kp2d']))
print(np.min(info_load['kp2d']))
print(type(info_load))