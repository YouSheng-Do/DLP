import numpy as np
import matplotlib.pyplot as plt
class kl_annealing():
    def __init__(self, current_epoch=0):
        # TODO
        self.current_epoch = 0
        self.frange_cycle_linear(70, 0, 1, 1, 0.5)
        
    def update(self):
        # TODO
        self.current_epoch += 1
    
    def get_beta(self):
        # TODO
        return self.beta_array[self.current_epoch]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        self.beta_array = np.ones(n_iter)
        # period = n_iter // n_cycle
        # step = (stop - start) / ((period - 1) * ratio)
        # for c in range(n_cycle):
        #     for i in range(period):
        #         beta = i * step if i * step < stop else stop
        #         self.beta_array[i + c * period] = beta

x = kl_annealing()

plt.figure(figsize=(8, 5))
plt.plot(x.beta_array)
plt.title('KL annealing')
plt.xlabel('Epoch')
plt.ylabel('beta')

plt.savefig('kl_annealing.png')



# def inverse_sigmoid_decay(k, epoch):
#     """Calculate the probability of using teacher forcing."""
#     return k / (k + np.exp(epoch / k))

# def exponential_decy(epoch):
#     if epoch >= 10:
#         return np.exp(-0.1 * (epoch - 10))
    
# # 設定 k 值和訓練週期範圍
# k = 10
# epochs = np.linspace(0, 70, 500)  # 從 0 到 50，共 500 點

# # 計算每個訓練週期的概率
# # probabilities = [inverse_sigmoid_decay(k, epoch) for epoch in epochs]
# probabilities = [exponential_decy(epoch) for epoch in epochs]

# # 畫圖
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, probabilities, label='Inverse Sigmoid Decay')
# plt.title('teacher forcing ratio')
# plt.xlabel('Epoch')
# plt.ylabel('Teacher forcing ratio')
# plt.legend()
# plt.savefig('test.png')
