import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers,lb, ub, activation='relu', kernel_initializer='glorot_normal',**kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = len(layers) - 2
        self.output_dim = layers[-1] 
        self.lb = lb
        self.ub = ub
        
        
        self.hidden = [tf.keras.layers.Dense(units,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for units in layers[1:-1]]
        self.out = tf.keras.layers.Dense(layers[-1])
    
    def call(self, inputs):
        X = 2.0 * (inputs - self.lb) / (self.ub-self.lb) - 1.0
        for layers in self.hidden:
            X = layers(X)
        return self.out(X)
    
class PINNs_Slover():
    def __init__(self, model,u_train, nu, m, d, load):
        self.model = model
        self.u_train = u_train
        self.nu = nu
        self.m = m
        self.d = d
        self.load = load
        
    def net_u(self,X_u_train):
        return self.model(X_u_train) 

    def net_f(self,X_f_train):
        with tf.GradientTape(persistent=True) as tape:
        
            tape.watch(X_f_train)
            usol1 = self.model(X_f_train)
            gradients = tape.gradient(usol1, X_f_train)
            u_t = gradients[:, 0]
            
        u_tt = tape.gradient(u_t,X_f_train)[:, 0]
        usol4 = X_f_train[:,1]
        wind_power = X_f_train[:,2]
        z = self.nu * tf.math.sin(usol1 - usol4)
        z = tf.squeeze(z)
        f = self.m * u_tt + self.d * u_t + z - (5/12) * (self.load - wind_power)/78.75

        return f

    def loss_fn(self, u_true, u_pred, f_pred):
        data_loss = tf.reduce_mean(tf.square(u_true - u_pred))
        physics_loss = tf.reduce_mean(tf.square(f_pred))
        total_loss = data_loss + physics_loss
        return total_loss, data_loss, physics_loss
    
    def train_step(self,X_u_train, X_f_train):
        with tf.GradientTape(persistent=True) as tape:
            
            u_pred = self.net_u(X_u_train)
            f_pred = self.net_f(X_f_train)
            
            total_loss, data_loss, physics_loss = self.loss_fn(self.u_train, u_pred, f_pred)
        gradients = tape.gradient(total_loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss, data_loss, physics_loss
    
    def fit(self, epochs, X_u_train, X_f_train, u_train, optimizer=tf.optimizers.Adam()):
        self.optimizer = optimizer
        
        for epoch in range(epochs):
            total_loss, data_loss, physics_loss = self.train_step(X_u_train, X_f_train)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}, Data Loss: {data_loss.numpy()}, Physics Loss: {physics_loss.numpy()}")
    
    def predict(self, X_star):
        u_star = self.net_u(X_star)
        f_star = self.net_f(X_star)
        return u_star,f_star


nu=1.63
m = 0.4
d = 1
load = 315
# noise = 0.0        
# # P1m = 0.18
# N_u = 600
N_f = 10000 
Epoch = 100
layers = [3, 15, 15, 15, 15, 15, 1]


# data_folder = './2_final_50'
# files = os.listdir(data_folder)
# all_t = []
# all_usol4 = []
# all_windPower = []
# all_usol1 = []

# count = 0  # 初始化计数器

# for file in files:
#     if file.endswith('.mat'):
#         mat_data = scipy.io.loadmat(os.path.join(data_folder, file))
#         for key in mat_data.keys():
#             if isinstance(mat_data[key], np.ndarray):  # 确保只处理 NumPy 数组类型的数据
#                 mat_data[key] = mat_data[key][:-1]        # 删除最后一行
#         #print(mat_data)
#         all_t.append(mat_data['time'].flatten()[:,None])
#         all_usol4.append(mat_data['usol4'].flatten()[:,None])
#         all_windPower.append(mat_data['windPower'].flatten()[:,None])
#         all_usol1.append(np.real(mat_data['usol1']))
#         count += 1  # 增加计数器
#         if count == 5:  # 达到五个文件后退出循环
#             break

# t = np.vstack(all_t)
# usol4 = np.vstack(all_usol4)
# windPower = np.vstack(all_windPower)
# usol1 = np.vstack(all_usol1)

# t_normalized = (t-t.min())/(t.max()-t.min())
# usol4_normalized = (usol4 - usol4.min())/(usol4.max()-usol4.min())
# windPower_normalized = (windPower - windPower.min())/(windPower.max()-windPower.min())


data = scipy.io.loadmat('./2_final_50/all_result_1.mat')

for key in data.keys():
    if isinstance(data[key], np.ndarray):  # 确保只处理 NumPy 数组类型的数据
        data[key] = data[key][:-1]        # 删除最后一行

t = data['time'].flatten()[:,None]
usol4 = data['usol4'].flatten()[:,None]
wind_power = data['windPower'].flatten()[:,None]

lb = np.array([t.min(),usol4.min(), wind_power.min()])
ub = np.array([t.max(), usol4.max(), wind_power.max()])

Exact = np.real(data['usol1'])

X_u_train = np.hstack([t, usol4, wind_power])
y_train = Exact.flatten()[:, None]

X_f_train = lb + (ub-lb)*lhs(3, N_f)

X_f_train = np.vstack((X_f_train, X_u_train))

X_u_train = tf.convert_to_tensor(X_u_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_f_train = tf.convert_to_tensor(X_f_train, dtype=tf.float32)

model = PhysicsInformedNN(layers=layers, lb=lb, ub=ub)
solver = PINNs_Slover(model,y_train, nu, m, d, load)
start_time_train = time.time()
solver.fit(Epoch,X_u_train, X_f_train, y_train)
train_time = time.time() - start_time_train
print('training time: %.4f' % (train_time))

start_time = time.time()
y_pred, f_pred = solver.predict(X_u_train)
elapsed = time.time() - start_time
print('testing time: %.4f' % (elapsed))

print(y_pred)

y_train_flattened = y_train.numpy().flatten()
y_pred_flattened = y_pred.numpy().flatten()



l2_error = np.sqrt(np.sum((y_pred_flattened[50:] - y_train_flattened[50:])**2))
print("L2 Error:", l2_error)

# # 确保时间数组的长度与展平后的数据长度一致
t_min = t.min()
t_max = t.max()

time = np.linspace(t_min, t_max, len(y_train_flattened))[50:]

# # 绘图
plt.figure(figsize=(10, 5))
plt.plot(time, y_train_flattened[50:], label="real value", linestyle='-')
plt.plot(time, y_pred_flattened[50:], label="predict value", linestyle='--')

plt.xlabel('time')
plt.ylabel('u(rad)')
plt.legend()
plt.title('Bus 1')
plt.show()
#plt.savefig("network_graph.png")