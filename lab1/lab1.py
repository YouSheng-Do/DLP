import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # use to keep the random number the same all the time. However, this is not good since every layer will be the same, so maybe the id of layer could be better seed.


'''
Generate linear data
'''
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

'''
Generate XOR data
'''
def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

'''
Show the comparison of the ground truth and the predicted result
'''
def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5: # threshold
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

'''
Show the learning rate
'''
def show_learning_rate(losses, learning_rate):
    plt.plot(np.squeeze(losses))
    plt.ylabel('loss')
    plt.xlabel('epoch (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

'''
Definition of liner layer
'''
class Linear():
    def __init__(self, n_x, n_y, seed):
        '''
        init weight and bias to random number
        '''
        self.n_x = n_x
        self.n_y = n_y
        self.W = np.random.uniform(size = (self.n_y, self.n_x)) # not sure it is necessary to use gaussion distribution
        self.b = np.zeros((self.n_y, 1)) # use zero initialization for the biases
        # print(self.W.shape)
        # print(self.b.shape)

    def forward(self, A):
        '''
        forward propagation
        '''
        # print(A)
        # print(self.W)
        # print(self.b.shape)
        Z = np.dot(self.W, A)
        # print(Z.shape)
        # Z = Z.reshape((self.W.shape[0],1))
        Z = Z + self.b
        # self.prev_A = A.reshape((A.shape[0],1))
        self.prev_A = A
        # print(Z.shape)
        return Z
    
    def backward(self, dZ):
        '''
        back propagation
        '''
        m = self.prev_A.shape[1]
        self.dW = 1/m * np.dot(dZ, np.transpose(self.prev_A))
        self.db = 1/m * np.transpose(np.sum(dZ, axis = 1)[np.newaxis])
        # print(self.dW)
        # print(self.db.shape)

        # print(self.W.shape)
        # print(dZ.shape)

        d_prev_A = np.dot(np.transpose(self.W), dZ)

        # print(d_prev_A.shape)
        return d_prev_A
    
    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


'''
Test for Linear
''' 
# linear = Linear(2,1)
# print(linear.W)
# print(linear.b)
# Z = linear.forward(np.array([[1.],[2.]]))
# print(Z)
# d_pre_A = linear.backward(np.array([[0.5]]))
# print(d_pre_A)
# linear.update(1.0)
# print(linear.dW)
# print(linear.db)
# print(linear.W)
# print(linear.b)
    
'''
Definition of Sigmoid 
'''
class Sigmoid():
    def sigmoid(self, x):
        self.Z = 1.0/(1.0 + np.exp(-x))
        return self.Z

    def derivative_sigmoid(self):
        x = self.Z
        return np.multiply(x, 1.0 - x)

'''
Test for Sigmoid
'''
# activation = Sigmoid()
# act_Z = activation.sigmoid(Z)
# print(act_Z)
# dZ = d_pre_A * activation.derivative_sigmoid()
# print(dZ)

'''
Definition of ReLU
'''
class ReLU():
    def relu(self, x):
        self.Z = np.maximum(0, x)
        return self.Z

    def derivative_relu(self):
        return (self.Z > 0).astype(float)

'''
Definition of Model
'''
class Model():
    def __init__(self, units):
        self.layers = []
        self.activations = []
        for i in range(len(units) - 1):
            linear = Linear(units[i], units[i+1], i)
            self.layers.append(linear)
            activation = Sigmoid() #coment this line to test ReLU
            # activation = ReLU() #uncomment this line to test ReLU
            self.activations.append(activation)
    
    def forward(self, X):
        A = X
        for i in range(len(self.layers)):
            tmp = self.layers[i].forward(A) # comment to test without activation functions
            A = self.activations[i].sigmoid(tmp) # comment to test without activation functions
            # A = self.activations[i].relu(tmp) # uncoment this line to test ReLU
            
            # A = self.layers[i].forward(A) # uncomment to test without activation functions
            # print(i)
            # print(tmp)
            # print(A)
        return A
    
    def backward(self, output, Y):
        loss = output - Y 
        dZ = loss * self.activations[-1].derivative_sigmoid() # comment to test without activation functions
        # dZ = loss * self.activations[-1].derivative_relu() # uncoment this line to test ReLU
        
        # dZ = loss # uncomment to test without activation functions

        d_prev_A = self.layers[-1].backward(dZ)
        for i in range(len(self.layers) - 2, -1, -1):
            dZ = d_prev_A * self.activations[i].derivative_sigmoid() # comment to test without activation functions
            # dZ = d_prev_A * self.activations[i].derivative_relu() # uncoment this line to test ReLU
            d_prev_A = self.layers[i].backward(dZ) # comment to test without activation functions
            # print(dZ)

            # dZ = d_prev_A # uncomment to test without activation functions
            # d_prev_A = self.layers[i].backward(dZ) # uncomment to test without activation functions
        return d_prev_A
    
    def update(self, learning_rate=0.05):
        for i in range(len(self.layers)):
            self.layers[i].update(learning_rate)

'''
Test for Model
'''
# model = Model([1,2,1])
# print(model.layers[0].W)
# print(model.layers[0].b)
# A = model.forward(np.array([[0.2, 0.3]]))
# print(A)
# d_prev_A = model.backward(A, np.array([[1.0]]))
# print(d_prev_A)
# model.update()
# print(model.layers[0].W)
# print(model.layers[0].b)

if __name__ == '__main__':

    # data generation
    x, y = generate_linear(n=100)
    x2, y2 = generate_XOR_easy()
    
    # print(x.shape)
    # print(x[0])
    # print(x2.shape)

    # parameters setup for model
    layers_dims = [2, 4, 4, 1]
    # layers_dims = [2, 8, 4, 1]
    # layers_dims = [2, 4, 2, 1]
    learning_rate = 0.01
    # learning_rate = 0.005
    num_epochs = 10000
    num_epochs_ex = 3000
    losses = []
    losses_ex = []
    batch_size = 4

    # train model
    
    # # SGD
    # model = Model(layers_dims)
    # # x_batch = x[0:batch_size].T
    # # print(x_batch.shape)
    # for epoch in range(num_epochs):
    #     loss = 0.0
    #     for i in range(0, x.shape[0], batch_size):  
    #         x_batch = x[i:i+batch_size].T  
    #         y_batch = y[i:i+batch_size].T
    #         # print(y_batch.T)
    #         pred_y = model.forward(x_batch)
    #         # print(pred_y)
    #         loss += np.mean(np.square(y_batch - pred_y))
    #         d_prev_A = model.backward(pred_y, y_batch)
    #         model.update(learning_rate)
    #     loss /= batch_size
    #     if epoch % 100 == 0:
    #         print(f'epoch {epoch} loss : {loss}')
    #         losses.append(loss)


    # result = model.forward(x.T).reshape(x.shape[0],1)
    # # print(result.shape)
    # # print(result)
    # # print(y)
    # loss = np.mean(np.square(y - result))
    
    # accuracy = 0
    # for i in range(x.shape[0]):
    #     print(f'Iter{i} |   Ground truth: {y[i][0]} | prediction: {result[i][0]} |')
    #     if y[i][0] == 0 and result[i][0] < 0.5:
    #         accuracy+=1
    #     elif y[i][0] == 1 and result[i][0] >= 0.5:
    #         accuracy+=1

    # accuracy/=x.shape[0]
    # accuracy*=100

    # # print(pred_y)
    # print(f'loss={loss} accurancy={accuracy}%')
    # show_result(x,y,result)
    # show_learning_rate(losses, learning_rate)

    # gradient descent for each case
    model_ex = Model(layers_dims)
    for epoch in range (num_epochs_ex):
        loss = 0.0
        for iter in range(x.shape[0]):
            pred_y = model_ex.forward(x[iter].reshape(2,1))
            # print(pred_y)
            loss += np.mean(np.square(y[iter] - pred_y))
            # print(loss)
            d_prev_A = model_ex.backward(pred_y, y[iter])
            model_ex.update(learning_rate)
        loss /= x.shape[0]
        if epoch % 100 == 0:
            print(f'epoch {epoch} loss : {loss}')
            losses_ex.append(loss)

    # test model
    result = model_ex.forward(x.T).reshape(x.shape[0],1)
    # print(result.shape)
    print(result)
    # print(y)
    loss = np.mean(np.square(y - result))
    
    accuracy = 0
    for i in range(x.shape[0]):
        print(f'Iter{i} |   Ground truth: {y[i][0]} | prediction: {result[i][0]} |')
        if y[i][0] == 0 and result[i][0] < 0.5:
            accuracy+=1
        elif y[i][0] == 1 and result[i][0] >= 0.5:
            accuracy+=1

    accuracy/=x.shape[0]
    accuracy*=100

    # print(pred_y)
    print(f'loss={loss} accurancy={accuracy}%')
    show_result(x,y,result)
    show_learning_rate(losses_ex, learning_rate)

    # parameters setup for model2
    layers_dims2 = [2, 4, 4, 1]
    # layers_dims2 = [2, 8, 4, 1]
    # layers_dims2 = [2, 4, 2, 1]
    learning_rate2 = 0.01
    # learning_rate2 = 0.005
    num_epochs2 = 1000000
    num_epochs2_ex = 100000
    losses2 = []
    losses2_ex = []
    batch_size2 = 4

    # train model2

    # # SGD
    # model2 = Model(layers_dims2)
    # for epoch in range(num_epochs2):
    #     loss = 0.0
    #     for i in range(0, x2.shape[0], batch_size2):  
    #         x_batch = x2[i:i+batch_size2].T  
    #         y_batch = y2[i:i+batch_size2].T
    #         # print(y_batch.T)
    #         pred_y = model2.forward(x_batch)
    #         # print(pred_y)
    #         loss += np.mean(np.square(y_batch - pred_y))
    #         d_prev_A = model2.backward(pred_y, y_batch)
    #         model2.update(learning_rate2)
    #     loss /= batch_size2
    #     if epoch % 100 == 0:
    #         print(f'epoch {epoch} loss : {loss}')
    #         losses2.append(loss)

    # result = model2.forward(x2.T).reshape(x2.shape[0],1)
    # # print(result.shape)
    # # print(y)
    # loss = np.mean(np.square(y2 - result))
    
    # accuracy = 0
    # for i in range(x2.shape[0]):
    #     print(f'Iter{i} |   Ground truth: {y[i][0]} | prediction: {result[i][0]} |')
    #     if y2[i][0] == 0 and result[i][0] < 0.5:
    #         accuracy+=1
    #     elif y2[i][0] == 1 and result[i][0] >= 0.5:
    #         accuracy+=1

    # accuracy/=x2.shape[0]
    # accuracy*=100

    # # print(pred_y)
    # print(f'loss={loss} accurancy={accuracy}%')
    # show_result(x2,y2,result)
    # show_learning_rate(losses2, learning_rate2)

    # gradient descent for each case
    model2_ex = Model(layers_dims2)
    for epoch in range (num_epochs2_ex):
        loss = 0.0
        for iter in range(x2.shape[0]):
            pred_y = model2_ex.forward(x2[iter].reshape(2,1))
            # print(pred_y)
            loss += np.mean(np.square(y2[iter] - pred_y))
            # print(loss)
            d_prev_A = model2_ex.backward(pred_y, y2[iter])
            model2_ex.update(learning_rate2)
        loss /= x2.shape[0]
        if epoch % 100 == 0:
            print(f'epoch {epoch} loss : {loss}')
            losses2_ex.append(loss)

    # test model
    result = model2_ex.forward(x2.T).reshape(x2.shape[0],1)
    # print(result)
    loss = np.mean(np.square(y2 - result))

    accuracy = 0
    for i in range(x2.shape[0]):
        print(f'Iter{i} |   Ground truth: {y2[i][0]} | prediction: {result[i][0]} |')
        if y2[i][0] == 0 and result[i][0] < 0.5:
            accuracy+=1
        elif y2[i][0] == 1 and result[i][0] >= 0.5:
            accuracy+=1

    accuracy/=x2.shape[0]
    accuracy*=100

    # print(pred_y)
    print(f'loss={loss} accurancy={accuracy}%')
    show_result(x2,y2,result)
    show_learning_rate(losses2_ex, learning_rate2)