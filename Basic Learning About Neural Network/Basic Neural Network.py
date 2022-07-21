import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_func(x):
    return x

def init_neural_network():
    net = {}
    net['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    net['b1'] = np.array([0.1, 0.2, 0.3])
    net['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    net['b2'] = np.array([0.1, 0.2])
    net['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    net['b3'] = np.array([0.1, 0.2])
    
    return net

def forward_net(net, x): 
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_func(a3)
    
    return y

new_net = init_neural_network()
x = np.array([1.0, 0.5])
y = forward_net(new_net, x)
print(y)