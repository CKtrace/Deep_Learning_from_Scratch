import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=False)
    
    return X_test, y_test

def init_network():
    # Learning Weight Parameter
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
        
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

# Batch X

x, t = get_data()
network = init_network()

accuarcy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuarcy_cnt += 1


print(f"Accuracy : {str(float(accuarcy_cnt)/len(x))}")


"""
Batch Application
-> 1. 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리 가능하도록 고도로 최적화되어 있기 때문
-> 2. 커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 자주 있는데, 배치 처리를 함으로써 버스에 주는 부하 감소
"""
x, t = get_data()
network = init_network()

batch_size = 100
accuarcy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuarcy_cnt += np.sum(p==t[i:i+batch_size])

print(f"Accuracy : {str(float(accuarcy_cnt)/len(x))}")
        
 