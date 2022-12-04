import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

x1, x2 = 1, 1

print(f"AND Gate : {AND(x1, x2)}")
print(f"NAND Gate : {NAND(x1, x2)}")
print(f"OR Gate : {OR(x1, x2)}")
print(f"XOR Gate : {XOR(x1, x2)}")

x1, x2 = 1, 0

print(f"AND Gate : {AND(x1, x2)}")
print(f"NAND Gate : {NAND(x1, x2)}")
print(f"OR Gate : {OR(x1, x2)}")
print(f"XOR Gate : {XOR(x1, x2)}")
