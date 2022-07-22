import numpy as np

# 아래와 같이 Softmax 함수를 구현시, Overflow 발생 가능성 존재
def softmax_overflow(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 임의의 정수(일반적으로 입력 신호 중 최댓값)를 통해 Overflow 발생 방지
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 각 클래스의 신호값
a = np.array([0.5, 3.1, 5.2])
y = softmax(a)
# 각 클래스일 확률을 나타냄
print(y)
# 각 클래스일 확률을 모두 더하면 1
print(np.sum(y))