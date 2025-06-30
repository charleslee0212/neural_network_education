import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: float) -> float:
    return x if x > 0 else 0


def derivative_relu(x: float) -> float:
    return 1 if x > 0 else 0
