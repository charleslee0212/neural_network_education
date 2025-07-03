import math


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def derivative_relu(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    sum_exp_logits = sum(math.exp(logit - max_logit) for logit in logits)
    return [math.exp(logit - max_logit) / sum_exp_logits for logit in logits]
