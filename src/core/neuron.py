from util.functions import sigmoid
import math
import random


class Neuron:
    def __init__(self):
        self.inputs: list[Neuron] = []
        self.weights: list[float] = []
        self.activation: float = 0.0
        self.bias: float = 0.0
        self.z: float = 0.0
        self.delta: float = 0.0

    def set_inputs(self, value: list["Neuron"]) -> None:
        self.inputs = value

    def initialize(self) -> None:
        num_input = len(self.inputs)
        assert num_input > 0
        limit = 1 / math.sqrt(num_input)
        self.weights = [random.uniform(-limit, limit) for _ in range(num_input)]
        self.bias = random.uniform(-limit, limit)

    def activate(self) -> None:
        num_weights = len(self.weights)
        assert num_weights > 0
        total = 0.0
        for i in range(num_weights):
            total += self.weights[i] * self.inputs[i].activation
        self.z = total + self.bias
        self.activation = sigmoid(self.z)
