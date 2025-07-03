from util.functions import sigmoid, relu
import math
import random


class Neuron:
    def __init__(self, activation_function: str = "sigmoid"):
        self.inputs: list[Neuron] = []
        self.weights: list[float] = []
        self.activation: float = 0.0
        self.bias: float = 0.0
        self.z: float = 0.0
        self.delta: float = 0.0
        assert activation_function == "sigmoid" or activation_function == "relu"
        self.activation_function = activation_function

    def set_inputs(self, value: list["Neuron"]) -> None:
        self.inputs = value

    def initialize(self) -> None:
        num_input = len(self.inputs)
        assert num_input > 0
        if self.activation_function == "sigmoid":
            limit = 1 / math.sqrt(num_input)
            self.weights = [random.uniform(-limit, limit) for _ in range(num_input)]
            self.bias = random.uniform(-limit, limit)
        if self.activation_function == "relu":
            stddev = math.sqrt(2 / num_input)
            self.weights = [random.gauss(0, stddev) for _ in range(num_input)]
            self.bias = 0.0

    def activate(self) -> None:
        num_weights = len(self.weights)
        assert num_weights > 0
        total = 0.0
        for i in range(num_weights):
            total += self.weights[i] * self.inputs[i].activation
        self.z = total + self.bias
        if self.activation_function == "sigmoid":
            self.activation = sigmoid(self.z)
        if self.activation_function == "relu":
            self.activation = relu(self.z)
