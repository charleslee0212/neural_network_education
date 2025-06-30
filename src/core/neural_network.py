from .neuron import Neuron
from util.functions import derivative_sigmoid
import itertools
import json
import os


class NeuralNetwork:
    def __init__(
        self,
        num_of_input_neurons: int,
        num_of_hidden_layers: int,
        num_of_hidden_neurons: int,
        num_of_output_neurons: int,
    ):
        assert (
            num_of_input_neurons > 0
            and num_of_hidden_layers > 0
            and num_of_hidden_neurons > 0
            and num_of_output_neurons > 0
        )
        self.meta = [
            num_of_input_neurons,
            num_of_hidden_layers,
            num_of_hidden_neurons,
            num_of_output_neurons,
        ]
        self.input_layer: list[Neuron] = [Neuron() for _ in range(num_of_input_neurons)]

        hidden_layers: list[list[Neuron]] = []
        for i in range(num_of_hidden_layers):
            layer = []
            for _ in range(num_of_hidden_neurons):
                neuron = Neuron()
                if i == 0:
                    neuron.set_inputs(self.input_layer)
                else:
                    neuron.set_inputs(hidden_layers[i - 1])
                neuron.initialize()
                layer.append(neuron)
            hidden_layers.append(layer)

        self.hidden_layers: list[list[Neuron]] = hidden_layers

        output_layer: list[Neuron] = []
        for _ in range(num_of_output_neurons):
            neuron = Neuron()
            neuron.set_inputs(hidden_layers[-1])
            neuron.initialize()
            output_layer.append(neuron)

        self.output_layer: list[Neuron] = output_layer

    def input(self, inputs: list[float]) -> None:
        assert len(inputs) == len(self.input_layer)
        for i, input in enumerate(inputs):
            self.input_layer[i].activation = input

    def activate(self) -> None:
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.activate()

        for neuron in self.output_layer:
            neuron.activate()

    def backpropagate(self, target: list[float], learning_rate: float) -> None:
        for i, neuron in enumerate(self.output_layer):
            error = 2 * (neuron.activation - target[i])
            neuron.delta = derivative_sigmoid(neuron.z) * error

        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            for j, neuron in enumerate(layer):
                next_layer = (
                    self.output_layer
                    if i == len(self.hidden_layers) - 1
                    else self.hidden_layers[i + 1]
                )
                error = sum(
                    next_neuron.weights[j] * next_neuron.delta
                    for next_neuron in next_layer
                )
                neuron.delta = derivative_sigmoid(neuron.z) * error

        for neuron in self.output_layer + list(
            itertools.chain.from_iterable(self.hidden_layers)
        ):
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= (
                    learning_rate * neuron.inputs[i].activation * neuron.delta
                )
            neuron.bias -= learning_rate * neuron.delta

    def get_output_activation(self):
        return list(map(lambda neuron: neuron.activation, self.output_layer))

    def save(self, file_name="model.json") -> None:
        dir = "./trained_model"
        os.makedirs(dir, exist_ok=True)

        file_path = os.path.join(dir, file_name)
        state = {
            "meta": self.meta,
            "input_layer": len(self.input_layer),
            "hidden_layers": [
                [{"weights": neuron.weights, "bias": neuron.bias} for neuron in layer]
                for layer in self.hidden_layers
            ],
            "output_layer": [
                {"weights": neuron.weights, "bias": neuron.bias}
                for neuron in self.output_layer
            ],
        }

        with open(file_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, file_name):
        dir = "./trained_model"

        file_path = os.path.join(dir, file_name)
        with open(file_path, "r") as f:
            state = json.load(f)

        (
            num_of_input_neurons,
            num_of_hidden_layers,
            num_of_hidden_neurons,
            num_of_output_neurons,
        ) = state["meta"]

        model = cls(
            num_of_input_neurons,
            num_of_hidden_layers,
            num_of_hidden_neurons,
            num_of_output_neurons,
        )

        for layer, saved_layer in zip(model.hidden_layers, state["hidden_layers"]):
            for neuron, saved_neuron in zip(layer, saved_layer):
                neuron.weights = saved_neuron["weights"]
                neuron.bias = saved_neuron["bias"]

        for neuron, saved_neuron in zip(model.output_layer, state["output_layer"]):
            neuron.weights = saved_neuron["weights"]
            neuron.bias = saved_neuron["bias"]

        return model
