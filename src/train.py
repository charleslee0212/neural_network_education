from core.neural_network import NeuralNetwork
from util.file_manager import load_images, load_labels


if __name__ == "__main__":
    images_train = load_images("train-images.idx3-ubyte")
    labels_train = load_labels("train-labels.idx1-ubyte")

    neural_network = NeuralNetwork(784, 4, 128, 10, activation_function="relu")

    epoch = 5

    for i in range(epoch):
        for j, image in enumerate(images_train):
            neural_network.input(image)
            neural_network.activate()

            target = [0.0 for _ in range(10)]
            target[labels_train[j]] = 1.0
            neural_network.backpropagate(target=target, learning_rate=0.01)
            print(
                f"Epoch: {i + 1} | Training in progress: {j}/{len(images_train)}",
                end="\r",
            )

    neural_network.save("relu_model.json")
