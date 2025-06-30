from core.neural_network import NeuralNetwork
from util.file_manager import load_images, load_labels


if __name__ == "__main__":
    images_train = load_images("train-images.idx3-ubyte")
    labels_train = load_labels("train-labels.idx1-ubyte")

    neural_network = NeuralNetwork(784, 1, 128, 10)

    for i, image in enumerate(images_train):
        neural_network.input(image)
        neural_network.activate()

        target = [0.0 for _ in range(10)]
        target[labels_train[i]] = 1.0
        neural_network.backpropagate(target=target, learning_rate=0.1)
        print(f"Training in progress: {i}/{len(images_train)}", end="\r")

    neural_network.save()
