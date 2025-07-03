from core.neural_network import NeuralNetwork
from util.file_manager import load_images, load_labels

if __name__ == "__main__":
    images_test = load_images("t10k-images.idx3-ubyte")
    labels_test = load_labels("t10k-labels.idx1-ubyte")

    neural_network = NeuralNetwork.load("relu_model.json")

    right = 0

    for i, image in enumerate(images_test):
        neural_network.input(image)
        neural_network.activate()

        choices = neural_network.get_output_activation()
        right += 1 if choices.index(max(choices)) == labels_test[i] else 0

        print(f"Testing in progress: {i}/{len(images_test)}", end="\r")

print("\n")

print(right / len(labels_test))
