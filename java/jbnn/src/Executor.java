public class Executor {
    public static void main(String[] args) {
        Network network = trainNetwork();
        testNetwork(network);
    }

    public static Network trainNetwork() {
        System.out.println("Beging Training");
        String imagePath = "C:\\projects\\jbnn\\data\\train-images-idx3-ubyte";
        String labelPath = "C:\\projects\\jbnn\\data\\train-labels-idx1-ubyte";

        float[][] images = MNISTTools.normalizeImages255(MNISTTools.readImages(imagePath));
        int[] labels = MNISTTools.readLabels(labelPath);
        Network network = new Network(new int[] { 784, 16, 16, 10 });
        network.train(images, labels, 0.5f, 50);
        int[] trainingValues = new int[labels.length];
        for (int i = 0; i < images.length; i++) {
            float[] classifiedOutput = network.classifyNetwork(images[i]);
            int output = network.singleOutput(classifiedOutput, 0.5f);
            trainingValues[i] = output;
        }
        float accuracy = network.calculateAccuracy(labels, trainingValues);
        System.out.println("Accuracy: " + accuracy);
        return network;
    }

    public static void testNetwork(Network network) {
        System.out.println("Beging Testing");
        String imagePath = "C:\\projects\\jbnn\\data\\t10k-images-idx3-ubyte";
        String labelPath = "C:\\projects\\jbnn\\data\\t10k-labels-idx1-ubyte";
        float[][] images = MNISTTools.normalizeImages255(MNISTTools.readImages(imagePath));
        int[] labels = MNISTTools.readLabels(labelPath);
        int[] trainingValues = new int[labels.length];
        for (int i = 0; i < images.length; i++) {
            float[] classifiedOutput = network.classifyNetwork(images[i]);
            int output = network.singleOutput(classifiedOutput, 0.5f);
            trainingValues[i] = output;
        }
        float accuracy = network.calculateAccuracy(labels, trainingValues);
        System.out.println("Accuracy: " + accuracy);
    }
}