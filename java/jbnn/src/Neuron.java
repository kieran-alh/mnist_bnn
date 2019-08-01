import java.util.Random;

public class Neuron {
    public static Random rand = new Random(1L);
    public float[] weights;
    public float bias;
    public float delta;
    public float output;

    public Neuron(int prevLayerLength) {
        this.weights = new float[prevLayerLength];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (float) rand.nextGaussian();
        }
        this.bias = (float) rand.nextGaussian();
        this.delta = 0.0f;
        this.output = 0.0f;
    }
}