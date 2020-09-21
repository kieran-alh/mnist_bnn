public class Neuron {
    public float[] weights;
    public float bias;
    public float delta;
    public float output;

    public Neuron(int prevLayerLength) {
        this.weights = new float[prevLayerLength];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (float) Activations.rand.nextGaussian();
        }
        this.bias = (float) Activations.rand.nextGaussian();
        this.delta = 0.0f;
        this.output = 0.0f;
    }
}