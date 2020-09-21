import java.lang.Math;
import java.util.Random;

public final class Activations {
    public static Random rand = new Random();
    public static float sigmoid(float input) {
        return 1.0f / (1.0f + (float) Math.exp(-input));
    }

    public static float tanh(float input) {
        return (float)(Math.sinh(input) / Math.cosh(input));
    }

    public static float dSigmoid(float output) {
        return output * (1.0f - output);
    }

    public static float dTanh(float output) {
        return 1.0f - (output * output);
    }
}