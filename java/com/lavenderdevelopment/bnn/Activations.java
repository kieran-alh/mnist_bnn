package com.lavenderdevelopment.bnn;

import java.lang.Math;

public final class Activations {
    public static float sigmoid(float input) {
        return 1.0f / (1.0f + (float) Math.exp((double) input));
    }

    public static float dSigmoid(float output) {
        return output * (1 - output);
    }

    public static float dTanh(float output) {
        return 1 - (output * output);
    }
}