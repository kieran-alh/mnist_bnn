package com.lavenderdevelopment.bnn;

import java.util.Random;

public class Neuron {
    public float[] weights;
    public float bias;
    public float delta;
    public float output;

    public Neuron(int prevLayerLength) {
        Random rand = new Random(1l);
        this.weights = new float[prevLayerLength];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (float) rand.nextGaussian();
        }
        this.bias = (float) rand.nextGaussian();
        this.delta = 0.0f;
        this.output = 0.0f;
    }
}