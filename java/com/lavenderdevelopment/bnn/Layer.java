package com.lavenderdevelopment.bnn;

public class Layer {
    public Neuron[] neurons;

    public Layer(int prevLayerLength, int curLayerLength) {
        this.neurons = new Neuron[curLayerLength];
        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i] = new Neuron(prevLayerLength);
        }
    }
}