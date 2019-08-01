package com.lavenderdevelopment.bnn;

import java.lang.Math;

public class Network {
    public Layer[] layers;

    public Network(int[] initialLayers) {
        this.layers = new Layer[initialLayers.length - 1];
        for (int i = 0; i < initialLayers.length - 1; i++) {
            this.layers[i] = new Layer(initialLayers[i], initialLayers[i + 1]);
        }
    }

    public float unactivatedOutput(float[] weights, float[] inputs, float bias) {
        float result = bias;
        for (int i = 0; i < weights.length; i++) {
            result += weights[i] * inputs[i];
        }
        return result;
    }

    public float activation(float output, String function) {
        if (function == "tanh") {
            return (float) Math.tanh(output);
        } else {
            return Activations.sigmoid(output);
        }
    }

    public float derivate(float output, String function) {
        if (function == "tanh") {
            return Activations.dTanh(output);
        } else {
            return Activations.dSigmoid(output);
        }
    }

    public float[] forwardPropogation(float[] initialInputs, boolean readOnly) {
        float[] inputs = initialInputs;
        for (int i = 0; i < this.layers.length; i++) {
            float[] newInputs = new float[this.layers[i].neurons.length];
            for (int j = 0; j < this.layers[i].neurons.length; j++) {
                float output = unactivatedOutput(this.layers[i].neurons[j].weights, inputs,
                        this.layers[i].neurons[j].bias);
                output = activation(output, "tanh");
                if (!readOnly) {
                    this.layers[i].neurons[j].output = output;
                }
                newInputs[j] = output;
            }
            inputs = newInputs;
        }
        return inputs;
    }

    public void backwardPropogation(float[] expected) {
        for (int i = this.layers.length - 1; i >= 0; i--) {
            float[] errors = new float[this.layers[i].neurons.length];
            for (int j = 0; j < this.layers[i].neurons.length; j++) {
                if (i == this.layers.length - 1) {
                    errors[j] = expected[j] - this.layers[i].neurons[j].output;
                } else {
                    float error = 0.0f;
                    for (int k = 0; k < this.layers[i + 1].neurons.length; k++) {
                        error += this.layers[i + 1].neurons[k].weights[j] * this.layers[i + 1].neurons[k].delta;
                    }
                    errors[j] = error;
                }
            }
            for (int j = 0; j < this.layers[i].neurons.length; j++) {
                this.layers[i].neurons[j].delta = errors[j] * derivate(this.layers[i].neurons[j].output, "tanh");
            }
        }
    }

    public void updateWeights(float[] initialInputs, float learnRate) {
        float[] inputs = initialInputs;
        for (int i = 0; i < this.layers.length; i++) {
            if (i != 0) {
                inputs = new float[this.layers[i - 1].neurons.length];
                for (int j = 0; j < this.layers[i - 1].neurons.length; j++) {
                    inputs[j] = this.layers[i - 1].neurons[j].output;
                }
            }
            for (int j = 0; j < this.layers[i].neurons.length; j++) {
                for (int k = 0; k < inputs.length; j++) {
                    float weightDelta = learnRate * this.layers[i].neurons[j].delta * inputs[k];
                    this.layers[i].neurons[j].weights[k] += weightDelta;
                }
                this.layers[i].neurons[j].bias += learnRate * this.layers[i].neurons[j].delta;
            }
        }
    }

    public float sumErrors(float[] expected, float[] output) {
        float result = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            result += Math.pow((expected[i] - output[i]), 2.0);
        }
        return result;
    }

    public float[] classifyNetwork(float[] inputs) {
        return this.forwardPropogation(inputs, true);
    }

    public int singleOutput(float[] outputs, float threshold) {
        float max = outputs[0];
        int index = 0;
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > max && outputs[i] >= threshold) {
                max = outputs[i];
                index = i;
            }
        }
        return index;
    }

    public float calculateAccuracy(int[] expected, int[] output) {
        float result = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == output[i]) {
                result += 1.0f;
            }
        }
        return result / expected.length;
    }

    public Network train(float[][] trainData, int[] trainLabels, float learnRate, int epochs) {
        float sumError = 0.0f;
        for (int i = 0; i < epochs; i++) {
            sumError = 0.0f;
            for (int j = 0; j < trainData.length; j++) {
                float[] outputs = this.forwardPropogation(trainData[j], false);
                float[] expected = new float[this.layers[this.layers.length - 1].neurons.length];
                expected[trainLabels[i]] = 1.0f;
                sumError += sumErrors(expected, outputs);
                this.backwardPropogation(expected);
                this.updateWeights(trainData[i], learnRate);
            }
            System.out.format(">epoch=%d | lrate=%f | error=%f %n", i, learnRate, sumError);
        }
        return this;
    }
}
