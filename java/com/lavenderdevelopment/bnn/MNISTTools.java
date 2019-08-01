package com.lavenderdevelopment.bnn;

import java.io.*;
import java.nio.*;
import java.lang.System;

public final class MNISTTools {
    public static float[][] readImages(String filePath) {
        float[][] images = null;
        try {
            InputStream input = new FileInputStream(filePath);
            byte[] magicNumber = new byte[4];
            input.read(magicNumber);
            byte[] setLength = new byte[4];
            input.read(setLength);
            byte[] rowSize = new byte[4];
            input.read(rowSize);
            byte[] colSize = new byte[4];
            input.read(colSize);
            int imageSize = ByteBuffer.wrap(rowSize).getInt() * ByteBuffer.wrap(colSize).getInt();
            int numOfImages = ByteBuffer.wrap(setLength).getInt();
            images = new float[numOfImages][];
            for (int i = 0; i < numOfImages; i++) {
                byte[] image = new byte[imageSize];
                float[] fimage = new float[imageSize];
                input.read(image);
                for (int j = 0; j < image.length; j++) {
                    // Java only supports signed bytes so we need to convert to unsigned
                    fimage[j] = (float) (image[j] & 0xFF);
                }
                images[i] = fimage;
            }
            input.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            return images;
        }
    }

    public static int[] readLabels(String filePath) {
        int[] labels = null;
        try {
            InputStream input = new FileInputStream(filePath);
            byte[] magicNumber = new byte[4];
            input.read(magicNumber);
            byte[] setLength = new byte[4];
            input.read(setLength);
            int numOfImages = ByteBuffer.wrap(setLength).getInt();
            labels = new int[numOfImages];
            for (int i = 0; i < numOfImages; i++) {
                labels[i] = input.read();
            }
            input.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            return labels;
        }
    }

    public static float[][] normalizeImages255(float[][] images) {
        float[][] normalized = new float[images.length][images[0].length];
        for (int i = 0; i < normalized.length; i++) {
            for (int j = 0; j < normalized[0].length; j++) {
                normalized[i][j] = images[i][j] / 255.0f;
            }
        }
        return normalized;
    }

    public static void printImage(float[] image, int row, int col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (image[(i * row) + j] > 0.0) {
                    System.out.print("*");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }
}
