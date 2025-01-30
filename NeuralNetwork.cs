using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AINumbersRecognize
{
    public class NeuralNetwork
    {
        public Layer[] layers;
        private float learningRate;

        private MathUtilities mathUtilities = new();

        public NeuralNetwork()
        {

        }

        public NeuralNetwork(float learningRate, int[] sizes)
        {
            this.learningRate = learningRate;

            layers = new Layer[sizes.Length];

            for (int i = 0; i < sizes.Length; i++)
            {
                int nextSize = 0;
                if (i + 1 < sizes.Length)
                {
                    nextSize = sizes[i + 1];
                }

                Layer layer = new Layer(sizes[i], nextSize);
                layers[i] = layer;
                RandomizeLayer(layer);
            }
        }

        private void RandomizeLayer(Layer layer)
        {
            for (int i = 0; i < layer.size; i++)
            {
                layer.biases[i] = mathUtilities.RandomFloat(-1f, 1f);

                for (int j = 0; j < layer.nextSize; j++)
                {
                    layer.weights[i, j] = mathUtilities.RandomFloat(-1f, 1f);
                }
            }
        }

        public float[] ForwardPropagation(float[] inputs)
        {
            int minLength = Math.Min(inputs.Length, layers[0].activations.Length);
            Array.Copy(inputs, layers[0].activations, minLength);

            for (int i = 1; i < layers.Length; i++)
            {
                Layer layer0 = layers[i - 1];
                Layer layer1 = layers[i];

                for (int j = 0; j < layer1.size; j++)
                {
                    layer1.activations[j] = layer1.biases[j];

                    for (int k = 0; k < layer0.size; k++)
                    {
                        layer1.activations[j] += layer0.activations[k] * layer0.weights[k, j];
                    }

                    layer1.activations[j] = mathUtilities.Sigmoid(layer1.activations[j]);
                }
            }

            return layers[^1].activations;
        }

        public void BackPropagation(float[] targets)
        {
            float[] errors = new float[layers[layers.Length - 1].size];

            for (int i = 0; i < layers[^1].size; i++)
            {
                errors[i] = targets[i] - layers[^1].activations[i];
            }

            for (int i = layers.Length - 2; i >= 0; i--)
            {
                Layer layer0 = layers[i];
                Layer layer1 = layers[i + 1];

                float[] gradient = new float[layer1.size];

                for (int j = 0; j < layer1.size; j++)
                {
                    gradient[j] = errors[j] * mathUtilities.DerivativeSigmoid(layer1.activations[j]) * learningRate;
                }

                float[,] weightsDeltas = new float[layer0.size, layer1.size];

                for (int j = 0; j < layer0.size; j++)
                {
                    for (int k = 0; k < layer1.size; k++)
                    {
                        weightsDeltas[j, k] = layer0.activations[j] * gradient[k];
                    }
                }

                float[] nextErrors = new float[layer0.size];

                for (int j = 0; j < layer0.size; j++)
                {
                    nextErrors[j] = 0;
                    for (int k = 0; k < layer1.size; k++)
                    {
                        nextErrors[j] += layer0.weights[j, k] * errors[k];
                    }
                }

                errors = nextErrors;

                for (int j = 0; j < layer0.size; j++)
                {
                    for (int k = 0; k < layer1.size; k++)
                    {
                        layer0.weights[j, k] += weightsDeltas[j, k];
                    }
                }

                for (int j = 0; j < layer1.size; j++)
                {
                    layer1.biases[j] += gradient[j];
                }
            }
        }
    }

    public class Layer
    {
        public int size;
        public int nextSize;

        public float[] activations;

        public float[] biases;
        public float[,] weights;
        
        public Layer()
        {

        }

        public Layer(int size, int nextSize)
        {
            this.size = size;
            this.nextSize = nextSize;

            activations = new float[size];

            biases = new float[size];
            weights = new float[size, nextSize];
        }
    }
}
