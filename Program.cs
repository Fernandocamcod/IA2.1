using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

class NeuralNetwork
{
    private List<List<double>> weights;
    private double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, double learningRate)
    {
        this.learningRate = learningRate;
        weights = new List<List<double>>
        {
            new List<double>(new double[hiddenSize * inputSize]),
            new List<double>(new double[3 * hiddenSize]) // Assuming 3 output classes
        };

        // Initialize random weights
        Random rand = new Random();
        for (int i = 0; i < weights.Count; i++)
        {
            for (int j = 0; j < weights[i].Count; j++)
            {
                weights[i][j] = rand.NextDouble() * 2 - 1; // Weights between -1 and 1
            }
        }
    }

    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private double SigmoidDerivative(double x) => x * (1 - x);

    public List<double> Predict(List<double> input)
    {
        int inputSize = weights[0].Count / input.Count;
        int hiddenSize = weights[0].Count / inputSize;

        // Hidden layer
        List<double> hidden = new List<double>(new double[hiddenSize]);
        for (int i = 0; i < hidden.Count; i++)
        {
            for (int j = 0; j < input.Count; j++)
            {
                hidden[i] += input[j] * weights[0][i * input.Count + j];
            }
            hidden[i] = Sigmoid(hidden[i]);
        }

        // Output layer
        int outputSize = weights[1].Count / hidden.Count;
        List<double> output = new List<double>(new double[outputSize]);
        for (int i = 0; i < output.Count; i++)
        {
            for (int j = 0; j < hidden.Count; j++)
            {
                output[i] += hidden[j] * weights[1][i * hidden.Count + j];
            }
            output[i] = Sigmoid(output[i]);
        }

        return output;
    }

    public void Train(List<double> input, List<double> target)
    {
        int inputSize = weights[0].Count / input.Count;
        int hiddenSize = weights[0].Count / inputSize;

        // Forward pass
        List<double> hidden = new List<double>(new double[hiddenSize]);
        for (int i = 0; i < hidden.Count; i++)
        {
            for (int j = 0; j < input.Count; j++)
            {
                hidden[i] += input[j] * weights[0][i * input.Count + j];
            }
            hidden[i] = Sigmoid(hidden[i]);
        }

        int outputSize = weights[1].Count / hidden.Count;
        List<double> output = new List<double>(new double[outputSize]);
        for (int i = 0; i < output.Count; i++)
        {
            for (int j = 0; j < hidden.Count; j++)
            {
                output[i] += hidden[j] * weights[1][i * hidden.Count + j];
            }
            output[i] = Sigmoid(output[i]);
        }

        // Backward pass (weight adjustment)
        List<double> outputErrors = new List<double>(new double[output.Count]);
        for (int i = 0; i < output.Count; i++)
        {
            outputErrors[i] = target[i] - output[i];
            for (int j = 0; j < hidden.Count; j++)
            {
                weights[1][i * hidden.Count + j] += learningRate * outputErrors[i] * SigmoidDerivative(output[i]) * hidden[j];
            }
        }

        List<double> hiddenErrors = new List<double>(new double[hidden.Count]);
        for (int i = 0; i < hidden.Count; i++)
        {
            for (int j = 0; j < output.Count; j++)
            {
                hiddenErrors[i] += outputErrors[j] * weights[1][j * hidden.Count + i];
            }
            for (int j = 0; j < input.Count; j++)
            {
                weights[0][i * input.Count + j] += learningRate * hiddenErrors[i] * SigmoidDerivative(hidden[i]) * input[j];
            }
        }
    }
}

class Program
{
    static Dictionary<string, List<double>> LoadEmbeddings(string path)
    {
        try
        {
            var json = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<Dictionary<string, List<double>>>(json);
        }
        catch (UnauthorizedAccessException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return null;
        }
        catch (FileNotFoundException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return null;
        }
    }

    static string GetEmbeddingsPath()
    {
        return @"C:\Users\USUARIO\embeddings.json";
    }

    static void Main()
    {
        var embeddingsPath = GetEmbeddingsPath();
        var tokenEmbeddings = LoadEmbeddings(embeddingsPath);

        if (tokenEmbeddings == null)
        {
            Console.WriteLine("No se pudieron cargar los embeddings. Verifique la ruta y los permisos del archivo.");
            return;
        }

        // Ejemplos de entrenamiento (inputs representando fragmentos de código)
        List<List<double>> trainingInputs;
        List<List<double>> trainingOutputs;

        try
        {
            trainingInputs = new List<List<double>>
            {
                tokenEmbeddings["digitalWrite"].Concat(tokenEmbeddings["HIGH"]).ToList(),
                tokenEmbeddings["analogRead"].Concat(tokenEmbeddings["LOW"]).ToList(),
                tokenEmbeddings["Serial.println"].Concat(tokenEmbeddings["sensorValue"]).ToList(),
                tokenEmbeddings["pinMode"].Concat(tokenEmbeddings["OUTPUT"]).ToList(),
                // Agregar más ejemplos aquí...
            };

            // Salidas correspondientes (representan clases de código)
            trainingOutputs = new List<List<double>>
            {
                new List<double> { 1, 0, 0 }, // Clase 1
                new List<double> { 0, 1, 0 }, // Clase 2
                new List<double> { 0, 0, 1 }, // Clase 3
                // Agregar más salidas aquí...
            };
        }
        catch (KeyNotFoundException ex)
        {
            Console.WriteLine($"Error: {ex.Message}. Verifique que todos los tokens necesarios estén presentes en el archivo de embeddings.");
            return;
        }

        // Dividir datos en entrenamiento y prueba
        int trainSize = Math.Min((int)(trainingInputs.Count * 0.8), trainingOutputs.Count);
        var trainInputs = trainingInputs.Take(trainSize).ToList();
        var trainOutputs = trainingOutputs.Take(trainSize).ToList();
        var testInputs = trainingInputs.Skip(trainSize).ToList();
        var testOutputs = trainingOutputs.Skip(trainSize).ToList();

        // Parámetros para experimentar
        double[] learningRates = { 0.01, 0.05, 0.1 };
        int[] hiddenSizes = { 10, 20, 30 };

        foreach (var learningRate in learningRates)
        {
            foreach (var hiddenSize in hiddenSizes)
            {
                NeuralNetwork nn = new NeuralNetwork(trainInputs[0].Count, hiddenSize, learningRate); // Cambiado el tamaño de entrada

                // Entrenar la red neuronal
                for (int epoch = 0; epoch < 50000; epoch++)
                {
                    for (int i = 0; i < trainInputs.Count; i++)
                    {
                        nn.Train(trainInputs[i], trainOutputs[i]);
                    }

                    // Monitorear rendimiento en el conjunto de prueba
                    if (epoch % 1000 == 0) // Cada 1000 épocas
                    {
                        double totalError = 0;
                        for (int i = 0; i < testInputs.Count; i++)
                        {
                            var output = nn.Predict(testInputs[i]);
                            totalError += testOutputs[i].Select((t, j) => Math.Pow(t - output[j], 2)).Sum(); // Error cuadrático
                        }
                        double averageError = totalError / testInputs.Count;
                        Console.WriteLine($"Epoch: {epoch}, Learning Rate: {learningRate}, Hidden Size: {hiddenSize}, Average Error: {averageError}");
                    }
                }
            }
        }

        Console.WriteLine("Entrenamiento completado. Presiona cualquier tecla para continuar...");
        Console.ReadKey();
    }
}
