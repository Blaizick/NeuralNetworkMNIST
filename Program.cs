using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using AINumbersRecognize;

public class Program
{
    private const int imageWidth = 28;
    private const int imageHeight = 28;

    private static NeuralNetwork neuralNetwork;

    private static string imagesFolderPath = "D:\\Pictures_for_AI_distinguish\\train";
    private static byte[][] learnImages;
    private static byte[] learnAnswers;

    private static Stopwatch stopwatch = new Stopwatch();
    private static NeuralNetworkSerializer neuralNetworkSerializer = new NeuralNetworkSerializer("D:\\Pictures_for_AI_distinguish\\saves\\s.json");

    private static MathUtilities mathUtilities = new MathUtilities();
    private static Random random = new Random();

    private static void Main(string[] args)
    {
        int inputNeurons = imageWidth * imageHeight;
        int[] sizes = { inputNeurons, 16, 16, 10 };

        neuralNetwork = new NeuralNetwork(0.01f, sizes);

        LoadAllImagesInDirectory();

        for (int epoche = 0; epoche < 1000; epoche++)
        {
            int right = 0;
            int batchSize = 64;

            for (int i = 0; i < batchSize; i++)
            {
                int imageIndex = random.Next(0, learnImages.Length);
                int digit = learnAnswers[imageIndex];

                byte[] image = learnImages[imageIndex];
                float[] imageFloats = new float[image.Length];
                for (int j = 0; j < image.Length; j++)
                {
                    imageFloats[j] = image[j] / 255f;
                }

                float[] outputs = neuralNetwork.ForwardPropagation(imageFloats);

                int predictedDigit = FindAnswer(outputs);

                if (digit == predictedDigit)
                {
                    right++;
                }

                float[] targets = new float[10];
                targets[digit] = 1;

                neuralNetwork.BackPropagation(targets);
            }

            Console.WriteLine(right);
        }

        neuralNetworkSerializer.SaveNeuralNetwork(neuralNetwork);
    }

    private static int FindAnswer(float[] answers)
    {
        float greatestOutput = 0;
        int greatestOutputId = 0;

        for (int i = 0; i < answers.Length; i++)
        {
            if (answers[i] > greatestOutput)
            {
                greatestOutput = answers[i];
                greatestOutputId = i;
            }
        }

        return greatestOutputId;
    }

    private static void LoadAllImagesInDirectory()
    {
        string[] names = Directory.GetFiles(imagesFolderPath);
        
        learnImages = new byte[names.Length][];
        learnAnswers = new byte[names.Length];

        stopwatch.Start();

        Parallel.For(0, names.Length, i => {
            learnAnswers[i] = GetRightAnswerFromPath(names[i]);
            learnImages[i] = LoadImageBytes(names[i]);
        });

        stopwatch.Stop();

        Console.WriteLine("Finished in: " + (stopwatch.ElapsedMilliseconds / 1000f) + "secs.");
        Console.WriteLine("Images loaded: " + names.Length);
    }

    private static byte GetRightAnswerFromPath(string filePath)
    {
        return byte.Parse(filePath[filePath.IndexOf("m") + 1].ToString());
    }

    private static byte[] LoadImageBytes(string path)
    {
        Bitmap bitmap = new Bitmap(path);

        BitmapData bitmapData = bitmap.LockBits(
            new Rectangle(0, 0, bitmap.Width, bitmap.Height),
            ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);

        byte[] bytes = new byte[bitmap.Width * bitmap.Height];

        unsafe
        {
            byte* ptr = (byte*)bitmapData.Scan0;

            int pixel = 0;

            for (int y = 0; y < bitmap.Height; y++)
            {
                byte* row = ptr + (y * bitmapData.Stride);

                for (int x = 0; x < bitmap.Width; x++)
                {
                    byte blue = row[x * 3];
                    byte green = row[x * 3 + 1];
                    byte red = row[x * 3 + 2];

                    bytes[pixel] = (byte)((red + green + blue) / 3);

                    pixel++;
                }
            }
        }

        bitmap.UnlockBits(bitmapData);

        return bytes;
    }
}