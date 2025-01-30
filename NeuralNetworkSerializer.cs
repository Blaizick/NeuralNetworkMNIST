using Newtonsoft.Json;
using System.Text.Json;

namespace AINumbersRecognize
{
    public class NeuralNetworkSerializer
    {
        private string savePath;

        public NeuralNetworkSerializer(string savePath) 
        { 
            this.savePath = savePath;
        }

        public void SaveNeuralNetwork(NeuralNetwork data)
        {
            string json = JsonConvert.SerializeObject(data, Formatting.Indented);

            File.WriteAllText(savePath, json);
        }

        public NeuralNetwork LoadNeuralNetwork()
        {
            if (!File.Exists(savePath))
            {
                return null;
            }

            string json = File.ReadAllText(savePath);

            return JsonConvert.DeserializeObject<NeuralNetwork>(json);
        }
    }
}
