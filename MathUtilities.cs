using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AINumbersRecognize
{
    public class MathUtilities
    {
        Random random = new Random();

        public float RandomFloat(float min, float max)
        {
            int minInt = (int)Math.Round(min * 10000);
            int maxInt = (int)Math.Round(max * 10000);

            int randomInt = random.Next(minInt, maxInt);

            return (float)randomInt / 10000f;
        }

        public float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        public float DerivativeSigmoid(float value)
        {
            return value * (1f - value);
        }

        public float SquareDifference(float a, float b)
        {
            float difference = a - b;
            return difference * difference;
        }
    }
}
