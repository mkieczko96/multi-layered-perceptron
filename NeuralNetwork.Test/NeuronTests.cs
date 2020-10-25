using System;
using NeuralNetwork.MLP;
using Xunit;

namespace NeuralNetwork.Test
{
    public class NeuronTests
    {
        [Theory]
        [InlineData (-3.5D, -3.5D)]
        [InlineData (-2.20005D, -2.20005D)]
        [InlineData (-1.0D, -1.0D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 1D)]
        [InlineData (2.20005D, 2.20005D)]
        [InlineData (3.5D, 3.5D)]
        public void Identity_ActivationFunction_Test(double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.Identity(value);
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0D)]
        [InlineData (-2.20005D, 0D)]
        [InlineData (-1.0D, 0D)]
        [InlineData (0D, 1D)]
        [InlineData (1D, 1D)]
        [InlineData (2.20005D, 1D)]
        [InlineData (3.5D, 1D)]
        public void BinaryStep_ActivationFunction_Test(double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.BinaryStep (value);
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0.02931223075135632D)]
        [InlineData (-2.20005D, 0.09974599919308862D)]
        [InlineData (-1.0D, 0.2689414213699951D)]
        [InlineData (0D, 0.5D)]
        [InlineData (1D, 0.7310585786300049D)]
        [InlineData (2.20005D, 0.9002540008069114D)]
        [InlineData (3.5D, 0.9706877692486436D)]
        public void Logistic_ActivationFunction_Test(double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.Logistic (value);
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.9981778976111987D)]
        [InlineData (-2.20005D, -0.9757455261817578D)]
        [InlineData (-1.0D, -0.7615941559557649D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.7615941559557649D)]
        [InlineData (2.20005D, 0.9757455261817578D)]
        [InlineData (3.5D, 0.9981778976111987D)]
        public void TanH_ActivationFunction_Test(double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.TanH (value);
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData(-3.5D, 0D)]
        [InlineData(-2.20005D, 0D)]
        [InlineData(-1.0D, 0D)]
        [InlineData(0D, 0D)]
        [InlineData(1D, 1D)]
        [InlineData(2.20005D, 2.20005D)]
        [InlineData(3.5D, 3.5D)]
        public void ReLU_ActivationFunction_Test (double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.ReLU (value);
            Assert.Equal (expected, result);
        }
    }
}
