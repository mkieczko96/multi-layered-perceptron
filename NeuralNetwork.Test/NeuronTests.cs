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
        [InlineData (-3.5D, 0.0293122307513563D)]
        [InlineData (-2.20005D, 0.09974599919308860D)]
        [InlineData (-1.0D, 0.2689414213699950D)]
        [InlineData (0D, 0.5D)]
        [InlineData (1D, 0.7310585786300050D)]
        [InlineData (2.20005D, 0.9002540008069110D)]
        [InlineData (3.5D, 0.9706877692486440D)]
        public void Logistic_ActivationFunction_Test(double value, double expected)
        {
            var neuron = new Neuron ();
            var result = neuron.Logistic (value);
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.998177897611199D)]
        [InlineData (-2.20005D, -0.975745526181758D)]
        [InlineData (-1.0D, -0.761594155955765D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.761594155955765D)]
        [InlineData (2.20005D, 0.975745526181758D)]
        [InlineData (3.5D, 0.998177897611199D)]
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
