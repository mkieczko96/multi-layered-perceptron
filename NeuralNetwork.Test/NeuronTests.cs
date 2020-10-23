using System;
using NeuralNetwork.MLP;
using Xunit;

namespace NeuralNetwork.Test
{
    public class NeuronTests
    {
        [Theory]
        [InlineData(-189.23D, -189.23D)]
        [InlineData(-3, -3)]
        [InlineData(-829.1972F, -829.1972F)]
        [InlineData(0, 0)]
        [InlineData(1892, 1892)]
        [InlineData(1283.31D, 1283.31D)]
        [InlineData(31921.2131F, 31921.2131F)]
        public void Identity_ActivationFunction_Test<T>(T value, T expected)
        {
            var result = Neuron.Identity(value);
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData(-189.23D, 0)]
        [InlineData(-3, 0)]
        [InlineData(-829.1972F, 0)]
        [InlineData(0, 1)]
        [InlineData(1892, 1)]
        [InlineData(1283.31D, 1)]
        [InlineData(31921.2131F, 1)]
        public void BinaryStep_ActivationFunction_Test<T>(T value, T expected)
        {
            var result = Neuron.BinaryStep(value);
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData(-189.23D, 0)]
        [InlineData(-3, 0)]
        [InlineData(-829.1972F, 0)]
        [InlineData(0, 0)]
        [InlineData(1892, 0)]
        [InlineData(1283.31D, 0)]
        [InlineData(31921.2131F, 0)]
        public void Logistic_ActivationFunction_Test<T>(T value, T expected)
        {
            var result = Neuron.Logistic(value);
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData(-189.23D, 0)]
        [InlineData(-3, 0)]
        [InlineData(-829.1972F, 0)]
        [InlineData(0, 0)]
        [InlineData(1892, 0)]
        [InlineData(1283.31D, 0)]
        [InlineData(31921.2131F, 0)]
        public void TanH_ActivationFunction_Test<T>(T value, T expected)
        {
            var result = Neuron.TanH(value);
            Assert.Equal(expected, result);
        }

        [Theory]
        [InlineData(-189.23D, 0)]
        [InlineData(-3, 0)]
        [InlineData(-829.1972F, 0)]
        [InlineData(0, 0)]
        [InlineData(1892, 0)]
        [InlineData(1283.31D, 0)]
        [InlineData(31921.2131F, 0)]
        public void ReLU_ActivationFunction_Test<T>(T value, T expected)
        {
            var result = Neuron.ReLU(value);
            Assert.Equal(expected, result);
        }
    }
}
