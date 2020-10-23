using System;
using NeuralNetwork.MLP;
using Xunit;

namespace NeuralNetwork.Test
{
    public class NeuronTests
    {
        [Theory]
        [InlineData(-189.23D)]
        [InlineData(-3)]
        [InlineData(-829.1972F)]
        [InlineData(0)]
        [InlineData(1892)]
        [InlineData(1283.31D)]
        [InlineData(31921.2131F)]
        public void Identity_ActivationFunction_Test<T>(T value)
        {
            var result = Neuron.Identity(value);
            Assert.Equal(value, result);
        }
    }
}
