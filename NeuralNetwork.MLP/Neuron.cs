using System;

namespace NeuralNetwork.MLP
{
    public class Neuron
    {
        public double Identity(double x) => x;
        public double  BinaryStep(double x) => throw new NotImplementedException();
        public double  Logistic (double x) => throw new NotImplementedException();
        public double  TanH(double x) => throw new NotImplementedException();
        public double  ReLU (double x) => throw new NotImplementedException();
    }
}
