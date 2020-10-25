using System;
using static System.Math;

namespace NeuralNetwork.MLP
{
    public class Neuron
    {
        public double Identity(double x) => x;
        public double  BinaryStep(double x) => x < 0 ? 0 : 1;
        public double  Logistic (double x) => 1 / (1 + Exp(-x));
        public double  TanH(double x) => (Exp(x) - Exp(-x)) / (Exp(x) + Exp(-x));
        public double  ReLU (double x) => Max(0, x);
    }
}
