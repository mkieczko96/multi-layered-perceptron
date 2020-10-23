using System;

namespace NeuralNetwork.MLP
{
    public class Neuron
    {
        public static T Identity<T>(T x) => x;
        public static T  BinaryStep<T> (T x) => throw new NotImplementedException();
        public static T  Logistic<T> (T x) => throw new NotImplementedException();
        public static T  TanH<T> (T x) => throw new NotImplementedException();
        public static T  ReLU<T> (T x) => throw new NotImplementedException();
    }
}
