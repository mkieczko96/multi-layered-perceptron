using System;
using static System.Math;

namespace NeuralNetwork.MLP
{
    public class Neuron
    {
        #region Activation Functions
        public double Identity(double x) => x;
        public double  BinaryStep(double x) => x < 0 ? 0 : 1;
        public double  Logistic (double x) => 1 / (1 + Exp(-x));
        public double  TanH(double x) => (Exp(x) - Exp(-x)) / (Exp(x) + Exp(-x));
        public double  ReLU (double x) => Max(0, x);
        //public double GELU (double x) => throw new NotImplementedException ();
        public double SoftPlus(double x) => throw new NotImplementedException ();
        public double ELU (double a, double x) => throw new NotImplementedException ();
        public double SELU (double a, double x) => throw new NotImplementedException ();
        public double LeakyReLU (double x) => throw new NotImplementedException ();
        public double PReLU (double a, double x) => throw new NotImplementedException ();
        public double ArcTan (double x) => throw new NotImplementedException ();
        public double ElliotSig (double x) => throw new NotImplementedException ();
        public double SQNL (double x) => throw new NotImplementedException ();
        //public double SReLU (double x) => throw new NotImplementedException ();
        public double BentIdentity (double x) => throw new NotImplementedException ();
        public double SiLU (double x) => throw new NotImplementedException ();
        public double Sin (double x) => throw new NotImplementedException ();
        public double Sinc (double x) => throw new NotImplementedException ();
        public double Gaussian (double x) => throw new NotImplementedException ();
        public double SQRBF (double x) => throw new NotImplementedException ();
        #endregion

        #region Properties
        #endregion
    }
}
