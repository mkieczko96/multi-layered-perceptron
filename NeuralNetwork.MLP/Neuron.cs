using System;
using static System.Math;

namespace NeuralNetwork.MLP
{
    public class Neuron
    {
        #region Activation Functions
        public double ArcTan (double x) => Atan(x);
        public double BentIdentity (double x) => (Sqrt(Pow(x, 2D) + 1D) - 1D) / 2D + x;
        public double BinaryStep(double x) => x < 0D ? 0D : 1D;
        public double ElliotSig (double x) => x / (1 + Abs(x));
        public double ELU (double a, double x) => x <= 0D ? a*(Exp(x) - 1D) : x;
        public double Gaussian (double x) => Exp(-Pow(x, 2D));
        public double Identity(double x) => x;
        public double LeakyReLU (double x) => x < 0D ? 0.01D * x : x;
        public double Logistic (double x) => 1D / (1D + Exp(-x));
        public double PReLU (double a, double x) => x < 0D ? a * x : x;
        public double ReLU (double x) => Max(0D, x);
        public double SELU (double a, double x) => x < 0D ? 1.0507D * a * (Exp(x) - 1D) : 1.0507D*x;
        public double SiLU (double x) => x / (1D + Exp(-x));
        public double Sin (double x) => Math.Sin(x);
        public double Sinc (double x) => x != 0D ? Math.Sin(x) / x : 1D;
        public double SoftPlus(double x) => Log(1D + Exp(x));
        public double SQNL (double x) => x > 2D ? 1D : x >= 0D ? x - (Pow(x, 2D) / 4D) : x >= -2D ? x + (Pow(x, 2D) / 4D) : -1D;
        public double SQRBF (double x) => Abs (x) <= 1D ? 1D - Pow (x, 2D) / 2D : Abs (x) < 2D ? Pow (2D - Abs (x), 2D) / 2D : 0D;
        public double TanH(double x) => (Exp(x) - Exp(-x)) / (Exp(x) + Exp(-x));
        #endregion

        #region Properties
        #endregion
    }
}
