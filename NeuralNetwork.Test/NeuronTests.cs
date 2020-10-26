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
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.Identity(value);

            // Assert
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
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.BinaryStep (value);

            // Assert
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
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.Logistic (value);

            // Assert
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
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.TanH (value);

            // Assert
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
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.ReLU (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0.02975041827262056519481117408175D)]
        [InlineData (-2.20005D, 0.10507833235648884696827279797222D)]
        [InlineData (-1.0D, 0.31326168751822283404899549496786D)]
        [InlineData (0D, 0.69314718055994530941723212145818D)]
        [InlineData (1D, 1.3132616875182228340489954949679D)]
        [InlineData (2.20005D, 2.30512833235648885D)]
        [InlineData (3.5D, 3.52975041827262057D)]
        public void SoftPlus_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.SoftPlus (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-0.5D, -3.5D, 0.48490130828884076D)]
        [InlineData (-0.5D, -2.20005D, 0.4446011908285413D)]
        [InlineData (-0.5D, -1.0D, 0.31606027941427883D)]
        [InlineData (-0.5D, 0D, 0D)]
        [InlineData (-0.5D, 1D, 1D)]
        [InlineData (-0.5D, 2.20005D, 2.20005D)]
        [InlineData (-0.5D, 3.5D, 3.5D)]
        public void ELU_ActivationFunction_Test (double parameter, double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.ELU (parameter, value);
            
            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (1.67326D, -3.5D, -1.7278968985776814992602137076364D)]
        [InlineData (1.67326D, -2.20005D, -1.5633016227320985929217454380341D)]
        [InlineData (1.67326D, -1.0D, -1.1113275400111318727101045428841D)]
        [InlineData (1.67326D, 0D, 0D)]
        [InlineData (1.67326D, 1D, 1.0507D)]
        [InlineData (1.67326D, 2.20005D, 2.311592535D)]
        [InlineData (1.67326D, 3.5D, 3.67745D)]
        public void SELU_ActivationFunction_Test (double parameter, double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.SELU (parameter, value);
            
            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.035D)]
        [InlineData (-2.20005D, -0.0220005D)]
        [InlineData (-1.0D, -0.01D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 1D)]
        [InlineData (2.20005D, 2.20005D)]
        [InlineData (3.5D, 3.5D)]
        public void LeakyReLU_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.LeakyReLU (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-0.5D, -3.5D, 1.75D)]
        [InlineData (-0.5D, -2.20005D, 1.100025D)]
        [InlineData (-0.5D, -1.0D, 0.5D)]
        [InlineData (1D, 0D, 0D)]
        [InlineData (1D, 1D, 1D)]
        [InlineData (1D, 2.20005D, 2.20005D)]
        [InlineData (1D, 3.5D, 3.5D)]
        public void PReLU_ActivationFunction_Test (double parameter, double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.PReLU (parameter, value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -74.0546041D)]
        [InlineData (-2.20005D, -65.55653576D)]
        [InlineData (-1.0D, -45D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 45D)]
        [InlineData (2.20005D, 65.55653576D)]
        [InlineData (3.5D, 74.0546041D)]
        public void ArcTan_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.ArcTan (value);
            
            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.77777777777777777777777777777778D)]
        [InlineData (-2.20005D, -0.68750488273620724676176934735395D)]
        [InlineData (-1.0D, -0.5D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.5D)]
        [InlineData (2.20005D, 0.68750488273620724676176934735395D)]
        [InlineData (3.5D, 0.77777777777777777777777777777778D)]
        public void ElliotSig_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.ElliotSig (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -1D)]
        [InlineData (-2.20005D, -1D)]
        [InlineData (-1.0D, -0.75D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.75D)]
        [InlineData (2.20005D, 1D)]
        [InlineData (3.5D, 1D)]
        public void SQNL_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.SQNL (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -2.1799725276798704322256743771182D)]
        [InlineData (-2.20005D, -1.4917226434343216443408867100337D)]
        [InlineData (-1.0D, -0.79289321881345247559915563789515D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 1.2071067811865475244008443621048D)]
        [InlineData (2.20005D, 2.9083773565656783556591132899663D)]
        [InlineData (3.5D, 4.8200274723201295677743256228818D)]
        public void BentIdentity_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.BentIdentity (value);
            
            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.10259280762974711602851345339964D)]
        [InlineData (-2.20005D, -0.21944618552475462979834452448062D)]
        [InlineData (-1.0D, -0.26894142136999512074884075817816D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.73105857863000487925115924182184D)]
        [InlineData (2.20005D, 1.9806038144752453702016554755194D)]
        [InlineData (3.5D, 3.3974071923702528839714865466004D)]
        public void SiLU_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            //Act
            var result = neuron.SiLU (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, -0.06104853953485687203667013038475D)]
        [InlineData (-2.20005D, -0.03838868110890454448063264826568D)]
        [InlineData (-1.0D, -0.01745240643728351281941897851632D)]
        [InlineData (0D, 0D)]
        [InlineData (1D, 0.01745240643728351281941897851632D)]
        [InlineData (2.20005D, 0.03838868110890454448063264826568D)]
        [InlineData (3.5D, 0.06104853953485687203667013038475D)]
        public void Sin_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.Sin (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0.01744243986710196343904860868136D)]
        [InlineData (-2.20005D, 0.01744900393577625257636537727128D)]
        [InlineData (-1.0D, 0.01745240643728351281941897851632D)]
        [InlineData (-0.5D, 0.01745307099674786992977642794717D)]
        [InlineData (0D, 1D)]
        [InlineData (0.5D, 0.01745307099674786992977642794717D)]
        [InlineData (1D, 0.01745240643728351281941897851632D)]
        [InlineData (2.20005D, 0.01744900393577625257636537727128D)]
        [InlineData (3.5D, 0.01744243986710196343904860868136D)]
        public void Sinc_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.Sinc (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0.00000478511739212901D)]
        [InlineData (-2.20005D, 0.007905314671275481D)]
        [InlineData (-1.0D, 0.36787944117144233D)]
        [InlineData (0D, 1D)]
        [InlineData (1D, 0.36787944117144233D)]
        [InlineData (2.20005D, 0.007905314671275481D)]
        [InlineData (3.5D, 0.00000478511739212901D)]
        public void Gaussian_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();

            // Act
            var result = neuron.Gaussian (value);

            // Assert
            Assert.Equal (expected, result);
        }

        [Theory]
        [InlineData (-3.5D, 0D)]
        [InlineData (-2.20005D, 0D)]
        [InlineData (-1.5D, 0.125D)]
        [InlineData (0D, 1D)]
        [InlineData (1D, 0.5D)]
        [InlineData (2.20005D, 0D)]
        [InlineData (3.5D, 0D)]
        public void SQRBF_ActivationFunction_Test (double value, double expected)
        {
            // Arrange
            var neuron = new Neuron ();
            
            // Act
            var result = neuron.SQRBF (value);
            
            // Assert
            Assert.Equal (expected, result);
        }
    }
}
