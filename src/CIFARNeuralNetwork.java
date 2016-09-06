import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class CIFARNeuralNetwork {

	public CIFARNeuralNetwork() {
		// TODO Auto-generated constructor stub
	}

	 // helper variables for accessibility
	 private static final int 	INPUT_LAYER = 0;
	 private int 				OUTPUT_LAYER = -1;

	 // Contains our layer of neurons
	 private Neuron[][] m_neurons = null;
	 
	 // how fast our network can learn
	 private double m_learnRate = 0.1;
	 
	// our activation function
	private static double sigmoid(double z) {
			
		return 1.0 / (1.0 + Math.exp(-z));	
	}
	
	// inverse activation function
	private static double inv_sigmoid(double y) {
	
		return Math.log(y / (1.0 - y));
	}
	
	// attach the image to the Input Neurons to prepare for training or testing
	private void attachImage(byte[] pixelData) {
		
		for(int i = 0; i < pixelData.length; i++) {
			int a = pixelData[i] & 0xff;
			m_neurons[INPUT_LAYER][i].setOutput((double)a / 255.0);
		}
	}

	// create a vector from the output values of the Output Neurons
	public double[] vectorFromOutput() {
		
		double[] vector = new double[m_neurons[OUTPUT_LAYER].length];
		for(int i = 0; i < vector.length; i++) {
			vector[i] = m_neurons[OUTPUT_LAYER][i].getOutput();
		}
		return vector;
	}

	// given a label (0-N), initialize a target vector to match (e.g. 3 = [0, 0, 0, 1.0, 0.0])
	private double[] vectorFromLabel(int label) {

		double[] vector = new double[m_neurons[OUTPUT_LAYER].length];	
		for(int i = 0; i < vector.length; i++) {
			if(i == label) 
				vector[i] = 1.0;
			else				
				vector[i] = 0.0;
		}
		
		return vector;		
	}

	// return the label of the network's output as an integer from 0-9 (ideally!)
	public int labelFromVector(double[] vector) {
		
		int index = -1;  		// means not found
		double bigValue = -1000.0;	// should be small enough, right?
		
		for(int i = 0; i < vector.length; i++) {
			if(vector[i] > bigValue) {
				index = i;
				bigValue = vector[i];
			}
		}
		
		return index;		
	}
		
	// propagate the values from one layer to the next
	private void forwardPropagate() {
		
		// starting with first hidden layer, let's sum forward our values..
		for(int i = INPUT_LAYER+1; i <= OUTPUT_LAYER; i++) {
			
			for(Neuron n:m_neurons[i]) {
				double sum = 0.0;					
				for(int j = 0; j < n.getNumWeights(); j++) {
					sum += n.getWeight(j) * m_neurons[i-1][j].getOutput();
				}
				
				n.setOutput(sigmoid(sum + n.getBias()));
			}
		}		
	}

	// First attempt at back propagation on a multi-layer network (hidden layers > 1)
	// 'vT' represents the ideal vector that the output neurons should register
	// it is created by turning the file label into a 10-dimensional vector of binary values [0.0,1.0]
	// 'debug' turns on/off console logging (performance)
	private void backPropagate(double[] vT, boolean debug) {
	
		// going to calculate total error in order to dynamically adjust learn rate as error drops...
		double totalE = 0.0;
		
		// let's start by getting the error relative to each output node
		// right now the error value is temporarily housed in the neuron, but for optimization
		// purposes, we can probably make a more transient store for this later
		for(int i = 0; i < m_neurons[OUTPUT_LAYER].length; i++) {
			
			Neuron n = m_neurons[OUTPUT_LAYER][i];
		
			// the error with respect to the activated output
			// the error with respect to the net output
			double dEdA = -(vT[i] - n.getOutput());
			double dEdN = n.getOutput() * (1.0 - n.getOutput());
			totalE += dEdA * dEdA * 0.5;

			// total error on the net side of this neuron
			n.setError(dEdA * dEdN);
		}
		
		
		// let's back propagate all the hidden layer errors now
		for(int l = OUTPUT_LAYER-1; l > INPUT_LAYER; l--) {

			for(int i = 0; i < m_neurons[l].length; i++) {
				
				Neuron n = m_neurons[l][i];
				double dE = 0.0;
				
				for(int j = 0; j < m_neurons[l+1].length; j++) {
					dE += m_neurons[l+1][j].getError() * m_neurons[l+1][j].getWeight(i);
				}
				dE *= n.getOutput() * (1.0 - n.getOutput());
				n.setError(dE);
			}
		}

		// basically, as error goes down, the learnRate goes down too, helping tune more precisely
		double dynLearnRate = m_learnRate;// * totalE;
		
		// let's move forward now, adjusting the weights as we go
		for(int l = INPUT_LAYER+1; l <= OUTPUT_LAYER; l++) {
		
			for(int i = 0; i < m_neurons[l].length; i++) {
				
				Neuron n = m_neurons[l][i];
				
				for(int j = 0; j < n.getNumWeights(); j++) {
					double dE = n.getError() * m_neurons[l-1][j].getOutput();
					double wP = n.getWeight(j) - (dynLearnRate * dE);
					n.setWeight(j,  wP);
				}
				
				double dB = n.getBias() - (dynLearnRate * n.getError());
				n.setBias(dB);
			}
		}
				
		// back propagation is complete
	}
	 
	public CIFARNeuralNetwork(int[] layers, double learnRate) {

		// Show what we're building
		System.out.print("Building a " + (layers.length - 1) + " layer network: {");
		for(int i = 0; i < layers.length; i++) {
			System.out.print(layers[i]);
			if(i < layers.length - 1)
				System.out.print(", ");
			else
				System.out.println("}");
		}
		
		// for indexing, update our OUTPUT_LAYER
		OUTPUT_LAYER = layers.length - 1;
		
		// count total neurons in network
		int totalN = 0;
		for(int i = 0; i < layers.length; i++) {
			totalN += layers[i];
		}

		// let's get our set of layers initialized
		m_neurons = new Neuron[layers.length][];
		for(int i = 0; i < layers.length; i++) {
			
			// now lets initialize each member in eeach layer
			m_neurons[i] = new Neuron[layers[i]];
			for(int j = 0; j < layers[i]; j++) {
				m_neurons[i][j] = new Neuron();
			}
		}
		
		// now, allocate the weights in our Neurons
		for(int i = INPUT_LAYER+1; i <= OUTPUT_LAYER; i++) {

			for(int j = 0; j < m_neurons[i].length; j++) {
				m_neurons[i][j].setNumWeights(layers[i-1]);
				m_neurons[i][j].randomize((double)totalN);
			}
		}
		
		// save our learn rate - should add validation??
		m_learnRate = learnRate;		
	}

	// teach the network by sending it an image and label
	public int trainNetwork(byte[] inputPixels, int label) {

		// setup our target vector 
		double[] targetVector = vectorFromLabel(label);

		// setup the image in the input layer
		attachImage(inputPixels);
		
		// forward propagate these inputs throughout the network
		forwardPropagate();
		
		// capture what our guess was here
		int guess = labelFromVector(vectorFromOutput());
		
		// calculate the error and apply backwards to the weights and biases
		backPropagate(targetVector, false);
		
		// return our guess in case anyone cares
		return guess;
	}
	
	// test the network by sending an image and getting a response
	public int testNetwork(byte[] inputPixels) {

		// get the image setup in the input layer
		attachImage(inputPixels);
		
		// forward propagate this image
		forwardPropagate();
		
		// let's make our guess from our Output Neurons output values
		return(labelFromVector(vectorFromOutput()));
	}
	
	// dump the imagination of our neural network out to an image
	// reverse activate a network to generate an image of it's 'memory'
	public void reverseActivateImage(String file, int output) {

		// let's set the right values on the output neurons
		for(int i = 0; i < m_neurons[OUTPUT_LAYER].length; i++) {
			
			if(i == output) {
				m_neurons[OUTPUT_LAYER][i].setOutput(0.99);
			}
			else {
				m_neurons[OUTPUT_LAYER][i].setOutput(0.01);
			}
		}
		
		// okay, now let's back propagate this signal based on connections
		for(int l = OUTPUT_LAYER - 1; l >= INPUT_LAYER; l--) {

			for(int i = 0; i < m_neurons[l].length; i++) {
				
				double hO = 0.0;
				for(int j = 0; j < m_neurons[l+1].length; j++) {
					hO += m_neurons[l+1][j].getOutput() * m_neurons[l+1][j].getWeight(i);
				}

				m_neurons[l][i].setOutput(sigmoid(hO + m_neurons[l][i].getBias()));
			}
		}
		
		// Create a new buffered image
		BufferedImage bi = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
		for(int y = 0; y < 32; y++) {
			for(int x = 0; x < 32; x++) {
				int r = (int)(m_neurons[INPUT_LAYER][y*32+x].getOutput() * 255.0);
				int g = (int)(m_neurons[INPUT_LAYER][1024 + (y*32+x)].getOutput() * 255.0);
				int b = (int)(m_neurons[INPUT_LAYER][2048 + (y*32+x)].getOutput() * 255.0);
				
				Color col = new Color(r, g, b);
				bi.setRGB(x, y, col.getRGB());
			}
		}
		
		try {
			File outputFile = new File(file);
			ImageIO.write(bi, "png", outputFile);
		}
		catch(IOException e) {
		}
	}

}

