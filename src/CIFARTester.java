import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class CIFARTester {

	// the files used to train and test the network: CIFAR-10 sample data
	private String DATA_BATCH[] = { "cifar-10-batches-bin/data_batch_1.bin",
									"cifar-10-batches-bin/data_batch_2.bin",
									"cifar-10-batches-bin/data_batch_3.bin",
									"cifar-10-batches-bin/data_batch_4.bin",
									"cifar-10-batches-bin/data_batch_5.bin",
									"cifar-10-batches-bin/test_batch.bin" };
	private String LABEL_MAP[] = { "airplane", "automobile", "bird", "cat", "deer", 
								   "dog", "frog", "horse", "ship", "truck" };
	
	// the Network that we will be testing against
	CIFARNeuralNetwork m_net = null;


	// Default constructor
	public CIFARTester() { }

	// this prepares our network to start training and testing
	public void init(int[] layers, double learn) {
		
		m_net = new CIFARNeuralNetwork(layers, learn);
	}
	
	public void trainNetwork() {
	
		int sessionScore = 0;
		
		// unlike MNIST we got a bunch of batches to do here but lets stop short of testing batch
		for(int batch = 0; batch < DATA_BATCH.length - 1; batch++) {
			
			FileInputStream fis = null;
			int batchScore = 0;
			
			try {
				fis = new FileInputStream(DATA_BATCH[batch]);
				
				// pre-allocate some storage for our channel information here
				byte[] colorChannels = new byte[1024 * 3];
				byte[] grayScale = new byte[1024];
				
				// each CIFAR batch has 10,000 images 
				// in format <index byte><1024 byte red channel><1024 byte green channel><1024 byte blue channel><... repeat>
				for(int image = 0; image < 10000; image++) {

					// get the image data
					int imageIndex = fis.read();
					fis.read(colorChannels);
					
					// convert to gray scale
/*					for(int i = 0; i < 1024; i++) {
						int g = (int)((float)(colorChannels[i] & 0xff) * 0.299) + 
								(int)((float)(colorChannels[i+1024] & 0xff) * 0.587) + 
								(int)((float)(colorChannels[i+2048] & 0xff) * 0.114);
						
						grayScale[i] = (byte)g;
					}
*/					
					
					// let's attach to the inputs of our neural net and run a session
					int guess = m_net.trainNetwork(colorChannels, imageIndex);
					
					// keeping track of our metrics here is good for debugging
					if(guess == imageIndex) {
						sessionScore++;
						batchScore++;
					}
				}
				
				// let's debug the current learning status
				System.out.format("Data Batch %1$d Complete: Batch Score = %2$4.2f%%, Session Score = %3$4.2f%% %n", batch+1, (float)batchScore / 100.0, (float)sessionScore / (100.0 * (float)(batch + 1)));
				
			}
			catch(FileNotFoundException e) {

			}
			catch(IOException e) {
				
			}
			finally {
				if(fis != null) { 
					try {
						fis.close();
					}
					catch(IOException e) {
					}
				}
			}
		}
	}

	// the method to test our network
	// returns the number of correct guesses out of the 10,000 set used
	public int testNetwork() {

		int sessionScore = 0;
		FileInputStream fis = null;
			
		try {
			fis = new FileInputStream(DATA_BATCH[5]);
				
			// pre-allocate some storage for our channel information here
			byte[] colorChannels = new byte[1024 * 3];
				
			// each CIFAR batch has 10,000 images 
			// in format <index byte><1024 byte red channel><1024 byte green channel><1024 byte blue channel><... repeat>
			for(int image = 0; image < 10000; image++) {

				int imageIndex = fis.read();
				fis.read(colorChannels);
					
				// let's attach to the inputs of our neural net and run a session
				int guess = m_net.testNetwork(colorChannels);
					
				// keeping track of our metrics here is good for debugging
				if(guess == imageIndex) {
					sessionScore++;
				}
			}
		}
		catch(FileNotFoundException e) {

		}
		catch(IOException e) {
				
		}
		finally {
			if(fis != null) {
				try {
					fis.close();
				}
				catch(IOException e) {
				}
			}
		}
		return sessionScore;
	}	
	
	public void imagination() {
		
		String base = "imagination/image_";
		
		for(int i = 0; i < 10; i++) {
			String filename = base + LABEL_MAP[i] +".png";
			m_net.reverseActivateImage(filename, i);
		}		
	}
}
