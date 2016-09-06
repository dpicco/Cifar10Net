
public class CifarNet {

	private static int epochs = 5;
	
	public CifarNet() { }

	
	// Entry point for our CifarNet work
	public static void main(String[] args) {

		CIFARTester test = new CIFARTester();
		
		// let's change some settings based on args
		
		// let's initialize the tester and prepare to go
		test.init(new int[]{32*32*3, 40, 30, 20, 10}, 0.05);
		
		// now let us train and test our net
		for(int e = 0; e < epochs; e++) {
			
			// let's provide some status to the user
			System.out.println("Starting Training Epoch " + (e+1));
			
			test.trainNetwork();
		}
		
		// training is complete, so now let's test to see how we're doing
		System.out.println("Starting Testing...");
		int score = test.testNetwork();
		
		// provide user feedback on how we did here
		System.out.println("Test Score: " + score + " out of 10,000 (" + ((double)score / 100.0) + "%)");
		
		// and let's put some images down that reflect what we believe about images
		test.imagination();
	}
}
