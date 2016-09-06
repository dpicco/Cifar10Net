/**
 * 
 */

/**
 * @author Dion
 * 
 */
public class Neuron {

	private double m_output = 0.0;
	private double m_bias = 0.0;
	private double m_error = 0.0;
	private double[] m_weights = null;
	
	/**
	 * 
	 */
	public Neuron() {
		// TODO Auto-generated constructor stub
	}

	public double getOutput() { 
		
		return m_output; 
	}
	
	public void setOutput(double val) {
		
		m_output = val;
	}
	
	public double getBias() {
		
		return m_bias;
	}
	
	public void setBias(double val) {
		
		m_bias = val;
	}

	public double getError() {
		
		return m_error;
	}
	
	public void setError(double err) {
		
		m_error = err;
	}
	
	public int getNumWeights() {
		
		if(m_weights != null) {
			return m_weights.length;
		}
		else {
			return 0;
		}
	}
	
	public void setNumWeights(int count) {
	
		m_weights = new double[count];
		for(int i = 0; i < count; i++) {
			m_weights[i] = 0.0;
		}
	}
	
	public double getWeight(int w) {

		if(m_weights == null)
			return 0.0;
		return m_weights[w];
	}
	
	public void setWeight(int w, double val) {
		
		if(m_weights == null)
			return;
		m_weights[w] = val;
	}
	
	public void randomize(double N) {
		
		// This is a method that generally works well
/*		m_bias = Math.random() / N;
		if(m_weights != null) {
			for(int i = 0; i < m_weights.length; i++) {
				m_weights[i] = (0.1 / N) + (Math.random() / N);
			}
		}
*/
		
		m_bias = (0.1 + (2 * Math.random())) * getRandomSign(); 
		if(m_weights != null) {
			for(int i = 0; i < m_weights.length; i++) {
				m_weights[i] = (0.1 + (2 * Math.random())) * getRandomSign();
			}
		}
	}
	
	public double getRandomSign() {
		
		double sign = Math.random() - 0.5;
		if(sign > -0.1 && sign < 0)
			sign -= 0.1;
		else if(sign < 0.1 && sign > 0)
			sign += 0.1;
		else if(sign == 0)
			sign -= (0.2 + Math.random());
		
		return sign / Math.abs(sign);
	}
}

