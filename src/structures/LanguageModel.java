/**
 * 
 */
package structures;

import java.util.HashMap;
import java.util.Map.Entry;
import java.util.*;
/**
 * @author Yingqiao Xiong
 * Suggested structure for constructing N-gram language model
 */
public class LanguageModel {

	public int m_N; // N-gram
	int m_V; // the vocabulary size
	public Map<String, Integer> unigram_count; // sparse structure for storing the maximum likelihood estimation of LM with the seen N-grams
	public Map<String, Map<String, Integer>> bigram_count;
	Map<String, HashMap<String, Double>> memo;
	LanguageModel m_reference; // pointer to the reference language model for smoothing purpose
	
	double m_lambda = 0.9; // parameter for linear interpolation smoothing
	double m_delta = 0.1; // parameter for absolute discount smoothing
	
	public LanguageModel(int N, Map<String, Integer> unigram, Map<String, Map<String, Integer>> bigram) {
		m_N = N;
		unigram_count = unigram;
		bigram_count = bigram;

	}

	public LanguageModel(int N, Map<String, Integer> unigram, Map<String, Map<String, Integer>> bigram, LanguageModel ref) {
		m_N = N;
		unigram_count = unigram;
		bigram_count = bigram;
		m_reference = ref;

	}
	
	public double calcMLProb(String ...tokens) {
		// used for calcualting bi-gram maximum likelihood
		// return m_model.get(token).getValue(); // should be something like this
		String token1 = tokens[0];
		String token2 = tokens[1];
		if(bigram_count.containsKey(token1) && bigram_count.get(token1).containsKey(token2)) {

			Map<String, Integer> token1_counts = bigram_count.get(token1);
			int token2_count = token1_counts.get(token2);

			return (double) token2_count / (double) sum(token1_counts.values());
		}else{
			return 0.0;
		}
	}

	public double calcAdditiveSmoothedProb(String token) {
		double uni_sum = sum(unigram_count.values()) + m_delta * unigram_count.size();

		if(!unigram_count.containsKey(token)) return 0.0;
		double prob = (unigram_count.get(token) + m_delta) / uni_sum;
		return prob; // please use additive smoothing to smooth a unigram language model

	}

	public double calcLinearSmoothedProb(String ...tokens){
		if(m_N == 1 || tokens.length == 1) return calcAdditiveSmoothedProb(tokens[0]);
		else {
			return (1.0 - m_lambda) * calcMLProb(tokens[0], tokens[1]) + m_lambda * m_reference.calcLinearSmoothedProb(tokens[1]);
		}

	}

	public double calcAbsDiscountProb(String ...tokens){
		if(m_N == 1 || tokens.length == 1) return calcAdditiveSmoothedProb(tokens[0]);
		else {
			if (bigram_count.containsKey(tokens[0])) {
				Map<String, Integer> token1_counts = bigram_count.get(tokens[0]);
				int count = token1_counts.get(tokens[1]);
				int S = token1_counts.size();
				int sum = sum(token1_counts.values());
				return Math.max(count - m_delta, 0.0) / sum + (m_delta * S / sum) * m_reference.calcAbsDiscountProb(tokens[1]);
			} else {
				return 0.0;
			}
		}

	}

	private int sum(Collection<Integer> list){
		int sum = 0;
		for(int i : list){
			sum += i;
		}
		return sum;
	}

	public class tuple{
		public String key;
		public Double value;
		public tuple(String k, Double v){
			key = k;
			value = v;
		}
	}
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public tuple sampling(String ...arguments) {
		if(m_N == 1 || arguments.length == 0) {
			double prob = Math.random(); // prepare to perform uniform sampling
			for (String token : unigram_count.keySet()) {
				double temp = calcLinearSmoothedProb(token);
				prob -= temp;
				if (prob <= 0) {
					return new tuple(token, temp);
				}
			}
			return null; //How to deal with this special case?
		}else{
			String prefix = arguments[0];
			String smoothing = arguments[1];
			double prob = Math.random();
			Map<String, Integer> count = bigram_count.get(prefix);
			for(String token : count.keySet()){
				double temp = smoothing.equals("Linear") ? calcLinearSmoothedProb(prefix, token) : calcAbsDiscountProb(prefix, token);
				prob -= temp;
				if(prob <= 0)
					return new tuple(token, temp);
			}
			return null;
		}
	}
	
	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public double logLikelihood(Post review) {
		double likelihood = 0;
		for(String token:review.getTokens()) {
			//likelihood += Math.log(calcLinearSmoothedProb(token));
		}
		return likelihood;
	}
}
