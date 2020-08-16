
package analyzer;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.HashSet;

import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.LanguageModel;
import structures.LanguageModel.tuple;
import structures.Post;
import structures.Token;

/**
 * @author Yingqiao Xiong
 */
public class DocAnalyzer {
	//N-gram to be created
	int m_N;
	
	//a list of stopwords
	HashSet<String> m_stopwords;
	
	//you can store the loaded reviews in this arraylist for further processing
	public ArrayList<Post> m_reviews;

	//you might need something like this to store the counting statistics for validating Zipf's and computing IDF
	HashMap<String, Token> m_stats;

	// total term frequency table
	Map<String, Integer> TTF;

	// document frequency table
	public Map<String, Integer> DF;

	Map<String, Double> IDF;

	//Unigram count
	Map<String, Integer> unigram_count = new HashMap<>();
	//Bigram count
	Map<String, Map<String, Integer>> bigram_count = new HashMap<>();

	// Vocabulary
	Set<String> vocab;
	//we have also provided a sample implementation of language model in src.structures.LanguageModel
	Tokenizer m_tokenizer;
	
	//this structure is for language modeling
	LanguageModel m_langModel;
	
	public DocAnalyzer(String tokenModel, int N) throws InvalidFormatException, FileNotFoundException, IOException {
		m_N = N;
		m_reviews = new ArrayList<Post>();
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
	}
	
	//sample code for loading a list of stopwords from file
	//you can manually modify the stopword file to include your newly selected words
	public void LoadStopwords(String filename) {
		try {
			m_stopwords = new HashSet<>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();

		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	public void analyzeDocument(JSONObject json) {		
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));

				//System.out.println(Normalization(review.getContent()));
				String[] tokens = Tokenize(review.getContent());
				//System.out.println(Arrays.toString(tokens));
				
				/**
				 * HINT: perform necessary text processing here based on the tokenization results
				 * e.g., tokens -> normalization -> stemming -> N-gram -> stopword removal -> to vector
				 * The Post class has defined a "HashMap<String, Token> m_vector" field to hold the vector representation 
				 * For efficiency purpose, you can accumulate a term's DF here as well
				 */
				String prev_token = null;
				List<String> list = new ArrayList<>();
				for(int j = 0; j < tokens.length; j++){
					String processed_token = SnowballStemming(Normalization(tokens[j]));

					// remove stopword
					if(!m_stopwords.contains(processed_token) && !processed_token.equals("")){
						list.add(processed_token);
					}

					if( !processed_token.equals("")){
						unigram_count.put(processed_token, unigram_count.getOrDefault(processed_token, 0) + 1);
					}



				}

				review.setTokens(list.toArray(new String[list.size()]));
				review.setVocab(new HashSet<String>(Arrays.asList(review.getTokens())));
				
				
				m_reviews.add(review);
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}



	public void getTop10(String prefix, LanguageModel lm){
		Set<String> words = lm.bigram_count.get(prefix).keySet();
		Map<String, Double> probs = new HashMap<>();
		PriorityQueue<Entry<String, Double>> pq = new PriorityQueue<Entry<String, Double>>(
				(e1, e2)-> Double.compare(e2.getValue(), e1.getValue()));
		for(String token : words){
			probs.put(token, lm.calcLinearSmoothedProb(prefix, token));
		}
		for(Entry e : probs.entrySet()){
			pq.offer(e);
		}
		System.out.println("Top10 for linear interpolation smoothing");
		for(int i = 0; i< 10; i++){
			Entry temp = pq.poll();
			System.out.println(temp.getKey() + " : " + temp.getValue());

		}
		System.out.println();
		probs = new HashMap<>();
		pq = new PriorityQueue<Entry<String, Double>>(
				(e1, e2)-> Double.compare(e2.getValue(), e1.getValue()));

		for(String token : words){
			probs.put(token, lm.calcAbsDiscountProb(prefix, token));
		}
		for(Entry e : probs.entrySet()){
			pq.offer(e);
		}
		System.out.println("Top10 for Absolute Discount smoothing");
		for(int i = 0; i< 10; i++){
			Entry temp = pq.poll();
			System.out.println(temp.getKey() + " : " + temp.getValue());

		}
		return;
	}

	public static void GenerateSentences(LanguageModel lm){
		if(lm.m_N == 1){
			System.out.println("10 sentences generated by unigram");
			for(int i = 0; i < 10; i++){
				String sentence = "";
				Double prob = 0.0;
				for(int j = 0; j < 15; j++){
					tuple t = lm.sampling();
					sentence += t.key + " ";
					prob += Math.log(t.value);
				}
				sentence.trim();
				System.out.println(sentence + " : " + prob);
			}
		}else{
			System.out.println("10 sentences generated by bigram with Linear Smoothing");
			for(int i = 0; i < 10; i ++){
				tuple t = lm.sampling();
				String prefix = t.key;
				Double prob = Math.log(t.value);

				String sentence = "" + prefix;
				for(int j = 0; j < 15; j++){
					tuple bi = lm.sampling(prefix, "Linear");
					String token = bi.key;
					prob += Math.log(bi.value);
					prefix = token;
					sentence += token + " ";
				}
				sentence.trim();
				System.out.println(sentence + " : " + prob);
			}

			System.out.println("10 sentences generated by bigram with Absolute Discount Smoothing");
			for(int i = 0; i < 10; i ++){
				tuple t = lm.sampling();
				String prefix = t.key;
				Double prob = Math.log(t.value);

				String sentence = "" + prefix;
				for(int j = 0; j < 15; j++){
					tuple bi = lm.sampling(prefix, "Abs");
					String token = bi.key;
					prob += Math.log(bi.value);
					prefix = token;
					sentence += token + " ";
				}
				sentence.trim();
				System.out.println(sentence + " : " + prob);
			}
		}
	}
	




	public void calculateTTF(){
		TTF = new HashMap<>();
		System.out.println("Calculating total term frequency...");
		for(Post review : m_reviews){
			String[] tokens = review.getTokens();
			for(String token : tokens){
				TTF.put(token, TTF.getOrDefault(token,0) + 1);
			}
		}
		System.out.println("vocabulary size is " + TTF.keySet().size());

	}

	public void calculateDF(){
		DF = new HashMap<>();
		System.out.println("Calculating document frequency...");
		for(Post review : m_reviews){
			Set<String> set = new HashSet<String>(Arrays.asList(review.getTokens()));
			for(String token : set){
				DF.put(token, DF.getOrDefault(token, 0) + 1);
			}

		}
		System.out.println("vocabulary size is " + DF.keySet().size());

	}

	public void calculateIDF() throws IOException{
		IDF = new HashMap<>();
		int total_document_size = m_reviews.size();

		for(String key : DF.keySet()){
			IDF.put(key, Math.log((double)total_document_size/(double) DF.get(key)));
		}

		FileWriter fw = new FileWriter("IDF.txt");

		for(String key : IDF.keySet()){
			fw.write(key + " " + DF.get(key) + " " + IDF.get(key) + "\n");
		}
	}


	public void writeTTF()throws IOException{
		FileWriter fw = new FileWriter("TTF.txt");
		System.out.println("Writing total term frequency...");
		// implement a heap sort
		PriorityQueue<Entry<String, Integer>> pq = new PriorityQueue<>(10, (e1, e2) -> e2.getValue() - e1.getValue());

		for(Entry e:TTF.entrySet()){
			pq.offer(e);
		}
		while(!pq.isEmpty()){
			Entry curr = pq.poll();
			fw.write(curr.getKey() + " " + curr.getValue() + "\n");

		}
		fw.close();
	}

	public void writeDF() throws IOException{
		FileWriter fw = new FileWriter("DF.txt");
		PriorityQueue<Entry<String, Integer>> pq = new PriorityQueue<>(10, (e1, e2) -> e2.getValue() - e1.getValue());

		for(Entry e: DF.entrySet()){
			pq.offer(e);
		}
		while(!pq.isEmpty()){
			Entry curr = pq.poll();
			fw.write(curr.getKey() + " " + curr.getValue() + " " + IDF.get(curr.getKey()) + "\n");
			//fw.write(curr.getKey() + " " + curr.getValue() + " " + "\n");

		}
		fw.close();
		/*
		// filter out token with DF less than 50
		FileWriter fw2 = new FileWriter("Less_than_50.txt");
		for(String key : DF.keySet()){
			if(DF.get(key) <= 50){
				fw2.write(key + "\n");
			}
		}
		fw2.close();
		*/
	}

	public void writeIDF() throws IOException{
		FileWriter fw = new FileWriter("IDF.txt");

		for(String key : IDF.keySet()){
			fw.write(key + " " + IDF.get(key) +  "\n");

		}
		fw.close();
	}

	public void loadIDF(String filename) throws IOException{
		try{
			IDF = new HashMap<>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
			String line;

			while((line = reader.readLine()) != null){
				if(!line.isEmpty()){
					String[] arr = line.split(" ");
					IDF.put(arr[0], Double.parseDouble(arr[1]));
				}
			}
			reader.close();
			System.out.println("loaded train model");

		}catch(IOException e){
			System.out.println("Failed loading model");
		}
	}


	public void getVector(Post review){
		// term frequency in a document
		Map<String, Double> TF = new HashMap<>();
		for(String token : review.getTokens()){
			if(IDF.containsKey(token)){
				TF.put(token, TF.getOrDefault(token, 0.0) + 1);
			}
		}

		// sub-linear normalization
		for(String key : TF.keySet()){
			TF.put(key, 1 + Math.log(TF.get(key)));
		}

		// calculate tf-idf weight
		HashMap<String, Double> vec = new HashMap<>();
		for(String key : TF.keySet()){
			if(IDF.containsKey(key)) {
				vec.put(key, TF.get(key) * IDF.get(key));
			}
		}
		review.setVct(vec);


	}

	public void computeCOSsimilarity() throws IOException{
		loadIDF("/Users/yingqiaoxiong/Downloads/MP1/IDF.txt");
		m_reviews = new ArrayList<>();
		LoadDirectory("/Users/yingqiaoxiong/Downloads/MP1/data/query", ".json");
		List<Post> query = m_reviews;

		for(Post review : query) getVector(review);

		m_reviews = new ArrayList<>();
		LoadDirectory("/Users/yingqiaoxiong/Downloads/MP1/data/yelp/test", ".json");
		List<Post> test = m_reviews;

		for(Post review : test) {
			getVector(review);

		}

		// compute cosine similarity and get the top 3 for each query
		for(Post q_review : query){
			Map<Post, Double> sim = new HashMap<>();
			for(Post t_review : test) {
				sim.put(t_review , q_review.similiarity(t_review));
			}

			PriorityQueue<Entry<Post, Double>> pq = new PriorityQueue<Entry<Post, Double>>(
					10, (e1, e2) -> Double.compare(e2.getValue(), e1.getValue()));

			for(Entry e : sim.entrySet()){
				pq.offer(e);
			}
			System.out.println("top3 for query : " + q_review.getID());
			System.out.println(q_review.getContent());
			System.out.println("");
			for(int i = 0; i < 3; i++){
				Entry<Post, Double> temp = pq.poll();
				Post curr = temp.getKey();
				System.out.println("Similarity: " + temp.getValue() + " Author: " + curr.getAuthor() +" Content: " + curr.getContent() + " Date: "
				+ curr.getDate());
			}
			System.out.println("\n");
		}

	}



	
	//sample code for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			
			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	// sample code for demonstrating how to recursively load files in a directory 
	public void LoadDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocument(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review documents from " + folder);
	}

	//sample code for demonstrating how to use Snowball stemmer
	public String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to use Porter stemmer
	public String PorterStemming(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to perform text normalization
	//you should implement your own normalization procedure here
	public String Normalization(String token) {
		// remove all non-word characters
		// please change this to removing all English punctuation
		token = token.replaceAll("\\p{Punct}", "");
		
		// convert to lower case
		token = token.toLowerCase(); 
		
		// add a line to recognize integers and doubles via regular expression
		// and convert the recognized integers and doubles to a special symbol "NUM"
		token = token.replaceAll("^[0-9]+([,.][0-9]?)?$", "NUM");
		
		return token;
	}
	
	String[] Tokenize(String text) {
		return m_tokenizer.tokenize(text);
	}
	
	public void TokenizerDemon(String text) {
		System.out.format("Token\tNormalization\tSnonball Stemmer\tPorter Stemmer\n");
		for(String token:m_tokenizer.tokenize(text)){
			System.out.format("%s\t%s\t%s\t%s\n", token, Normalization(token), SnowballStemming(token), PorterStemming(token));
		}		
	}
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {		
		DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 2);
		
		//code for demonstrating tokenization and stemming
		analyzer.TokenizerDemon("I've practiced for 30 years in pediatrics, and I've never seen anything quite like this.");

		analyzer.LoadStopwords("/Users/yingqiaoxiong/Desktop/text mining/MP3/data/stop_word");
		//entry point to deal with a collection of documents

		// load both train and test
		//analyzer.LoadDirectory("/Users/yingqiaoxiong/Downloads/MP1/data/yelp", ".json");



		// load only train

		analyzer.LoadDirectory("/Users/yingqiaoxiong/Desktop/text mining/MP3/data/yelp", ".json");


		/*
		analyzer.calculateTTF();

		analyzer.writeTTF();

		analyzer.calculateDF();

		analyzer.calculateIDF();

		analyzer.writeDF();

		analyzer.writeIDF();
`
		analyzer.computeCOSsimilarity();
		*/

		Map<String, Integer> unigram_count = analyzer.unigram_count;
		Map<String, Map<String, Integer>> bigram_count = analyzer.bigram_count;
		LanguageModel unigramLM = new LanguageModel(1, unigram_count, bigram_count );
		LanguageModel bigramLM = new LanguageModel(2, unigram_count, bigram_count, unigramLM);

		analyzer.getTop10("good", bigramLM);

		//analyzer.GenerateSentences(unigramLM);
		analyzer.GenerateSentences(bigramLM);



	}

}
