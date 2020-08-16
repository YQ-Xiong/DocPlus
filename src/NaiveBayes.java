import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import structures.Post;
import analyzer.DocAnalyzer;

public class NaiveBayes {
    List<Post> corpus;
    Set<String> vocab;
    double delta;

    LanguageModel positive;
    LanguageModel negative;
    public NaiveBayes (List<Post> Corpus, Set<String> Vocab, double Delta){
        corpus = Corpus;
        vocab = Vocab;
        delta = Delta;
        Map<String, Integer> positive_count = new HashMap<>();
        Map<String, Integer> negative_count = new HashMap<>();
        int num_positive = 0;
        int num_negative = 0;
        for(Post review : Corpus){
            if(review.getLabel() == 1) num_positive++;
            else num_negative++;
            for(String token : review.getTokens()){
                if(review.getLabel() == 1) positive_count.put(token, positive_count.getOrDefault(token, 0) + 1);
                else negative_count.put(token, negative_count.getOrDefault(token, 0) + 1);
            }
        }

        positive = new LanguageModel(positive_count, num_positive, vocab, delta);
        negative = new LanguageModel(negative_count, num_negative, vocab, delta);
        System.out.println("Finished training Naive Bayes...");

    }

    public void calculateLogRatio() throws IOException{
        System.out.println("Calculating log ratios..");
        Map<String, Double> map = new HashMap<>();
        for(String v : vocab){
            double log = Math.log(positive.calculateAdditiveSmoothProb(v)/negative.calculateAdditiveSmoothProb(v));
            map.put(v, log);
        }

        PriorityQueue<Entry<String,Double>> pq = new PriorityQueue<>(10, (e1,e2) -> Double.compare(e2.getValue(), e1.getValue()));

        for(Entry e : map.entrySet()){
            pq.offer(e);
        }

        FileWriter fw = new FileWriter("log_ratio.txt");

        while(!pq.isEmpty()){
            Entry<String, Double> curr = pq.poll();
            fw.write(curr.getKey() + ", " + curr.getValue() + "\n");
        }
        fw.close();
        return;
    }


    public double f(Post review){
        int num_document = corpus.size();
        double prob_positive = (double)positive.total/num_document;
        double prob_negative = (double)negative.total/num_document;
        double score = Math.log(prob_positive/prob_negative);

        for(String token : review.getTokens()){
            score += Math.log(positive.calculateAdditiveSmoothProb(token)) - Math.log(negative.calculateAdditiveSmoothProb(token));
        }
        return score;
    }

    public int predict(Post review, double threshold){
        return f(review) >= threshold? 1 : 0;
    }

    public List<Integer> predict(List<Post> reviews){
        List<Integer> res = new ArrayList<>();
        for(Post review : reviews){
            res.add(predict(review, 0.0));
        }
        return res;
    }

    public double[] metrics(List<Integer> predictions, List<Integer> truth){
        assert predictions.size() == truth.size();
        double[] res = new double[3]; // 0: accuracy, 1: precision, 2: recall
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int total = predictions.size();
        for(int i= 0; i < predictions.size(); i++){
            if(predictions.get(i) == truth.get(i)) tp ++;
            else if(predictions.get(i) == 1 && truth.get(i) == 0) fp++;
            else if(predictions.get(i) == 0 && truth.get(i) == 1) fn++;
        }
        res[0] = (double)tp/ total;
        res[1] = (double) tp/ (tp + fp);
        res[2] = (double) tp/(tp + fn);
        return res;
    }

    public class LanguageModel{
        double delta;
        Map<String, Integer> count;
        Set<String> vocab;
        int total;

        public LanguageModel(Map<String, Integer> Count, int Total, Set<String> Vocab, double Delta){
            // Total: record total number of positive/negative documents in the corpus
            count =Count;
            vocab = Vocab;
            delta = Delta;
            total = Total;
        }

        public double calculateAdditiveSmoothProb(String token){
            double divisor = total + delta * vocab.size();

            double dividend = count.containsKey(token)? count.get(token) + delta : 0 + delta;
            return dividend/ divisor;
        }
    }



    public static void main(String[] args){
        try{

            FileInputStream fis = new FileInputStream("controlled_vocab.txt");
            InputStreamReader is = new InputStreamReader(fis);
            BufferedReader reader = new BufferedReader(is);
            String line;

            Set<String> controlledVocab = new HashSet<>();
            while((line = reader.readLine()) != null){
                controlledVocab.add(line);
            }
            reader.close();
            is.close();
            fis.close();

            System.out.println("finished loading vocab");

            System.out.println("size of controlled vocab is " + controlledVocab.size());

            /*
             fis = new FileInputStream("corpus");
            ObjectInputStream ois = new ObjectInputStream(fis);
            List<Post> corpus = (List<Post>)ois.readObject();

            System.out.println("Finished reading corpus...");

            ois.close();
            NaiveBayes model = new NaiveBayes(corpus, controlledVocab, 0.1);
            model.calculateLogRatio();
            */


            // load train and test dataset
            fis = new FileInputStream("train");
            ObjectInputStream ois = new ObjectInputStream(fis);
            List<Post> train = (List<Post>) ois.readObject();
            fis.close();
            ois.close();
            System.out.println("finished reading train set...");

            fis = new FileInputStream("test");
            ois = new ObjectInputStream(fis);
            List<Post> test = (List<Post>) ois.readObject();
            fis.close();
            ois.close();
            System.out.println("finished reading test set...");


            NaiveBayes model = new NaiveBayes(train, controlledVocab, 10);

            PriorityQueue<Double> pq = new PriorityQueue<>(10, (d1,d2) -> Double.compare(d2, d1));
            for(Post review : test){
                pq.offer(model.f(review));
            }
            int count = 0;

            System.out.println(pq.size());

            List<Double> f_score = new ArrayList<>();
            List<Integer> truth = new ArrayList<>();
            for(Post review : test){
                f_score.add(model.f(review));
                truth.add(review.getLabel());
            }

            List<Double> precision = new ArrayList<>();
            List<Double> recall = new ArrayList<>();
            while(!pq.isEmpty()){
                System.out.println("times: " + ++count);
                double threshold = pq.poll();
                List<Integer> predictions  = new ArrayList<>();
                for(Double f : f_score){
                    predictions.add(f >= threshold ? 1 : 0);
                }
                double[] stats = model.metrics(predictions, truth);
                precision.add(stats[1]);
                recall.add(stats[2]);
                System.out.println(Arrays.toString(stats));
            }
            FileWriter fw = new FileWriter("delta10.txt");
            for(int i = 0; i < precision.size(); i++){
                fw.write(precision.get(i) + "," + recall.get(i) + "\n");
            }
            fw.close();



        }catch(FileNotFoundException e){
            System.out.println("File not found");
        }catch(IOException e){
            System.out.println("Error initializing stream");
        }catch(ClassNotFoundException e){
            e.printStackTrace();
        }

    }


}
