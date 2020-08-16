import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import structures.Post;
import analyzer.DocAnalyzer;



public class KNN {
    List<Post> corpus;
    Map<String, List<Post>> bucket;
    List<Map<String, Double>> random_vectors;
    Set<String> vocab;
    int k;
    int l;

    public KNN(int L, int K, List<Post> Corpus, Set<String> Vocab){
        vocab = Vocab;
        corpus = Corpus;
        l = L;
        k = K;

        random_vectors = new ArrayList<>();
        for(int i = 0; i < l; i++) {
            Map<String, Double> vec = new HashMap<>();
            for (String v : vocab) {
                vec.put(v, generateRandom());
            }
            random_vectors.add(vec);
        }
        System.out.println("finished generating random vectors...");

        constructBucket();

    }


    public void constructBucket(){
        bucket = new HashMap<>();
        for(Post review : corpus){
            HashMap<String, Double> vec = review.getVct();
            String proj = projection(vec);
            if(!bucket.containsKey(proj)) bucket.put(proj, new ArrayList<>());
            bucket.get(proj).add(review);
        }

        System.out.println("finished constructing bucket...");
        return;
    }

    public Double vectorMult(Map<String, Double> vec1, Map<String, Double> vec2){
        Double product = 0.0;
        for(String v : vocab){
            if(vec1.containsKey(v) && vec2.containsKey(v)){
                product += vec1.get(v) * vec2.get(v);
            }
        }
        return product;
    }

    public double generateRandom(){
        Random rand = new Random();
        double num = rand.nextDouble()* 2 - 1;
        return num;
    }

    public String projection(Map<String, Double> vec){
        String res = "";
        for(Map<String, Double> randomVec : random_vectors){
            res += vectorMult(vec, randomVec) >= 0 ? 1 : 0;
        }
        return res;
    }




    public List<Integer> predict(List<Post> X){
        List<Integer> prediction = new ArrayList<>();
        for(Post x : X){
            String proj = projection(x.getVct());
            List<Post> neighbors = getKNeighbors(x, bucket.get(proj));
            /*
            System.out.println("top3 for query : " + x.getID());
            System.out.println(x.getContent());
            System.out.println("");

            for(Post n : neighbors){
                System.out.println( " Author: " + n.getAuthor() +" Content: " + n.getContent() + " Date: "
                        + n.getDate());
            }
            */
            int count = 0;
            for(Post n : neighbors){
                if(n.getLabel() == 1 ) count++;
                else count --;
            }
            prediction.add(count >= 0 ? 1 : 0);
        }
        return prediction;
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

    public List<Integer> predictBruteForce(List<Post> X){
        List<Integer> prediction = new ArrayList<>();
        for(Post x : X){
            List<Post> neighbors = getKNeighbors(x);
            int count = 0;
            for(Post n : neighbors){
                if(n.getLabel() == 1 ) count++;
                else count --;
            }
            prediction.add(count >= 0 ? 1 : 0);
        }
        return prediction;
    }

    public List<Post> getKNeighbors(Post review, List<Post> neighbors){
        Map<Post, Double> map = new HashMap<>();
        for(Post neighbor : neighbors){
            map.put(neighbor, review.similiarity(neighbor));
        }
        PriorityQueue<Entry<Post, Double>> pq = new PriorityQueue<>(10,
                                (e1, e2) -> Double.compare(e2.getValue(), e1.getValue()));
        for(Entry e : map.entrySet()){
            pq.offer(e);
        }

        List<Post> res= new ArrayList<>();
        for(int i =0; i < Math.min(k, pq.size()); i++){
            Entry<Post, Double> curr = pq.poll();
            res.add(curr.getKey());
        }
        return res;
    }

    public List<Post> getKNeighbors(Post review){
        // bruth force KNN
        Map<Post, Double> map = new HashMap<>();
        for(Post neighbor : corpus){
            map.put(neighbor, review.similiarity(neighbor));
        }
        PriorityQueue<Entry<Post, Double>> pq = new PriorityQueue<>(10,
                (e1, e2) -> Double.compare(e2.getValue(), e1.getValue()));
        for(Entry e : map.entrySet()){
            pq.offer(e);
        }

        List<Post> res= new ArrayList<>();
        for(int i =0; i < Math.min(k, pq.size()); i++){
            Entry<Post, Double> curr = pq.poll();
            res.add(curr.getKey());
        }
        return res;
    }



    public static void main(String[] args){
        try {
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

            fis = new FileInputStream("corpus");
            ObjectInputStream ois = new ObjectInputStream(fis);
            List<Post> corpus = (List<Post>) ois.readObject();

            ois.close();
            fis.close();
            System.out.println("Finished reading corpus...");

            preprocessor pre = new preprocessor(corpus, controlledVocab);
            List<Post> processed_corpus = pre.getProcessedCorpus();
            KNN model = new KNN(5, 5, processed_corpus, controlledVocab);


            fis = new FileInputStream("query");
            ois = new ObjectInputStream(fis);
            List<Post> query = (List<Post>) ois.readObject();

            ois.close();
            fis.close();

            pre = new preprocessor(query, controlledVocab);
            List<Post> processed_query = pre.getProcessedCorpus();

            long start = System.nanoTime();
            model.predict(processed_query);
            long end = System.nanoTime();
            long duration = end - start;
            double seconds = (double) duration / 1_000_000_000.0;
            System.out.println("Running time of random projection is " + seconds + "s");

            start = System.nanoTime();
            model.predictBruteForce(processed_query);
            end = System.nanoTime();
            duration = end - start;
            seconds = (double) duration / 1_000_000_000.0;
            System.out.println("Running time of brute force is " + seconds + "s");



        }catch(IOException e){
            System.out.println();
        }catch(ClassNotFoundException e){
            e.printStackTrace();
        }
    }



}
