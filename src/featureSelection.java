import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.ObjectOutputStream;
import java.util.*;

import analyzer.DocAnalyzer;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import structures.LanguageModel;
import structures.LanguageModel.tuple;
import structures.Post;
import structures.Token;
import java.io.IOException;
import java.util.Map.Entry;

public class featureSelection {
    public static double num_documents;
    public static double num_positive;
    public static double num_negative;
    public static Map<String, Integer> DF;
    public static List<Post> reviews;
    public static Set<String> vocab;

    public static Set<String> controlled_vocab = new HashSet<>();


    public static void setVariable(DocAnalyzer analyzer){

        // filter out vocab that has less than 10 DF
        reviews = analyzer.m_reviews;
        DF = analyzer.DF;

        num_documents = 0;
        num_positive = 0;
        num_negative = 0;
        for(Post review : reviews){
            num_documents++;
            if(review.getLabel() ==1 ){
                num_positive++;
            }else{
                num_negative++;
            }
        }
        vocab = new HashSet<>();

        for(String key : DF.keySet()){
            if(DF.get(key) >= 10) vocab.add(key);
        }
        return;

    }

    public static double log(double n){
        if (n == 0) return 0;
        else return Math.log(n);
    }

    public static Double infoGain(String token){
        // positive document with term t, negative document with term t

        double DF_t = DF.get(token);
        double positive_t = 0;
        double positive_without_t =0;
        for(Post review : reviews){
            Set<String> set = review.getVocabs();
            if(set.contains(token) && review.getLabel() == 1) positive_t++;
            if(!set.contains(token) && review.getLabel() == 1) positive_without_t++;
        }
        double num_documents_without_t = num_documents - DF_t;
        double negative_t = DF_t - positive_t;
        double negative_without_t = num_documents_without_t - positive_without_t;

        double ig = - (num_positive/num_documents * log(num_positive/num_documents) + num_negative/num_documents *
                    log(num_negative/num_documents));
        ig += DF_t/ num_documents * (positive_t/ DF_t * log(positive_t/ DF_t) + negative_t/DF_t * log(negative_t/DF_t));

        ig += num_documents_without_t/num_documents * (positive_without_t/num_documents_without_t * log(positive_without_t/num_documents_without_t)
                + negative_without_t/num_documents_without_t * log(negative_without_t/num_documents_without_t));
        return ig;
    }
    public static Double chiSquare(String token){
        //A is the number of positive documents containing term t, B is the number of positive documents not containing term t,
        // C is the number of negative documents containing term t, and D is the number of negative documents not containing term t.
        double DF_t = DF.get(token);
        double positive_t = 0;
        double positive_without_t =0;
        for(Post review : reviews){
            Set<String> set = review.getVocabs();
            if(set.contains(token) && review.getLabel() == 1) positive_t++;
            if(!set.contains(token) && review.getLabel() == 1) positive_without_t++;
        }
        double num_documents_without_t = num_documents - DF_t;
        double negative_t = DF_t - positive_t;
        double negative_without_t = num_documents_without_t - positive_without_t;
        double A = positive_t;
        double B = positive_without_t;
        double C = negative_t;
        double D = negative_without_t;

        double chi = (A + B + C + D) * Math.pow((A  * D - B * C), 2);
        chi = chi / ((A + C)* (B + D) * (A + B) * (C + D));

        return chi;
    }


    public static void calculateIG() throws IOException{
        FileWriter fw = new FileWriter("IG.txt");
        Map<String, Double> map = new HashMap<>();
        int count = 0;
        for(String t : vocab){
            map.put(t, infoGain(t));
            System.out.println(++count  +"/" + vocab.size());
        }
        PriorityQueue<Entry<String, Double>> pq = new PriorityQueue<>(10, (e1, e2) ->Double.compare(e2.getValue(), e1.getValue()));

        for(Entry e:map.entrySet()){
            pq.offer(e);
        }

        for(int i = 0; i < 5000; i ++){
            Entry<String, Double> curr = pq.poll();
            controlled_vocab.add(curr.getKey());
            fw.write(curr.getKey() + ", " + curr.getValue() + "\n");
        }
        fw.close();
        return;
    }

    public static void calculateChi() throws IOException{
        FileWriter fw = new FileWriter("Chi.txt");
        Map<String, Double> map = new HashMap<>();
        int count = 0;
        for(String t : vocab){
            double temp = chiSquare(t);
            if(temp >= 3.841) {
                map.put(t, chiSquare(t));
            }
            System.out.println(++count  +"/" + vocab.size());
        }
        PriorityQueue<Entry<String, Double>> pq = new PriorityQueue<>(10, (e1, e2) ->Double.compare(e2.getValue(), e1.getValue()));

        for(Entry e:map.entrySet()){
            pq.offer(e);
        }

        for(int i = 0; i < Math.min(5000, pq.size()); i ++){
            Entry<String, Double> curr = pq.poll();
            controlled_vocab.add(curr.getKey());
            fw.write(curr.getKey() + ", " + curr.getValue() + "\n");
        }
        fw.close();
        return;
    }

    public static void saveControlledVocab()throws IOException{
        System.out.println("The size of controlled vocab: " + controlled_vocab.size());
        FileWriter fw = new FileWriter("controlled_vocab.txt");

        for(String v : controlled_vocab){
            fw.write(v + "\n");
        }
        fw.close();
        return;
    }

    public static void filterReview() throws IOException{
        List<Post> corpus = new ArrayList<>();
        for(int i = 0; i < reviews.size(); i++){
            Post review = reviews.get(i);
            Set<String> set = review.getVocabs();
            int count = 0;
            for(String s : set) {
                if (controlled_vocab.contains(s)) count++;
                if(count > 5) {
                    corpus.add(review);
                    break;
                }
            }
        }

        System.out.println("Number of review in corpus: " + corpus.size());

        FileOutputStream fos = new FileOutputStream("corpus");
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(corpus);
        oos.close();
        fos.close();
        return;


    }




    public static void main(String[] args)throws IOException{
        DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 1);
        analyzer.LoadStopwords("/Users/yingqiaoxiong/Desktop/text mining/MP3/data/stop_word");
        analyzer.LoadDirectory("/Users/yingqiaoxiong/Desktop/text mining/MP3/data/yelp", ".json");
        analyzer.calculateDF();

        setVariable(analyzer);
        calculateIG();
        calculateChi();

        saveControlledVocab();

        filterReview();


    }


}


