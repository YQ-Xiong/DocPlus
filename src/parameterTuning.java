import structures.Post;
import java.util.*;
import java.io.*;

public class parameterTuning {

    public static void KNN_kfold(List<Post> corpus, Set<String> vocab, int l, int k){

        int foldSize = corpus.size() / 10;

        double KNNprecision_total = 0.0;
        double KNNrecall_total = 0.0;
        double KNNF1_total = 0.0;


        for(int i = 0; i < 10; i++){
            List<Post> test = corpus.subList(i * foldSize, Math.min((i+1) * foldSize, corpus.size()));
            List<Post> train = new ArrayList<>();
            train.addAll(corpus.subList(0, i * foldSize));
            train.addAll(corpus.subList(Math.min((i+1) * foldSize, corpus.size()), corpus.size()));
            List<Integer> truth = new ArrayList<>();
            for(Post review : test){
                truth.add(review.getLabel());
            }

            System.out.println("in " + (i+1) + " test" );


            KNN knn = new KNN(l, k, train, vocab);
            double[] KNNmetrics = knn.metrics(knn.predict(test), truth);
            double KNNprecision = KNNmetrics[1];
            double KNNrecall = KNNmetrics[2];
            double KNNF1 = 2 *(KNNprecision * KNNrecall)/ (KNNprecision + KNNrecall);
            System.out.println("KNN " + "precision: " + KNNprecision + " recall: " + KNNrecall + " F1: " + KNNF1);

            KNNprecision_total += KNNprecision;
            KNNrecall_total += KNNrecall;
            KNNF1_total += KNNF1;

            // F1  = 2 * (precision * recall) / (precision + recall);

        }


        double average_KNNprecision = KNNprecision_total /10;
        double average_KNNrecall = KNNrecall_total / 10;
        double average_KNNF1 = KNNF1_total / 10 ;
        System.out.println("KNN l = " + l + " k = " + k);
        System.out.println("average precision: " + average_KNNprecision + " average recall: " + average_KNNrecall +
                " average F1: " + average_KNNF1);

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
            fis.close();
            ois.close();
            System.out.println("finished reading train set...");

            preprocessor pre = new preprocessor(corpus, controlledVocab);
            List<Post> processed_corpus = pre.getProcessedCorpus();

            int[] ls = new int[]{3,10,20};
            int[] ks = new int[]{3,10,15};

            for(int l : ls){
                for(int k : ks){
                    KNN_kfold(processed_corpus, controlledVocab,l, k);
                }
            }







        }catch(IOException e){
            System.out.println("Error initializing stream");
        }catch(ClassNotFoundException e){
            e.printStackTrace();
        }
    }
}
