import structures.Post;
import java.util.*;
import java.io.*;


public class CrossValidation {



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

            int foldSize = processed_corpus.size() / 10;


            int k = 10;

            double NBprecision_total = 0.0;
            double NBrecall_total = 0.0;
            double NBF1_total = 0.0;
            double KNNprecision_total = 0.0;
            double KNNrecall_total = 0.0;
            double KNNF1_total = 0.0;


            for(int i = 0; i < k; i++){
                List<Post> test = processed_corpus.subList(i * foldSize, Math.min((i+1) * foldSize, processed_corpus.size()));
                List<Post> train = new ArrayList<>();
                train.addAll(processed_corpus.subList(0, i * foldSize));
                train.addAll(processed_corpus.subList(Math.min((i+1) * foldSize, processed_corpus.size()), processed_corpus.size()));

                NaiveBayes nb = new NaiveBayes(train, controlledVocab, 0.1);
                List<Integer> truth = new ArrayList<>();
                for(Post review : test){
                    truth.add(review.getLabel());
                }

                System.out.println("in " + (i+1) + " test" );
                double[] NBmetrics = nb.metrics(nb.predict(test), truth);
                double NBprecision = NBmetrics[1];
                double NBrecall = NBmetrics[2];
                double NBF1 = 2 * (NBprecision * NBrecall)/ (NBprecision + NBrecall);
                System.out.println("Naive Bayes " + "precision: " + NBprecision + " recall: " + NBrecall + " F1: " + NBF1);

                NBprecision_total += NBprecision;
                NBrecall_total += NBrecall;
                NBF1_total += NBF1;

                KNN knn = new KNN(5, 5, train, controlledVocab);
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

            double average_NBprecision = NBprecision_total / k;
            double average_NBrecall = NBrecall_total / k;
            double average_NBF1 = NBF1_total / k;
            System.out.println("For Naive Bayes, average precision: " + average_NBprecision + " average recall: " + average_NBrecall +
            " average F1: " + average_NBF1);


            double average_KNNprecision = KNNprecision_total /k;
            double average_KNNrecall = KNNrecall_total / k;
            double average_KNNF1 = KNNF1_total / k ;
            System.out.println("For KNN, average precision: " + average_KNNprecision + " average recall: " + average_KNNrecall +
                    " average F1: " + average_KNNF1);






        }catch(IOException e){
            System.out.println("Error initializing stream");
        }catch(ClassNotFoundException e){
            e.printStackTrace();
        }

    }

}
