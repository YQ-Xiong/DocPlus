import java.io.*;
import java.util.*;

import structures.Post;

public class preprocessor {

    List<Post>  corpus;
    Set<String> vocab;
    Map<String, Integer> DF;
    Map<String, Double> IDF;
    public preprocessor(List<Post> Corpus, Set<String> Vocab){
        corpus = Corpus;
        vocab = Vocab;

        calculateDF();
        calculateIDF();
        constructVectors();
    }

    public List<Post> getProcessedCorpus(){
        return corpus;
    }

    public void calculateDF(){
        DF = new HashMap<>();
        for(Post review : corpus){
            for(String v : review.getVocabs()){
                if(vocab.contains(v)) {
                    DF.put(v, DF.getOrDefault(v, 0) + 1);
                }
            }
        }
        System.out.println("DF contains: " + DF.size() + " keys");
    }

    public void calculateIDF() {
        IDF = new HashMap<>();
        int total_document_size = corpus.size();

        for(String key : DF.keySet()){
            IDF.put(key, Math.log((double)total_document_size/(double) DF.get(key)));
        }

    }

    public void constructVectors(){
        for(Post review : corpus){
            getVector(review);
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



        }catch(IOException e){
            System.out.println();
        }catch(ClassNotFoundException e){
            e.printStackTrace();
        }
    }




}
