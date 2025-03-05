package org.example;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class Word2VecExample {
    public static void main(String[] args) {
        WordVectors model = WordVectorSerializer.readWord2VecModel(
            new File("GoogleNews-vectors-negative300.bin")
        );

        System.out.println(model.similarity("car", "bus"));
        System.out.println(model.similarity("yellow", "bus"));

        List<String> similarWords = new ArrayList<>(model.wordsNearest("worst", 10));
        System.out.println(similarWords);
    }
}
