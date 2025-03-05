package org.example;
import org.slf4j.Logger;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import java.io.File;
import java.util.List;
import java.util.ArrayList;

public class Word2VecExample {
    public static void main(String[] args) {
        String home = System.getProperty("user.home");
        String modelFolderPath = home + File.separator + "Desktop" + File.separator + "ai_models";
        String pathToModel = modelFolderPath + File.separator + "GoogleNews-vectors-negative300.bin";

        WordVectors model = WordVectorSerializer.readWord2VecModel(new File(pathToModel));

        System.out.println(model.similarity("car", "bus"));
        System.out.println(model.similarity("yellow", "bus"));

        List<String> similarWords = new ArrayList<>(model.wordsNearest("worst", 10));
        System.out.println(similarWords);
    }
}
