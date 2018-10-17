/*
* REFERENCE:
* https://www.caveofprogramming.com/java/java-file-reading-and-writing-files-in-java.html
* http://thiele.nu/attachments/article/3/NearestNeighbour.java
* Author : clam7738, clia8758
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Pair implements Comparable<Pair> {
    private Double distance;
    private double classifier; // 1.0 = yes, 0.0 = no

    public Pair(double value, double classifier) {
        this.distance = value;
        this.classifier = classifier;
    }

    public double getClassifier(){
        return this.classifier;
    }

    // The list sorts regards to the distance value
    @Override
    public int compareTo(Pair o) {
        return this.distance.compareTo(o.distance);
    }
}

public class fold {

    private static void removeRange(List<Pair> list, int from, int to) {
        for (int i = to; i >= from; i--) {
            list.remove(i);
        }
    }

    // Calculate Eucledian distance between training and sample
    private static List<List<Pair>> distance(List<List<Double>> trainingData, List<List<Double>> testingData){
        List<List<Pair>> distance = new ArrayList<>();
        int classIndex = trainingData.get(0).size() - 1;

        // This loop (i) processes each row of testing data and store result in distance list
        for (int i = 0; i < testingData.size(); i++) {
            distance.add(new ArrayList<>());

            // This loop (j) goes through each row of the training data
            for (int j = 0; j < trainingData.size(); j++) {
                double dist = 0.0;

                // This loop (k) goes through every element on the row of
                // training data and testing data and calculate the distances
                // between them.
                for (int k = 0; k < classIndex; k++) {
                    double test = testingData.get(i).get(k);
                    double train = trainingData.get(j).get(k);
                    double t = test - train;
                    dist += t * t;
                }

                Pair pair = new Pair(Math.sqrt(dist), trainingData.get(j).get(classIndex));
                distance.get(i).add(pair);
            }

            // Sort the distances of each row in ascending order, after that get
            // rid of all distances greater than 5NN in the list to save memory.
            Collections.sort(distance.get(i));
            removeRange(distance.get(i), 5, distance.get(i).size()-1);
        }

        return distance;
    }

    // Classifier: compare k to the distance to find nearest neighbour and classify.
    private static List<String> classify(List<List<Pair>> distances, int k){
        List<String> result = new ArrayList<String>();

        for(int i = 0; i < distances.size();i++) {
            int yes = 0, no = 0;
            for(int j = 0; j < k; j++){
                if(distances.get(i).get(j).getClassifier() == 1.0){
                    yes++;
                }else if(distances.get(i).get(j).getClassifier() == 0.0){
                    no++;
                }
            }
            result.add(yes > no ? "yes" : "no");
        }

        return result;
    }

    // Method : k-nn classifier
    private static List<String> knnClassifier(int k, List<List<Double>> trainingData, List<List<Double>> testingData){
        List<List<Pair>> distance = new ArrayList<>(); // distance of the data item and specific sample
        List<String> result = new ArrayList<String>(); // return the result of the k nearest neighbour

        // calculate the "distance" from that data item to my specific sample.
        distance = distance(trainingData,testingData);

        // find the closest value and predict result
        result = classify(distance, k);

        return result;
    }

    private static List<Double> calculateMean(List<List<Double>> trainingData, double classifier) {

        List<Double> mean = new ArrayList<>();
        final int cIndex = trainingData.get(0).size()-1; // classifier index
        int counter = 0; // no. of rows

        // Initialise mean array to have values 0
        for (int i = cIndex; i != 0; i--)
            mean.add(0.0);

        // Sum up values of each row
        for (List<Double> rowData : trainingData) {
            if (rowData.get(cIndex) != classifier)
                continue;

            counter++;
            for (int i = 0; i < cIndex; i++) {
                mean.set(i, rowData.get(i) + mean.get(i));
            }
        }

        // Calculate the mean
        for (int i = 0; i < mean.size(); i++)
            mean.set(i, mean.get(i) / counter);

        return mean;
    }

    private static List<Double> calculateStd(List<List<Double>> trainingData, List<Double> mean, double classifier) {

        List<Double> std = new ArrayList<>();
        final int cIndex = trainingData.get(0).size()-1; // class index
        int n = 0; // number of data points

        for (int i = cIndex; i != 0; i--)
            std.add(0.0);

        for (List<Double> rowData : trainingData) {
            if (!rowData.get(cIndex).equals(classifier))
                continue;

            n++;
            for (int i = 0; i < cIndex; i++) {
                double s = rowData.get(i) - mean.get(i);
                s *= s;
                std.set(i, std.get(i) + s);
            }
        }

        for (int i = 0; i < std.size(); i++) {
            double s = std.get(i) / (n-1);
            s = Math.sqrt(s);

            std.set(i, s);
        }

        return std;
    }

    private static List<String> nbClassifier(List<List<Double>> trainingData, List<List<Double>> testingData, int yesCount){

        List<Double> meanYes = calculateMean(trainingData, 1.0);
        List<Double> meanNo = calculateMean(trainingData, 0.0);
        List<Double> stdYes = calculateStd(trainingData, meanYes, 1.0);
        List<Double> stdNo = calculateStd(trainingData, meanNo, 0.0);

        List<String> result = new ArrayList<>();

        for (List<Double> rowData : testingData) {
            double probYes = ((double) yesCount / trainingData.size());
            double probNo = 1 - probYes;

            for (int i = 0; i < rowData.size()-1; i++) {
                // calculate the probability of yes for each row of test data
                double value = (rowData.get(i) - meanYes.get(i)) / stdYes.get(i);
                value = Math.exp(value * value / -2);
                value = value / stdYes.get(i) / Math.sqrt(2 * Math.PI);
                probYes *= value;

                // calculate the probability of no for each row of test data
                value = (rowData.get(i) - meanNo.get(i)) / stdNo.get(i);
                value = Math.exp(value * value / -2);
                value = value / stdNo.get(i) / Math.sqrt(2 * Math.PI);
                probNo *= value;
            }

            result.add((probYes > probNo) ? "yes" : "no");
        }

        return result;
    }
    
    private static List<List<Double>> cloneList(List<List<Double>> list) {
        List<List<Double>> clone = new ArrayList<>(list.size());

        for (List<Double> rowList : list) {
            clone.add(rowList);
        }
        return clone;
    }

    // Generating the 10 fold for dataset
    private static void into_fold(List<List<Double>> dataset, int knn){
        List<List<Double>> no = new ArrayList<>();
        List<List<Double>> yes = new ArrayList<>();
        
        int classIndex = dataset.get(0).size() - 1;

        // adding all the no and yes in their seperate list
        for(int i = 0; i < dataset.size();i++){
            List<Double> cur = dataset.get(i);
            Double clas = cur.get(classIndex);
            if(clas == 0.0){
                no.add(cur);
            }else if(clas == 1.0){
                yes.add(cur);
            }
        }
        
        double accuracy = 0;
        int noInEachFold = Math.abs(no.size()/10); // number of no in each fold 500
        int yesInEachFold = yes.size() / 10 +1; // number of yes in each fold 268

        // add the yes final rst then add the no
        for(int i = 0; i < 10; i++){
            List<List<Double>> noCopy = cloneList(no);
            List<List<Double>> yesCopy = cloneList(yes);
            List<List<Double>> training = new ArrayList<>();
            List<List<Double>> testing = new ArrayList<>();
            
            if (i == 8) { // number of yes row is 26
                for (int j = 241; j >= 216; j--){
                    testing.add(yesCopy.remove(j));
                }
            } else if (i == 9) { // number of yes row is 26
                for (int j = 267; j >= 242; j--){
                    testing.add(yesCopy.remove(j));
                }
            } else { // number of yes row is 27
                for (int j = 0; j < yesInEachFold; j++){
                    testing.add(yesCopy.remove(i * yesInEachFold));
                }
            }
            
            for (int j = 0; j < noInEachFold; j++) {
                testing.add(noCopy.remove(i * noInEachFold));
            }

            List<String> result;
            training.addAll(yesCopy); // add the remaining rows from yesCopy to training set
            training.addAll(noCopy);
            
            if (knn == 0)
                result = nbClassifier(training, testing, yes.size());
            else
                result = knnClassifier(knn, training, testing);

            System.out.println("fold" + (i+1));

            for (int j = 0; j < testing.size(); j++) {
                for (int k = 0; k < testing.get(j).size()-1; k++) {
                    System.out.print(testing.get(j).get(k) + ",");
                }

                System.out.println(testing.get(j).get(classIndex) == 1.0 ? "yes" : "no");

                if (result.get(j) == "yes") {
                    accuracy += testing.get(j).get(classIndex).equals(1.0) ? 1 : 0;
                } else {
                    accuracy += testing.get(j).get(classIndex).equals(0.0) ? 1 : 0;
                }
            }
            System.out.println();
        }
        
        System.out.println("accuracy: " + accuracy / dataset.size());
    }

    public static void main(String[] args) {

        // checking arguments
        if (args.length < 2) {
            System.out.println("Not enough arguments");
            System.exit(1);
        }

        String training_file = args[0]; // file for the original dataset
        String method = args[1];

        List<List<Double>> trainingData = new ArrayList<>();

        // Read training file
        try {
            FileReader training = new FileReader(training_file);
            BufferedReader training_Reader = new BufferedReader(training);

            String line;
            int i = 0;

            while ((line = training_Reader.readLine()) != null) {
                String[] tokens = line.split(",");
                int classIndex = tokens.length - 1;
                trainingData.add(new ArrayList<>());

                for (int j = 0; j < classIndex; j++) {
                    trainingData.get(i).add(Double.parseDouble(tokens[j]));
                }
                trainingData.get(i).add(tokens[classIndex].equals("yes") ? 1.0 : 0.0);
                i++;
            }

            training_Reader.close();
        } catch (IOException ex) {
            System.out.println("Error reading file '" + training_file + "'");
        }

        if (method.toLowerCase().equals("nb")) {
            into_fold(trainingData, 0);
        } else {
            int k = Integer.parseInt("" + method.charAt(0));
            into_fold(trainingData, k);
        }

    }
}
