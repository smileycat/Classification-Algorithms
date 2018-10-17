/*
* REFERENCE:
* https://www.caveofprogramming.com/java/java-file-reading-and-writing-files-in-java.html
* http://thiele.nu/attachments/article/3/NearestNeighbour.java
* Author : clam7738, clia8758
*/

import java.util.*;
import java.io.*;

class Pair implements Comparable<Pair> {
    Double distance;
    double classifier; // 1.0 = yes, 0.0 = no

    public Pair(double value, double classifier,int index) {
        this.distance = value;
        this.classifier = classifier;
    }

    public Double getdistance(){
        return this.distance;
    }

    public double getClassifier(){
        return this.classifier;
    }
    public double setClassifier(double i){
        this.classifier = i;
        return classifier;
    }

    // The list sorts regards to the distance value
    @Override
    public int compareTo(Pair o) {
        return this.distance.compareTo(o.distance);
    }
}

public class MyClassifier {

    private static void removeRange(List<Pair> list, int from, int to) {
        for (int i = to; i >= from; i--) {
            list.remove(i);
        }
    }

    // Calculate Eucledian distance between training and sample
    private static List<List<Pair>> distance(List<List<Double>> trainingData, List<List<Double>> testingData){
        List<List<Pair>> distance = new ArrayList<>();

        // This loop (i) processes each row of testing data and store result in distance list
        for (int i = 0; i < testingData.size(); i++) {
            distance.add(new ArrayList<>());

            // This loop (j) goes through each row of the training data
            for (int j = 0; j < trainingData.size(); j++) {
                double dist = 0.0;

                // This loop (k) goes through every element on the row of
                // training data and testing data and calculate the distances
                // between them.
                for (int k = 0; k < trainingData.get(j).size()-1; k++) {
                    double test = testingData.get(i).get(k);
                    double train = trainingData.get(j).get(k);
                    double t = test - train;
                    dist += t * t;
                }

                Pair pair = new Pair(Math.sqrt(dist), trainingData.get(j).get(8),j);
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
        
        for (List<Pair> rowData : distances) {
	    	int yes = 0, no = 0;
	    	for (int i = 0; i < k; i++) {
	    		if (rowData.get(i).getClassifier() == 1.0)
	    			yes++;
     		   	else
     		   		no++;
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
		final int cIndex = trainingData.get(0).size()-1; // classifier index
		int n = 0; // number of rows (data points)
		
        // Initialise std array to have values 0
		for (int i = cIndex; i != 0; i--)
			std.add(0.0);
		
        // Calculate the standard deviations
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
		
        // Continue calculate the standard deviations
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

			for (int i = 0; i < rowData.size(); i++) {
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

	public static void main(String[] args){

		// checking arguments
		if(args.length < 3){
			System.out.println("Not enough arguments\n");
			System.exit(1);
		}

		String training_file = args[0]; // file for the original dataset
		String testing_file = args[1]; // Target File name
		String method = args[2];
		
		List<String> result;
		List<List<Double>> trainingData = new ArrayList<>();
		List<List<Double>> testingData = new ArrayList<>();
		int yesCount = 0;

		// Read training file
		try {
            FileReader training = new FileReader(training_file);
            BufferedReader training_Reader = new BufferedReader(training);
            
            String line;
            int i = 0;
            
            while ((line = training_Reader.readLine()) != null) {
            	String[] tokens = line.split(",");
            	trainingData.add(new ArrayList<>());
        		
            	for (int j = 0; j < tokens.length-1; j++) {
        			trainingData.get(i).add(Double.parseDouble(tokens[j]));
            	}
        		trainingData.get(i).add(tokens[8].equals("yes") ? 1.0 : 0.0);
        		yesCount += tokens[8].equals("yes") ? 1 : 0;
        		i++;
            }
            
            training_Reader.close();
        }
        catch(IOException ex) {
            System.out.println("Error reading file '" + training_file + "'");
        }

        // read testing file
        try {
            FileReader testing = new FileReader(testing_file);
            BufferedReader testing_Reader =  new BufferedReader(testing);
            
            String line;
            int i = 0;
            
            while ((line = testing_Reader.readLine()) != null) {
            	String[] tokens = line.split(",");
            	testingData.add(new ArrayList<>());
        		for (String j : tokens) {
        			testingData.get(i).add(Double.parseDouble(j));
            	}
        		i++;
            }
            
            testing_Reader.close();
        
        } catch(IOException ex) {
            System.out.println("Error reading file '" + testing_file + "'");
        }

		if (method.equals("NB")) {
			result = nbClassifier(trainingData, testingData, yesCount);
		} else {
			int k = Integer.parseInt("" + method.charAt(0));
			result = knnClassifier(k, trainingData, testingData);
		}
		
		for (String s : result)
			System.out.println(s);
	}
}