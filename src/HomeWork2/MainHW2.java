package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.core.pmml.jaxbbindings.Decision;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class MainHW2 {
	private static double best_Pvalue; // global variables
	private static double best_ValidationError = Double.MAX_VALUE;
	private static int best_PvalueIndex = -1;

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception
	{
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		DecisionTree dt = new DecisionTree();
		chooseImpurityMeasure(dt, trainingCancer, validationCancer);
		printMessages(dt, trainingCancer, validationCancer);
		System.out.println("-----------------------------------------------------------");
		System.out.println("Best validation error at p_value: " + best_Pvalue);
		dt.m_p_value = best_PvalueIndex;
		trainData(dt, dt.impurityMeasure, trainingCancer);
		System.out.println("Test error with best tree: " + dt.calcAvgError(testingCancer));
		System.out.println("----------- Printing tree -----------");
		dt.printTree();
	}

	private static void printMessages(DecisionTree dt, Instances trainingCancer, Instances validationCancer) throws  Exception
	{
		printMessage(dt, -1, 1, trainingCancer, validationCancer);
		printMessage(dt, 0, 0.75, trainingCancer, validationCancer);
		printMessage(dt, 1, 0.5, trainingCancer, validationCancer);
		printMessage(dt, 2, 0.25, trainingCancer, validationCancer);
		printMessage(dt, 3, 0.05, trainingCancer, validationCancer);
		printMessage(dt, 4, 0.005, trainingCancer, validationCancer);
	}

	private static void printMessage(DecisionTree dt, int p_value, double realValue, Instances trainingCancer, Instances validationCancer) throws Exception
	{
		dt.m_p_value = p_value;
		trainData(dt, 0, trainingCancer);
		System.out.println("-----------------------------------------------------------");
		System.out.println("Decision tree with p_value of: " + realValue);
		System.out.println("The train error of the decision tree is: " + dt.calcAvgError(trainingCancer));
		dt.validationError = true;
		double error = dt.calcAvgError(validationCancer);
		if (error < best_ValidationError)
		{
			best_Pvalue = realValue;
			best_ValidationError = error;
			best_PvalueIndex = p_value;
		}
	}

	private static void trainData(DecisionTree dt, int impurityMeasure, Instances trainingCancer) throws Exception
	{
		dt.impurityMeasure = impurityMeasure;
		dt.buildClassifier(trainingCancer);
	}

	private static void chooseImpurityMeasure(DecisionTree dt, Instances trainingCancer, Instances validationCancer) throws Exception
	{
		trainData(dt, 1, trainingCancer); // train with Entropy
		double entropyError = dt.calcAvgError(validationCancer);
		System.out.println("Validation error using Entropy: " + entropyError);
		trainData(dt, 0, trainingCancer); // train with Gini
		double giniError = dt.calcAvgError(validationCancer);
		System.out.println("Validation error using Gini: " + giniError);

		dt.impurityMeasure = giniError < entropyError ? 0 : 1;
	}
}
