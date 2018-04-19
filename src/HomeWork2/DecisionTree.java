package HomeWork2;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.unsupervised.instance.RemoveWithValues;
import java.util.ArrayDeque;
import java.util.Queue;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	Instances m_Instances;
	private int m_classIndex;
	boolean m_leaf;
	int m_freedomDegree;

	Node(Instances data)
	{
		children = new Node[0];
		m_Instances = data;
		m_classIndex = m_Instances.classIndex();
		returnValue = getReturnValue();
		m_leaf = false;
	}

	private double getReturnValue()
	{
		int countOnes = 0;
		int countZeros;
		double classification;

		for (int i = 0; i < m_Instances.size(); i++)
		{
			if (m_Instances.instance(i).value(m_classIndex) == 1.0)
			{
				countOnes++;
			}
		}

		countZeros = m_Instances.size() - countOnes;
		classification = (countOnes >= countZeros) ? 1.0 : 0.0;

		return classification;
	}

	boolean isPerfectlyClassified()
	{

		double classify = m_Instances.size() > 0 ? m_Instances.firstInstance().value(m_classIndex): 0;
		double nextClassify;
		boolean isPerfectlyClassified = true;

		for (int i = 1; i < m_Instances.size(); i++)
		{
			nextClassify = m_Instances.instance(i).value(m_classIndex);
			if (classify != nextClassify)
			{
				isPerfectlyClassified = false;
				break;
			}

			classify = nextClassify;
		}

		if (isPerfectlyClassified)
		{
			m_leaf = true;
		}

		return isPerfectlyClassified;
	}

}

public class DecisionTree implements Classifier {
	private Node rootNode;
	private int m_classIndex;
	private int m_truNumAttributes;
	static int impurityMeasure; // 0 for gini , 1 for entropy
	int m_p_value = -1;
	private int depth = -1;
	private static double avgHeight;
	private static double sumOfHeight;
	boolean validationError;
	private double[][] chiTable = {
			{0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438}, // p-value 0.75
			{0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340}, // p-value 0.5
			{1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845}, // p-value 0.25
			{3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026}, // p-value 0.05
			{7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300}}; // p-value 0.005

	@Override
	public void buildClassifier(Instances arg0) throws Exception
	{
		depth = -1;
		validationError = false;
		m_classIndex = arg0.classIndex();
		m_truNumAttributes = arg0.numAttributes()-1;
		buildTree(arg0, impurityMeasure);
		avgHeight = (sumOfHeight / arg0.size());
	}

	@Override
	public double classifyInstance(Instance instance)
	{
		Node n = rootNode;
		int currentAttrIndex;
		int height = 0;

		while (!n.m_leaf)
		{
			currentAttrIndex = n.attributeIndex;
			if (n.children.length == 0) break;
			if (n.children[(int)(instance.value(currentAttrIndex))] == null) break;

			n = n.children[(int) (instance.value(currentAttrIndex))];
			height++;
		}

		if (validationError)
		{
			if (height > depth)
			{
				depth = height;
			}

			sumOfHeight += height;
		}

		return n.returnValue;
	}

	void printTree()
	{
		System.out.println("Root");
		printTree(rootNode,"");
	}

	private void printTree(Node n, String tab)
	{
		if (n.m_leaf || n.children.length == 0)
		{
			System.out.println("  " + tab + "Leaf. Returning value: " + n.returnValue);
		}
		else
		{
			System.out.println(tab + "Returning value: " + n.returnValue);
			for (int i = 0; i < n.children.length; i++)
			{
				if (n.children[i] != null && n.children[i].m_Instances.size() > 0)
				{
					System.out.println("  " + tab + "If attribute " + n.attributeIndex  + " = " + i);
					printTree(n.children[i],("  " + tab));
				}
			}
		}
	}


	/** Builds the decision tree on given data set **/
	private void buildTree(Instances data, int impurityMeasure) throws Exception
	{
		rootNode = new Node(data);
		rootNode.parent = null;
		Node next;
		Queue<Node> q = new ArrayDeque<>();
		double chiSquare = 0;

		q.add(rootNode);
		while (q.size() > 0)
		{
			next = q.remove();
			if (!next.isPerfectlyClassified() && next.m_Instances.size() > 0)
			{
				next.attributeIndex = getBestAttribute(next.m_Instances, next, impurityMeasure);
				if (next.attributeIndex != -1)
				{
					if (m_p_value > -1) // calculates chi and freedomDegree only when we are pruning
					{
						next.m_freedomDegree = (next.m_Instances.numDistinctValues(next.attributeIndex) - 2);
						chiSquare = calcChiSquare(next.m_Instances, next.attributeIndex);
					}
					if (m_p_value == -1 || chiSquare >= chiTable[m_p_value][next.m_freedomDegree])
					{
						createChildrenForNode(next, data.attribute(next.attributeIndex).numValues());
						insertChildrenToQueue(next, q);
					}
				}
				else
				{
					next.m_leaf = true;
				}
			}
		}
	}

	/** inserting the relevant children to the queue **/
	private void insertChildrenToQueue(Node n, Queue<Node> q)
	{
		for (int i = 0; i < n.children.length; i++)
		{
			if (n.children[i] != null && n.children[i].m_Instances.size() > 0)
			{
				q.add(n.children[i]);
			}
		}
	}

	/** splits the instances between the children of a specific node according to best attribute chosen **/
	private void createChildrenForNode(Node n, int numOfChildren) throws Exception
	{
		n.children = new Node[numOfChildren];
		Instances instances;
		int i = 0;
		while (i < numOfChildren)
		{
			instances = createSubset(n.attributeIndex, n.m_Instances, i);
			if (instances.size() > 0)
			{
				n.children[i] = new Node(instances);
				n.children[i].parent = n;

			}

			i++;
		}
	}

	/** gets the best attribute to split according to using the formula learnt in class **/
	private int getBestAttribute(Instances data, Node n, int impurityMeasure) throws Exception
	{
		int bestAttributeSeenSoFar = -1;
		double bestGainSeenSoFar = Double.MIN_VALUE;
		double currentGain;

		for (int i = 0; i < m_truNumAttributes; i++)
		{
			if (isAvailableAttribute(n, i))				// checks what attributes available for this branch
			{
				currentGain = calcGain(data, i, impurityMeasure);
				if (currentGain > bestGainSeenSoFar) {
					bestGainSeenSoFar = currentGain;
					bestAttributeSeenSoFar = i;
				}
			}
		}

		return bestAttributeSeenSoFar;
	}

	/** checks if specific branch was splitted already according to some attribute **/
	private boolean isAvailableAttribute(Node n, int i)
	{
		boolean isAvailable = true;
		n = n.parent;
		while (n != null)
		{
			if (n.attributeIndex == i)
			{
				isAvailable = false;
				break;
			}

			n = n.parent;
		}

		return isAvailable;
	}

	/** Calculate the average error on a given instances set **/
	double calcAvgError(Instances data)
	{
		if (validationError)
		{
			avgHeight = 0;
			sumOfHeight = 0;
		}

		double numOfClassificationMistakes = 0;
		Instance currentInstance;
		int numOfInstances = data.size();

		for (int i = 0; i < numOfInstances; i++)
		{
			currentInstance = data.instance(i);
			if (classifyInstance(currentInstance) != currentInstance.value(m_classIndex))
			{
				numOfClassificationMistakes++;
			}
		}

		if(validationError)
		{
			avgHeight = sumOfHeight / (double) data.size();
			System.out.println("Max height on validation data: " + depth);
			System.out.println("Average height on validation data: " + avgHeight);
			System.out.println("The validation error of the decision tree is: " + (numOfClassificationMistakes / numOfInstances));
		}

		return (numOfClassificationMistakes / numOfInstances);
	}

	/** calculates the gain
	 (giniGain or informationGain depending on the impurity measure)
	 of splitting the input data according to the attribute.
	 **/
	private double calcGain(Instances data, int attributeIndex, int impurityMeasure) throws Exception
	{
		double giniGain = calcGiniGain(data, attributeIndex);
		double infoGain = calcInformationGain(data,attributeIndex);

		return (impurityMeasure == 0) ? giniGain : infoGain;
	}

	private double calcGiniGain(Instances data, int attributeIndex) throws Exception
	{
		double impurityBeforeSplit = calcGini(getProbabilities(data));	// G(S)
		double weightedAvgOfImpurity = 0;
		double giniOfSubset;	// G(Sv)
		double ratio;			// |Sv|/|S|
		Instances subSet;

		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++)
		{
			subSet = createSubset(attributeIndex, data, i);
			giniOfSubset = calcGini(getProbabilities(subSet)); // G(Sv)
			if (data.size() > 0)
			{
				ratio = ((double)subSet.size() / (double)data.size());
				weightedAvgOfImpurity += (ratio * giniOfSubset);
			}
		}
		return impurityBeforeSplit - weightedAvgOfImpurity;
	}

	private double calcInformationGain(Instances data, int attributeIndex) throws  Exception
	{
		double entropyOfSet = calcEntropy(getProbabilities(data)); // H(S)
		double weightedAvgOfImpurity = 0;
		double entropyOfSubset;
		double ratio;
		Instances subSet;

		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++)
		{
			subSet = createSubset(attributeIndex, data, i);
			entropyOfSubset = calcEntropy(getProbabilities(subSet)); // H(Sv)
			if (data.size() > 0)
			{
				ratio = ((double)subSet.size() / (double)data.size());
				weightedAvgOfImpurity += (ratio * entropyOfSubset);
			}
		}

		return entropyOfSet - weightedAvgOfImpurity;
	}

	/** creates a subset of instances according to some specific attribute value **/
	private Instances createSubset(int attributeIndex, Instances data, int index) throws Exception
	{
		RemoveWithValues rmv = new RemoveWithValues();
		rmv.setInvertSelection(true);
		rmv.setAttributeIndex("" + (attributeIndex+1));
		rmv.setNominalIndices("" + (index+1));
		rmv.setInputFormat(data);

		return rmv.useFilter(data, rmv);
	}

	/** Calculates the Entropy of a random variable. **/
	private double calcEntropy(double[] probabilities)
	{
		double sumOfEntropy = 0;
		for (int i = 0; i < probabilities.length; i++)
		{
			sumOfEntropy += (probabilities[i] * log2(probabilities[i]));
		}

		return -sumOfEntropy;
	}

	private double log2(double x)
	{
		if (x == 0)
		{
			return 0;
		}

		return (Math.log10(x) / Math.log10(2));
	}

	/** Calculates the Gini of a random variable. **/
	private double calcGini(double[] probabilities)
	{
		double sumOfSquaredProbabilities = 0;

		for(int i = 0; i < probabilities.length; i++)
		{
			sumOfSquaredProbabilities += probabilities[i]*probabilities[i];
		}

		return (1 - sumOfSquaredProbabilities);
	}

	private double[] getProbabilities(Instances data)
	{
		double countOnes = 0;
		double dataSize = data.size();
		double[] probabilities = new double[2];

		for (int i = 0; i < dataSize; i++)
		{
			if (data.instance(i).value(m_classIndex) == 1)
			{
				countOnes++;
			}
		}

		if (dataSize > 0)
		{
			probabilities[0] = (countOnes / dataSize);
			probabilities[1] = 1 - probabilities[0];
		}

		return probabilities;
	}

	/** Calculates the chi square statistic of splitting the data according to the
	 splitting attribute as learned in class **/
	private double calcChiSquare(Instances data, int attributeIndex) throws Exception
	{
		double numOfInstancesWithAttrValue;				// Df
		double numOfInstancesWithAttrValueAndNegative;		// pf
		double numOfInstancesWithAttrValueAndPositive;		// nf
		double expectedValueNegative;					// E0
		double expectedValuePositive;					// E1
		double chiSquare = 0;
		double leftSide;
		double rightSide;
		Instances subSet;

		for(int i = 0; i < data.attribute(attributeIndex).numValues(); i++) {
			rightSide = 0;
			leftSide = 0;
			subSet = createSubset(attributeIndex, data, i);
			if (subSet.size() > 0) {
				numOfInstancesWithAttrValue = subSet.size();
				numOfInstancesWithAttrValueAndNegative =  (getProbabilities(subSet)[1] * numOfInstancesWithAttrValue);
				numOfInstancesWithAttrValueAndPositive =  (getProbabilities(subSet)[0] * numOfInstancesWithAttrValue);
				expectedValueNegative = numOfInstancesWithAttrValue * getProbabilities(data)[1];
				expectedValuePositive = numOfInstancesWithAttrValue * getProbabilities(data)[0];

				if (expectedValueNegative != 0) {
					leftSide = ((Math.pow((numOfInstancesWithAttrValueAndNegative - expectedValueNegative), 2)) / expectedValueNegative);
				}
				if (expectedValuePositive != 0) {
					rightSide = ((Math.pow((numOfInstancesWithAttrValueAndPositive - expectedValuePositive), 2)) / expectedValuePositive);
				}

				chiSquare += (leftSide + rightSide);
			}
		}

		return chiSquare;
	}


	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
