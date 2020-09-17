package com.mycompany.app;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
public class D4JExample {
	private static Logger log = LoggerFactory.getLogger(D4JExample.class);
	public long SparkMultiLayerNetwork(String filePath,int labelIndex,int numClasses,int numEpochs) throws  Exception{
		SparkSession ss = SparkSession.builder().getOrCreate();
		SparkContext sc = ss.sparkContext();
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);
		int numLinesToSkip = 0;
		char delimiter = ',';
		int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

		//load to local and parallelize
		RecordReader trainrecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
		trainrecordReader.initialize(new FileSplit(new File(filePath)));

		RecordReader testrecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
		testrecordReader.initialize(new FileSplit(new File(filePath)));

		// Load all the train to calculate normalizer
		DataSetIterator fulliterator = new RecordReaderDataSetIterator(trainrecordReader, 98, labelIndex, numClasses);

		DataSetIterator iterTrain = new RecordReaderDataSetIterator(trainrecordReader, batchSize, labelIndex, numClasses);
		DataSetIterator iterTest = new RecordReaderDataSetIterator(testrecordReader, batchSize, labelIndex, numClasses);

		//Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network

		NormalizerStandardize preProcessor = new NormalizerStandardize();
		preProcessor.fit(fulliterator);
		iterTrain.setPreProcessor(preProcessor);
		iterTest.setPreProcessor(preProcessor);

		List<DataSet> trainDataList = new ArrayList<>();
		List<DataSet> testDataList = new ArrayList<>();
		while (iterTrain.hasNext()) {
		    trainDataList.add(iterTrain.next());
		}
		while (iterTest.hasNext()) {
		    testDataList.add(iterTest.next());
		}

		JavaRDD<DataSet> trainData = jsc.parallelize(trainDataList);
		JavaRDD<DataSet> testData = jsc.parallelize(testDataList);

		//use rdd at the beginning
		/*JavaRDD<String> rddString = jsc.textFile(filePath);
		RecordReader recordReader = new CSVRecordReader(',');
		JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));
		JavaRDD<DataSet> trainData = rddWritables.map(new DataVecDataSetFunction(labelIndex, numClasses, false));
		JavaRDD<DataSet> testData = trainData;*/
		//DataSet trainingData = iterTrain.next();
		//DataSet testData = iterTest.next();

		final int numInputs = 4;
		int outputNum = 3;
		int iterations = 100;
		long seed = 6;
		int batchSizePerWorker = 16;


		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		        .seed(seed)
		        //.iterations(iterations)
		        .activation(Activation.TANH)
		        .weightInit(WeightInit.XAVIER)
		        //.learningRate(0.1)
		        /*.regularization(true)*/.l2(1e-4)
		        .list()
		        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
		                .build())
		        .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
		                .build())
		        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
		                .activation(Activation.SOFTMAX)
		                .nIn(3).nOut(outputNum).build())
		        /*.backprop(true).pretrain(false)*/
		        .build();

		TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
		        .averagingFrequency(5)
		        .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
		        .batchSizePerWorker(batchSizePerWorker)
		        .build();

		SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(jsc, conf, tm);

		for (int i = 0; i < numEpochs; i++) {
		    sparkNet.fit(trainData);
		    log.info("Completed Epoch {}", i);
		}
		MultiLayerNetwork localmultilayer = sparkNet.getNetwork();
		org.deeplearning4j.nn.layers.OutputLayer outLayer = (org.deeplearning4j.nn.layers.OutputLayer)localmultilayer.getOutputLayer(); 
		
		Evaluation eval = new Evaluation(3);
		MultiLayerNetwork model = sparkNet.getNetwork();
		testData.collect().forEach(entry -> {
		    INDArray output = model.output(entry.getFeatures()); //get the networks prediction
		    eval.eval(entry.getLabels(), output); //check the prediction against the true class
		});
		Map<String,INDArray> param = model.paramTable();
		long numLable = model.numParams();
		return numLable;//eval.accuracy();
	}

}
