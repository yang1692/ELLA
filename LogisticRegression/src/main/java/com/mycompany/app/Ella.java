package com.mycompany.app;
import com.mycompany.app.BasicOperationOnMatrix;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.inversion.GaussJordanInverter;
import org.la4j.inversion.MatrixInverter;
import org.la4j.Matrices;
import java.util.Arrays;
import java.lang.Math;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
public class Ella {
    static BasicOperationOnMatrix BO = BasicOperationOnMatrix.getInstance(); 
    int k, d;
    HashMap<String, DenseMatrix> sMap, thetaMap, DMap ;
    DenseMatrix A,b,L;
    static SparkSession ss;
    static SparkContext sc;
    static SQLContext sql;
    static JavaSparkContext jsc;
    int counter = 0;
    public Ella(int k, int d){
        Random r = new Random();
        this.k = k;
        this.d = d;
        sMap = new HashMap<String, DenseMatrix>();
        thetaMap = new HashMap<String, DenseMatrix>();
        DMap = new HashMap<String, DenseMatrix>();
        A = DenseMatrix.zeros(k*d, k*d);
        b = DenseMatrix.zeros(k*d, 1);
        L = DenseMatrix.rand(d,k,r);
	ss = SparkSession.builder().getOrCreate();
	sc = ss.sparkContext();
	sql = ss.sqlContext();
	jsc = JavaSparkContext.fromSparkContext(sc);
    }
    public double[] startJob(String tableName, String featureName){
        if(sMap.get(tableName) == null){
            addNewJob(tableName, featureName);
        }
        else{
            updateExistJob(tableName, featureName);
        }
        updateLMatrix(0.1, tableName);
	return L.multiply(sMap.get(tableName)).toArray();
    }
    public void predict(){
	//DenseVector result = new DenseVector( L.multiply(sMap.get(tableName)).toArray() );
	//LogisticRegressionModel ellaLR = new LogisticRegressionModel("myLR",(Vector)result, 0.0);
    }
    public void addNewJob(String tableName, String featureName){
        Dataset<Row> training = sql.table(tableName);
	Dataset<Row> processed = training.withColumnRenamed(featureName,"features");
	processed.registerTempTable(tableName);
	//update theta
        DenseMatrix newTheta = myLR(tableName);
        thetaMap.put( tableName, newTheta );
        //update D
        DenseMatrix newD = calculateD(tableName);
        DMap.put( tableName, newD);
        //updaye s
        DenseMatrix news = SGD(tableName);
        sMap.put( tableName, SGD(tableName) );
        //update A
        DenseMatrix STimesStranspose = news.multiply( news.transpose() );
	DenseMatrix SSTtensorproductD = BO.tensorProduct(STimesStranspose, newD);
        try{A = BO.addMatrix(A, SSTtensorproductD);}
        catch (Exception e) {
            		e.printStackTrace();
        }
	//update B
        DenseMatrix thetaTtimesD = newTheta.transpose().multiply(newD);
	DenseMatrix sTensorProductthetaTtimesD = BO.tensorProduct(news.transpose(), thetaTtimesD);
        try{b = BO.addMatrix(b, sTensorProductthetaTtimesD);}
	catch (Exception e) {
            		e.printStackTrace();
        }
    }
    private void updateExistJob(String tableName, String featureName){
	Dataset<Row> training = sql.table(tableName);
	Dataset<Row> processed = training.withColumnRenamed(featureName,"features");
	processed.registerTempTable(tableName);
        //get old Matrics
        DenseMatrix olds = this.sMap.get(tableName);
        DenseMatrix oldD = this.DMap.get(tableName);
        DenseMatrix oldtheta = this.thetaMap.get(tableName);
        // A minus old data
        DenseMatrix STimesStranspose = olds.multiply( olds.transpose() );
	DenseMatrix SSTtensorproductD = BO.tensorProduct(STimesStranspose, oldD);
        try{A = BO.subMatrix(A, SSTtensorproductD);}
	catch (Exception e) {
            		e.printStackTrace();
        }
        // B minus old data
        DenseMatrix thetaTtimesD = oldtheta.transpose().multiply(oldD);
	DenseMatrix sTensorProductthetaTtimesD = BO.tensorProduct(olds.transpose(), thetaTtimesD);
        try{b = BO.subMatrix(b, sTensorProductthetaTtimesD);}
	catch (Exception e) {
            		e.printStackTrace();
        }
        //update theta
        DenseMatrix newTheta = myLR(tableName);
        thetaMap.put( tableName, newTheta );
        //update D
        DenseMatrix newD = calculateD(tableName);
        DMap.put( tableName, newD);
        //updaye s
        DenseMatrix news = SGD(tableName);
        sMap.put( tableName, SGD(tableName) );
        //update A
        try{STimesStranspose = news.multiply( news.transpose() );}
	catch (Exception e) {
            		e.printStackTrace();
        }
	SSTtensorproductD = BO.tensorProduct(STimesStranspose, newD);
        try{A = BO.addMatrix(A, SSTtensorproductD);}
	catch (Exception e) {
            		e.printStackTrace();
        }
        //update B
        thetaTtimesD = newTheta.transpose().multiply(newD);
	sTensorProductthetaTtimesD = BO.tensorProduct(news.transpose(), thetaTtimesD);
        try{b = BO.addMatrix(b, sTensorProductthetaTtimesD);}
	catch (Exception e) {
            		e.printStackTrace();
        }
    }
    public double test(String tableName, String featureName){
        Dataset<Row> training = sql.table(tableName);
	Dataset<Row> processed = training.withColumnRenamed(featureName,"features");
	processed.registerTempTable(tableName);
	return testThreshold(tableName);
    }
    public double testThreshold(String tableName){
	Dataset<Row> training = sql.table(tableName).select("features", "label");
	// Print the coefficients and intercept for logistic regression
	// We can also use the multinomial family for binary classification
	LogisticRegression mlr = new LogisticRegression()
		.setMaxIter(10)
		.setRegParam(0.3)
		.setElasticNetParam(0.8)
		.setFamily("binomial");
	// Fit the model
	LogisticRegressionModel mlrModel = mlr.fit(training);
	return 	mlr.getThreshold();
    }
    public double lossFunction(double[] thetaArray, String tableName/*, DenseMatrix feature, double[] prediction*/){
	Dataset<Row> data = sql.table(tableName);
	String sum = "";
	for (int i = 0; i < d; i++){
	    if(i != 0) sum += "+";
	    sum += thetaArray[i]+"*f"+i;
	}
	String query = "SELECT sum(if(label=1, -log10("+sum+"), -log10(1-("+sum+"))))/"+data.count()+" FROM "+ tableName;
	Row[] r = (Row[])ss.sql(query).collect();
	return r[0].getDouble(0);
    }

    public DenseMatrix calculateD(String tableName){
        DenseMatrix theta = this.thetaMap.get(tableName);
        double[] thetaArray = theta.toArray();
	int length =thetaArray.length;
	double[] matrix = new double[length*length];
	for(int i = 0; i < length; i++)
		for(int j = 0; j < length; j++)
		{
			matrix[i*length+j] = this.calSecondDerivative(i,j,tableName);
		}
	DenseMatrix result = new DenseMatrix(length, length, matrix);
	return result;
    } 
   public DenseMatrix updateLMatrix(double lambda, String tableName){
        int t = sMap.size();
	DenseMatrix AdividedT = BO.mulConst(1.0/t, A);
	try{DenseMatrix AdividedTAddI = BO.addMatrix(AdividedT, BO.createIdentityMatrix(k*d));
		DenseMatrix InversedA = BO.getInverseMatrix(AdividedTAddI);
		DenseMatrix bdividedT = BO.mulConst(1.0/t, b);
		return InversedA.multiply(b);
	}
	catch (Exception e) {
            		e.printStackTrace();
        }
	return DenseMatrix.ones(1,1);
	
    }
    //                                s: d*1 matrix; theta: d*1 matrix; L: d*d matrix; D: d*d matrix
   public double Lfunction(double u, DenseMatrix s, DenseMatrix theta, DenseMatrix L, DenseMatrix D){
	int dim = s.numRows();
	double sSum = BO.L1(s);
	double result = 0;
	result += u*sSum;
	DenseMatrix LtimesS = L.multiply(s);//d*d matrix time d*1 matrix = d*1 matrix;
	//v: d*1 matrix
	try{
		DenseMatrix v = BO.subMatrix(theta, LtimesS);
		DenseMatrix rightPart =  v.transpose().multiply(D.multiply(v));
		result += rightPart.apply(0,0);
	}
	catch (Exception e) {
            		e.printStackTrace();
        }
	return  result;
    }
    public DenseMatrix calDerivative(double u, String tableName ){
        DenseMatrix s = this.sMap.get(tableName);
        DenseMatrix D = this.DMap.get(tableName);
        DenseMatrix theta = this.thetaMap.get(tableName);
	int dim = s.numRows();
	double h = 0.001;
	double[] result = new double[dim];
	double[] sArray = s.toArray();
	double fx0 = Lfunction(u,s,theta,L,D);
	double[] sModified = sArray;
	for(int i = 0; i < dim; i++){
		if(i != 0) sModified[i-1] -= h; 
		sModified[i] += h;
		DenseMatrix sModifiedMatrix = new DenseMatrix(dim, 1, sModified);
		double fx1 = Lfunction(u,sModifiedMatrix,theta,L,D);
		result[i] = (fx1 - fx0) / h;
	}
	return new DenseMatrix(dim, 1, result);
    }
    public DenseMatrix SGD(String tableName){
        Random r = new Random();
        DenseMatrix s = DenseMatrix.rand(this.k, 1,r);
        DenseMatrix theta = this.thetaMap.get(tableName);
        DenseMatrix L = this.L;
        DenseMatrix D = this.DMap.get(tableName);
        double gamma = 0.1;
        double u = 0.1;
        double l0 = Lfunction(u,s,theta,L,D);	
        int iter = 1000;
	for(int i = 0; i < iter; i++){
		try{s = BO.subMatrix(s, BO.mulConst(gamma,calDerivative(i,tableName)));}
		catch (Exception e) {
            		e.printStackTrace();
        	}	
	}
	return s;
    }
    public double calSecondDerivative(int i, int j,String tableName){
	double h = 0.001;
	DenseMatrix theta = this.thetaMap.get(tableName);	
	double[]thetaModified = theta.toArray();
	thetaModified[i] += h;
	thetaModified[j] += h;
	double fx11 = lossFunction(thetaModified, tableName);
	thetaModified[j] -= 2*h;
	double fx10 = lossFunction(thetaModified, tableName);
	thetaModified[i] -= 2*h;	
	double fx00 = lossFunction(thetaModified, tableName);
	thetaModified[j] += 2*h;
	double fx01 = lossFunction(thetaModified, tableName);
	return (fx11+fx00-fx10-fx01)/(4*h*h);
    }
  
    public DenseMatrix myLR(String tableName){
	Dataset<Row> training = sql.table(tableName).select("features", "label");
	// Print the coefficients and intercept for logistic regression
	// We can also use the multinomial family for binary classification
	LogisticRegression lr = new LogisticRegression()
		.setMaxIter(10)
		.setRegParam(0.3)
		.setElasticNetParam(0.8)
		.setFitIntercept(false)
		.setFamily("binomial");
	// Fit the model
	LogisticRegressionModel lrModel = lr.fit(training);
	return 	new DenseMatrix(lrModel.coefficients().size(),1, lrModel.coefficients().toArray());
    }
}
