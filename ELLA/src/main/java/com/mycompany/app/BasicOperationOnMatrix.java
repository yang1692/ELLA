package com.mycompany.app;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.DenseVector;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.inversion.GaussJordanInverter;
import org.la4j.inversion.MatrixInverter;
import org.la4j.Matrices;
import java.util.Arrays;
import java.lang.Math;
public class BasicOperationOnMatrix {
    private static final BasicOperationOnMatrix INSTANCE = new BasicOperationOnMatrix();

    private BasicOperationOnMatrix() {}

    public static BasicOperationOnMatrix getInstance() {
        return INSTANCE;
    }
    public DenseMatrix createIdentityMatrix(int size){
	double[] diag = new double[size];
	Arrays.fill(diag, 1);
	DenseVector v = new DenseVector(diag);
	return DenseMatrix.diag(v);
    }
    public double L1(DenseMatrix m){
	int row = m.numRows(), col = m.numCols();
	double result = 0;
        for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++)
			result += Math.abs(m.apply(i,j));	
	}
	return result;
   }
    public DenseMatrix tensorProduct(DenseMatrix m1, DenseMatrix m2){
  	int row1 = m1.numRows(), row2 = m2.numRows();
	int col1 = m1.numCols(), col2 = m2.numCols();
	double[] result = new double[row1*row2*col1*col2];
	for(int i = 0; i < row1; i++){
		for(int j = 0; j < col1; j++){
			for(int k = 0; k < row2; k++){
				for(int l = 0; l < col2; l++){
					result[(col2*j+l)*row1*row2+row2*i+k] = m1.apply(i,j)*m2.apply(k,l);
				}
			}	
		}
	}
	return new DenseMatrix(row1*row2, col1*col2, result);
    }
    public DenseMatrix addMatrix (DenseMatrix m1, DenseMatrix m2)throws Exception{
	int row = m1.numRows(), col = m1.numCols();
    	if(row != m2.numRows() || col != m2.numCols()){
		throw new IllegalArgumentException("The size of A don't match the size of B. A("+row+", "+col+") B:("+m2.numRows()+", "+m2.numCols()+")");
	}
	else{
		double[] m1Array = m1.toArray();
		double[] m2Array = m2.toArray();
		int length = m1Array.length;
		for(int i = 0; i < length; i++)
			m1Array[i] += m2Array[i];
		return new DenseMatrix(row, col, m1Array);	
	}
    }
    public DenseMatrix subMatrix (DenseMatrix m1, DenseMatrix m2)throws Exception{
	int row = m1.numRows(), col = m1.numCols();
    	if(row != m2.numRows() || col != m2.numCols()){
		throw new IllegalArgumentException("The size of A don't match the size of B. A("+row+", "+col+") B:("+m2.numRows()+", "+m2.numCols()+")");
	}
	else{
		double[] m1Array = m1.toArray();
		double[] m2Array = m2.toArray();
		int length = m1Array.length;
		for(int i = 0; i < length; i++)
			m1Array[i] -= m2Array[i];
		return new DenseMatrix(row, col, m1Array);	
	}
    }
    public DenseMatrix mulConst(Double constant, DenseMatrix m2){
	int row = m2.numRows(), col = m2.numCols();
	double[] m2Array = m2.toArray();
	int length = m2Array.length;
	for(int i = 0; i < length; i++)
		m2Array[i] *= constant;
	return new DenseMatrix(row, col, m2Array);	
    }

    public DenseMatrix getInverseMatrix(DenseMatrix m){
	double[] dataArray = m.toArray();
	int row = m.numRows(), col = m.numCols();
	org.la4j.Matrix a = new Basic1DMatrix(row, col, dataArray);
	MatrixInverter gauss = new GaussJordanInverter(a);
	org.la4j.Matrix b = gauss.inverse();
	for(int i = 0; i < col; i++){
		for(int j =0; j < row; j++){
			dataArray[i*row+j] = b.get(i,j);		
		}
        }
	return new DenseMatrix(row, col, dataArray);
    }
}
