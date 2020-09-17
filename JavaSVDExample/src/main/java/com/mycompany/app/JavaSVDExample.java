package com.mycompany.app;

// $example on$
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
// $example off$
import org.apache.spark.sql.SQLContext;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.rdd.RDD;
// $example off$

/**
 * Example for SingularValueDecomposition.
 */
public class JavaSVDExample {
 
  public double[][] computeSVD( String tableName, int k) {
    //get the existing spark connection 
    SparkContext sc = SparkContext.getOrCreate();
    //get the sql connection of the spark connection
    SQLContext sql = SQLContext.getOrCreate(sc);
    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);
    List<Vector> listOfData = new ArrayList<Vector>();
    //read the table from sql by its name and convert it to a List. Dataset<Row> => List<Row>
    List<Row> table = sql.table(tableName).collectAsList();
    //get the size of the table
    int numOfCol = table.size();
    int numOfRow = table.get(0).size();
    //load the distribute data into a local List<Vector>
    for(int i = 0; i < numOfRow; i++){
	double[] array = new double[numOfCol];
	Row r = table.get(i);
	for(int j = 0; j < numOfCol; j++){
		array[j] = r.getDouble(j);	
	}
	Vector vec = Vectors.dense(array);
	listOfData.add(vec);
    }
    //Create a JavaRDD<Vector> from List<Row>
    JavaRDD<Vector> rows = jsc.parallelize(listOfData);
    // Create a RowMatrix from JavaRDD<Vector>.
    RowMatrix mat = new RowMatrix(rows.rdd());
    // Compute the top n singular values and corresponding singular vectors.
    SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD((numOfRow+1)/2, true, 1.0E-9d);
    RowMatrix U = svd.U();  // The U factor is a RowMatrix.
    // $example off$
    Vector[] collectPartitions = (Vector[]) U.rows().collect();
    double[][] result = new double[collectPartitions.length][k];
    for(int i = 0; i < collectPartitions.length; i++){
	result[i] = Arrays.copyOf(collectPartitions[i].toArray(), k);
    }
    return result;
  }

}
