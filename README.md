## JavaSVDExample folder contains a project called JavaSVDExample based on **Apache maven 3.6.3**, **spark 2.4.5** and **scala 2.11**.  

### This project uses java spark to solve SVD problem and prepare for life-long learning algorithm. 
### The class will only consider (numOfRow+1)/2 top singular value. And it will ingore value smaller than 1.0E-9d.
#### Descirption
#### The source file is /JavaSVDExample/src/main/java/com/mycompany/app/JavaSVDExample.   
#### The configuration file is pom.xml. If you wanna change the location or dependencies of source files, please pay attention to ***pom.xml***. It's the key. 
#### This class has a method called computeSVD. It takes a string, which is a table name in spark sql, and an integer k, which is the number of latent tasks. And it is designed as an interface for R. Thus, before using it, you need to create a dataframe in R and use copy_to function to upload it to spark sql.   
 
If you wanna re-compile it, use command ***mvn package***.  
It will create a jar file without dependencies which is JavaSVDExample/target/JavaSVDExample-1.0-SNAPSHOT.jar. It cannot run in ordianay way.  
If you wanna use it in R, put it inside $SPARK_HOME/jars. Then by calling invoke_new function, you can intialize a jobj in R. After getting a jobj,by calling invoke(jobj_name, "method_name",parameter1, parameter2,...), you can call a method of this java object.
Remember that, if a parameter is an integer, mark it with suffix *"L"*. For example, **invoke(my_jobj, "addInt", 1L, 2L)**.  

If you wanna create a jar file with all dependencies, use command ***mvn package assembly:single***. It will create a jar, which is JavaSVDExample/target/JavaSVDExample-1.0-SNAPSHOT-jar-with-dependencies.jar.   
If your class has a main method, under this condition, you can use command ***java -jar /JavaSVDExample/target/JavaSVDExample-1.0-SNAPSHOT-jar-with-dependencies.jar*** to run the jar file.   

D4JExample is a project shows how to use deeplearning4j with spark:  
	1. It does not include spark dependencies in its pom.xml file  
	2. Use command ***mvn package -DskipTests*** to compile the project  
	3. Put the *-bin.jar file into $SPARK_HOME/jars   
#### Created by Kui Yang 05/22/2020 

## D4JExample folder contains a functional prototype using D4J library. 
####D4J was abandoned in the end.  

## Ella folder contains the ELLA project based on **Apache maven 3.6.3**, **spark 2.4.5** and **scala 2.11**.  
0. Depends on the AutoEncoder code (repo:https://bitbucket.org/BeulahWorks/autoencoder/src/master/).   
  
1. There are 2 java classes:   
a. BasicOperationOnMatrix: singleton, includes all basic linear regression operations.  
b. Ella: includes all high level operations described in the ELLA paper.  
  
2. Usage:  
a. Call Autoencoder.sparkAutoEncoder() to get the model and the autoencoded table.  
b. Create an Ella object by calling its constructor. 1st parameter is k (number of latent tasks / model parameters) and 2nd is d (the length of features as the autoencoder output). See the paper for details.  
c. Call Ella.startJob().   
1st param is the spark dataframe's name, which contains both a vector column (aggregation of all features, required by spark's classifiers) and all raw columns/features.   
2nd param is the name of the vector column. It's an output of the autoencoder.  

The caller (R code) need to combine any new dataset with old ones and pass it to this function. The dataset name should remain same if it's an older task.  
It returns the trained model's theta.  

d. Caller (R code) then build a logistic regression model using the new theta and use that model for prediction.  

Limitations: 
1. currently it only supports binary classification, because spark's logistic regression model gives 1 theta (model parameter) which matches the ELLA algorithm.
For multiclass classification, spark will output multiple thetas and we still need to figure out how to use that with ELLA. Also it may affects the loss function derivative.

2. Currently the algorithm uses 0.5 as the model threshold. 
Spark's logistic regression allows changing the threshold. This is helpful when working with unbalanced class labels. This feature needs to be implemented in future.
#### Created by Kui Yang 08/01/2020  
