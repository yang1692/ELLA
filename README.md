## This repo contains a project called JavaSVDExample based on **Apache maven 3.6.3**, **spark 2.4.5** and **scala 2.11**.  

### This project uses java spark to solve SVD problem in life-long learning algorithm. 
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
