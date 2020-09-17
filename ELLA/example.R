library(dplyr)
library(sparklyr)
install.packages("c5")
irisDF = data.frame(iris.train)
irisDF[,5] <- sapply(irisDF[,5],switch,"setosa"=1,"versicolor"=0,"virginica"=0)
irisDF <- irisDF[1:5]
irisDF


sparklyr::spark_disconnect_all()
sc <- sparklyr::spark_connect(master = "local")
irisSpark <- sparklyr::copy_to(sc,irisDF,"iris",overwrite = TRUE)
encoder <- sparklyr::invoke_new(sc, "org.apache.spark.ml.scaladl.AutoEncoder")
result <- sparklyr::invoke(encoder,"sparkAutoEncoder","iris",list("Sepal_Length", "Sepal_Width", "Petal_Length","Petal_Width"),"Species",list(4L,3L,4L),128L,333L,1L,"features","output","_trans")

iris_trans <- spark_read_table(sc,"iris_trans")
iris_sep <- sparklyr::sdf_separate_column(iris_trans,"output",c("f0","f1","f2"))
#iris_sep
sparklyr::sdf_register(iris_sep,"iris_trans")
ella <- sparklyr::invoke_new(sc, "com.mycompany.app.Ella",2L,3L)
result2 <- sparklyr::invoke(ella,"test","iris_trans","output")
#result2 <- sparklyr::invoke(ella,"startJob","iris_trans","output")
spark_read_table(sc,"splitted")
iris
result2
