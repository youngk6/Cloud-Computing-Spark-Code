package spark.project;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;

import scala.Tuple2;

public class Project {
	
	public static Dataset<Row> getDataFrame(SparkSession spark, String dataset_name) {
		StructType schema = new StructType()
			    .add("profile pic", "integer")
			    .add("nums/length username", "float")
			    .add("fullname words", "integer")
			    .add("nums/length fullname", "float")
			    .add("name==username", "integer")
			    .add("description length", "integer")
			    .add("external URL", "integer")
			    .add("private", "integer")
			    .add("#posts", "integer")
			    .add("#followers", "integer")
			    .add("#follows", "integer")
			    .add("fake", "integer");
		// For "fake_or_not"... 0 means genuine, 1 means spammer
		
		Dataset<Row> df = spark.read()
			    .option("mode", "DROPMALFORMED")
			    .option("header", "true")
			    .schema(schema)
			    .csv(dataset_name);
		
		return df;
	}

	public static void main(String[] args)
	{
		// create Spark session
		SparkSession spark = SparkSession
			    .builder()
			    .appName("Fake Instagram Accounts")
			    .getOrCreate();
		
		// read train and test data
		Dataset<Row> train_df = getDataFrame(spark, "insta_train.csv");
		
		// Create vector of features... since it has to be in that form
		// output is a sparse vector representation
		String[] featureCols = {"profile pic", "nums/length username", "fullname words", "nums/length fullname", "name==username", "description length", "external URL", "private", "#posts", "#followers", "#follows"};
				VectorAssembler featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features");
				Dataset<Row> train_df_feat = featureAssembler.transform(train_df);
			
		/*
		Row[] dataRows = (Row[]) train_df_feat.collect();
		for (Row row : dataRows) {
			System.out.println("Row : "+ row);
			//for (int i = 0; i < row.length(); i++) {
			//	System.out.println("Row Data : "+row.get(i));
			//}
		}
		*/
				
		// start timer
		long start = System.currentTimeMillis();
		
		StringIndexerModel labelIndexer = new StringIndexer()
				  .setInputCol("fake")
				  .setOutputCol("indexedLabel")
				  .fit(train_df_feat);
		
		VectorIndexerModel featureIndexer = new VectorIndexer()
				  .setInputCol("features")
				  .setOutputCol("indexedFeatures")
				  .fit(train_df_feat);
		
		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = train_df_feat.randomSplit(new double[] {0.7, 0.3});
		Dataset<Row> train_train = splits[0];
		Dataset<Row> train_test = splits[1];
		
		
		// start creating model
		RandomForestClassifier rf = new RandomForestClassifier()
				  .setLabelCol("indexedLabel")
				  .setFeaturesCol("indexedFeatures");
		
		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
		  .setInputCol("prediction")
		  .setOutputCol("predictedLabel")
		  .setLabels(labelIndexer.labels());
		
		// Chain indexers and forest in a Pipeline
		Pipeline pipeline = new Pipeline()
		  .setStages(new PipelineStage[] {labelIndexer, featureIndexer, rf, labelConverter});
		
		// Train model. This also runs the indexers.
		PipelineModel model = pipeline.fit(train_train);

		// Make predictions.
		Dataset<Row> predictions = model.transform(train_test);
		
		// Select example rows to display.
		//predictions.show();
		
		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("indexedLabel")
		  .setPredictionCol("prediction")
		  .setMetricName("accuracy");
		
		double accuracy = evaluator.evaluate(predictions);
		
		System.out.println("Test Error = " + (1.0 - accuracy));
		System.out.println("Accuracy = " + accuracy);
		
		long end = System.currentTimeMillis();
		float sec = (end - start) / 1000F; 
		
		System.out.println("-------------------------------");
		System.out.println("Ran in " + sec + " seconds");
		System.out.println("-------------------------------");
		
		
		
	}
}