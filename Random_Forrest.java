package spark.project;

import org.apache.hadoop.fs.DF;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class Random_Forrest {
	
	public static Dataset<Row> getDataFrame(SparkSession spark, String dataset_name) {
		StructType schema = new StructType()
			    .add("_c0", "integer")
			    .add("_c1", "float")
			    .add("_c2", "float")
			    .add("_c3", "float")
			    .add("_c4", "float")
			    .add("_c5", "float")
			    .add("_c6", "float");
		
		Dataset<Row> df = spark.read()
			    .option("mode", "DROPMALFORMED")
			    .option("header", "false")
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
		String df_name = args[0];
		Dataset<Row> train_df = getDataFrame(spark, df_name);
		
		//train_df.show();
		 
		// Create vector of features... since it has to be in that form
		String[] featureCols = {"_c1", "_c2", "_c3", "_c4", "_c5", "_c6"};
				VectorAssembler featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features");
				Dataset<Row> train_df_feat = featureAssembler.transform(train_df);
				
		// start timer
		long start = System.currentTimeMillis();
		
		StringIndexerModel labelIndexer = new StringIndexer()
				  .setInputCol("_c0")
				  .setOutputCol("indexedLabel")
				  .fit(train_df_feat);
		
		VectorIndexerModel featureIndexer = new VectorIndexer()
				  .setInputCol("features")
				  .setOutputCol("indexedFeatures")
				  .fit(train_df_feat);
		
		//train_df_feat.show();
		
		
		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = train_df_feat.randomSplit(new double[] {0.7, 0.3});
		Dataset<Row> air_train = splits[0];
		Dataset<Row> air_test = splits[1];
		
		
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
		PipelineModel model = pipeline.fit(air_train);

		// Make predictions.
		Dataset<Row> predictions = model.transform(air_test);
		
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
		System.out.println("-------------------------------");
		System.out.println("Ran in " + sec + " seconds");
		System.out.println("-------------------------------");
		System.out.println("-------------------------------");
		
		
		
	}
	
	
}
