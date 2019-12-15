package spark.project;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class Sort {
	// this sorts the raw words by key within each mapper.
	
	public static void main(String[] args){
		// http://codingjunkie.net/spark-secondary-sort/
		SparkConf sparkConf = new SparkConf().setAppName("sort");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		
		String fileName = args[0];
		JavaRDD<String> textFile = jsc.textFile(fileName);
		
		long start = System.currentTimeMillis();
		
		JavaRDD<String> words = textFile.flatMap(new FlatMapFunction<String, String>() {
			public Iterator<String> call(String s) { return Arrays.stream(s.split(" ")).iterator(); }
		});
		
		JavaPairRDD<String, Integer> pairs = words.mapToPair(new PairFunction<String, String, Integer>() {
			public Tuple2<String, Integer> call(String s) { return new Tuple2<String, Integer>(s, 1); }
		});
		
		JavaPairRDD<String, Integer> sorted = pairs.sortByKey(true);
		
		long end = System.currentTimeMillis();
		float sec = (end - start) / 1000F;
		
		System.out.println("-------------------------------");
		System.out.println("-------------------------------");
		System.out.println("Ran in " + sec + " seconds");
		System.out.println("-------------------------------");
		System.out.println("-------------------------------");

		sorted.saveAsTextFile(args[1]);
	}
	
	

}
