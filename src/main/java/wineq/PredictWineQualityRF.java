package wineq;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

public class PredictWineQualityRF {

	private static String TEST_DATASET_FILENAME = "TestDataset.csv";
	private static String TRAINED_MODEL_PATH = "TrainedModel";

	public static void main(String[] args) {

		// Check if the training data set file exists
		System.out.println("Checking if the Test data file is present..");
		
		File testFile = new File(TEST_DATASET_FILENAME);

		if (testFile.exists()) {
			
			System.out.println("Test data set file found. Starting Spark session.");

			// Create Spark session with Driver memory 1 GB & Executor memory 2 GB
	        Logger.getLogger("org").setLevel(Level.ERROR);
	        Logger.getLogger("akka").setLevel(Level.ERROR);
	        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
	        Logger.getLogger("com.github").setLevel(Level.ERROR);

	        SparkSession session = SparkSession.builder()
					.appName("wine-quality-model-trainer-job")
					.master("local[*]")
					.config("spark.executor.memory", "2147480000")
					.config("spark.driver.memory", "1073741824")
					.getOrCreate();

			PredictWineQualityRF predictWineQuality = new PredictWineQualityRF();
			predictWineQuality.predict(session, TRAINED_MODEL_PATH);
			
		} else {
			System.out.print(TEST_DATASET_FILENAME + " not found. Please copy the file and try again.");
		}

	}

	public void predict(SparkSession session, String modelPath) {

		PipelineModel pipelineModel = PipelineModel.load(modelPath);
		System.out.println("Loaded the Random Forest model.");
		
		Dataset<Row> testDS = getTestDataSet(session, TEST_DATASET_FILENAME).cache();
		System.out.println("Loaded the test data set.");
		
		Dataset<Row> predictedDS = pipelineModel.transform(testDS).cache();
		System.out.println("Scored the data set");
		
		//predictedDS.select("features", "label", "prediction").show(5, false);
		showMetrics(predictedDS);

	}
	public static Dataset<Row> getTestDataSet(SparkSession session, String fileName) {

		Dataset<Row> inputDS = session.read()
				.format("csv")
				.option("header", "true")
				.option("inferSchema", true)
				.option("sep", ";")
				.option("multiline", true)
				.option("quote", "\"")
				.option("dateFormat", "M/d/y")
				.load(fileName);

		System.out.println("Read " + fileName + " as Spark Data Set.");

		//inputDS.printSchema();

		// We are trying to predict the value for "Quality". Renaming that as "Label" as a ML standard.
		// Also rename the columns to replace the space with underscore
		
		Dataset<Row> inputDSRenamed = inputDS.withColumnRenamed("quality", "label")
				.withColumnRenamed("fixed acidity", "fixed_acidity")
                .withColumnRenamed("volatile acidity", "volatile_acidity")
                .withColumnRenamed("citric acid", "citric_acid")
                .withColumnRenamed("residual sugar", "residual_sugar")
                .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
                .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
                .cache();
		
		
		VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(
				new String[] { "alcohol", "sulphates", "pH", "density", "free_sulfur_dioxide", "total_sulfur_dioxide",
						"chlorides", "residual_sugar", "citric_acid", "volatile_acidity", "fixed_acidity" })
				.setOutputCol("features");

		Dataset<Row> featuresDS = vectorAssembler.transform(inputDSRenamed).select("label", "features");

		return featuresDS;
	}
	
	public static void showMetrics(Dataset<Row> predictions) {
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

		evaluator.setMetricName("accuracy");
		System.out.println("Accuracy for the Random Forest Model prediction is: " + evaluator.evaluate(predictions));

		evaluator.setMetricName("f1");
		System.out.println("F1 value of the Random Forest Model prediction is: " + evaluator.evaluate(predictions));
	}
}