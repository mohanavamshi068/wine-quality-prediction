package wineq;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;

public class TrainRF {

	private static String TRAINING_DATASET_FILENAME = "TrainingDataset.csv";
	private static String VALIDATION_DATASET_FILENAME = "ValidationDataset.csv";
	private static String TRAINED_MODEL_PATH = "TrainedModel";

	public static void main(String[] args) {

		try {
			// Check if the training data set file exists
			System.out.println("Checking if the Training data file is present.");
			File trainDataFile = new File(TRAINING_DATASET_FILENAME);

			if (trainDataFile.exists()) {
				System.out.println("Training data set found.");

				System.out.println("Going to create Spark Session.");

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

				System.out.println("Spark Session created.");

				// Now start model training
				TrainRF trainRF = new TrainRF();
				trainRF.trainUsingRF(session);

			} else {
				System.out.print(TRAINING_DATASET_FILENAME + " not found. Please copy the file and try again.");
			}
		} catch (Exception e) {
			System.out.println("Something wrong: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public void trainUsingRF(SparkSession session) {

		// Read the training data file into a Spark DataSet
		Dataset<Row> trainingDataDS = getDataSet(session, TRAINING_DATASET_FILENAME).cache();
		
//		// Index the label column
//		StringIndexerModel labelIndexer = new StringIndexer()
//		  .setInputCol("label")
//		  .setOutputCol("idxLabel")
//		  .fit(trainingDataDS);
//		
//		// Index feature columns
//		VectorIndexerModel featureIndexer = new VectorIndexer()
//		  .setInputCol("features")
//		  .setOutputCol("idxFeatures")
//		  .setMaxCategories(4)
//		  .fit(trainingDataDS);		
//		

		RandomForestClassifier rf = new RandomForestClassifier()
				.setLabelCol("label")
				.setFeaturesCol("features");
		
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {rf});
		
		PipelineModel pipelineModel = pipeline.fit(trainingDataDS);
		
		Dataset<Row> validationDataDS = getDataSet(session, VALIDATION_DATASET_FILENAME).cache();
		Dataset<Row> predictions = pipelineModel.transform(validationDataDS);
		predictions.select("features", "label", "prediction").show(5);
		
		// Now print the prediction.
		showModelMetrics(predictions);

//		RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(pipelineModel.stages()[2]);
//		System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());
		
		try {
			pipelineModel.write().overwrite().save(TRAINED_MODEL_PATH);
		} catch (IOException e) {
			System.out.println("Something went wrong: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public static void showModelMetrics(Dataset<Row> predictions) {
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction");

		evaluator.setMetricName("accuracy");
		System.out.println("Accuracy for the Random Forest Model prediction is: " + evaluator.evaluate(predictions));

		evaluator.setMetricName("f1");
		System.out.println("F1 value of the Random Forest Model prediction is: " + evaluator.evaluate(predictions));
	}

	public static Dataset<Row> getDataSet(SparkSession session, String fileName) {

		Dataset<Row> inputDS = session.read()
				.format("csv")
				.option("header", "true")
				.option("inferSchema", true)
				.option("sep", ";")
				.option("multiline", true)
				.option("quote", "\"")
				.option("dateFormat", "M/d/y")
				.load(fileName);

		System.out.println("Read " + fileName + " as Spark Data Set. Printing schema.");

		inputDS.printSchema();

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
		
		System.out.println("Printing schema after renaming columns");
		inputDSRenamed.printSchema();

		System.out.println("Displaying first 10 records of the Dataset");
		inputDSRenamed.show(10);

		VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(
				new String[] { "alcohol", "sulphates", "pH", "density", "free_sulfur_dioxide", "total_sulfur_dioxide",
						"chlorides", "residual_sugar", "citric_acid", "volatile_acidity", "fixed_acidity" })
				.setOutputCol("features");

		Dataset<Row> featuresDS = vectorAssembler.transform(inputDSRenamed).select("label", "features");

		return featuresDS;
	}
}
