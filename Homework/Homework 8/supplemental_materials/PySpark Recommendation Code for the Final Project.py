# Import necessary PySpark modules
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# Initialize SparkSession
spark = SparkSession.builder.appName("RecommendationSystem").getOrCreate()

# Step 1: Load training data
training = spark.read.csv("trainItem.data", header=False)
training = training.withColumnRenamed("_c0", "userID") \
                   .withColumnRenamed("_c1", "itemID") \
                   .withColumnRenamed("_c2", "rating")

# Convert columns to appropriate data types
training = training.withColumn("userID", training["userID"].cast(IntegerType())) \
                   .withColumn("itemID", training["itemID"].cast(IntegerType())) \
                   .withColumn("rating", training["rating"].cast(FloatType()))

# Display the first few rows of training data
training.show(5)

# Step 2: Configure ALS model
als = ALS(
    maxIter=5,
    rank=5,
    regParam=0.01,
    userCol="userID",
    itemCol="itemID",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop"
)

# Step 3: Train the ALS model
model = als.fit(training)

# Step 4: Load testing data
testing = spark.read.csv("testItem.data", header=False)
testing = testing.withColumnRenamed("_c0", "userID") \
                 .withColumnRenamed("_c1", "itemID") \
                 .withColumnRenamed("_c2", "rating")

# Convert columns to appropriate data types
testing = testing.withColumn("userID", testing["userID"].cast(IntegerType())) \
                 .withColumn("itemID", testing["itemID"].cast(IntegerType())) \
                 .withColumn("rating", testing["rating"].cast(FloatType()))

# Display the first few rows of testing data
testing.show(5)

# Step 5: Make predictions
predictions = model.transform(testing)

# Display predictions
predictions.show(5)

# Step 6.1: Save predictions as multiple files in a folder
predictions.coalesce(1).write.csv("predictions", header=True)

# Step 6.2: Save predictions to a single CSV file
predictions.toPandas().to_csv("mypredictions.csv", index=False)

# Stop the SparkSession
spark.stop()