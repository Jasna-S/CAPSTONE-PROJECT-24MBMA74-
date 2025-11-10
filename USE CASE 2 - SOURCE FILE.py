# Databricks notebook source
# Load as Delta table (since workspace.default.use_case_2 is a managed table)
loan_df = spark.table("workspace.default.use_case_2")

print("✅ Dataset loaded successfully!")
print(f"Total Rows: {loan_df.count()}")
print(f"Total Columns: {len(loan_df.columns)}")

display(loan_df.limit(5))


# COMMAND ----------

loan_df.printSchema()

# Show summary statistics for numeric columns
display(loan_df.describe())


# COMMAND ----------

from pyspark.sql.functions import col, sum

missing = loan_df.select(
    *[sum(col(c).isNull().cast("int")).alias(c) for c in loan_df.columns]
)
display(missing)


# COMMAND ----------

# DBTITLE 1,Total loans grouped by loan status
display(
    loan_df.groupBy("loan_status")
    .count()
    .withColumnRenamed("count", "Total Loans")
)


# COMMAND ----------

# DBTITLE 1,Average metrics grouped by loan status
from pyspark.sql.functions import avg

avg_df = (
    loan_df.groupBy("loan_status")
    .agg(
        avg("loan_amount").alias("Avg_Loan_Amount"),
        avg("annual_income").alias("Avg_Income"),
        avg("credit_score").alias("Avg_Credit_Score")
    )
)
display(avg_df)


# COMMAND ----------

# Home ownership vs default
display(loan_df.groupBy("home_ownership", "loan_status").count().orderBy("home_ownership"))

# Loan purpose vs default
display(loan_df.groupBy("purpose", "loan_status").count().orderBy("purpose"))


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

numeric_cols = [
    "loan_amount", "term", "interest_rate", "annual_income",
    "credit_score", "employment_length", "dti", "delinq_2yrs",
    "revol_util", "total_acc", "loan_status"
]

# Combine numeric features
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
vector_df = assembler.transform(loan_df).select("features")

# Correlation matrix
corr = Correlation.corr(vector_df, "features", "pearson").head()[0].toArray()

import pandas as pd
import numpy as np

corr_matrix = pd.DataFrame(corr, index=numeric_cols, columns=numeric_cols)
display(corr_matrix)


# COMMAND ----------

from pyspark.sql.functions import when, col

# Create income categories
loan_df_cat = loan_df.withColumn(
    "loan_bracket",
    when(col("loan_amount") < 10000, "Low") \
    .when((col("loan_amount") >= 10000) & (col("loan_amount") < 30000), "Medium") \
    .otherwise("High")
)

display(
    loan_df_cat.groupBy("loan_bracket", "loan_status").count().orderBy("loan_bracket")
)


# COMMAND ----------

loan_df_inc = loan_df.withColumn(
    "income_bracket",
    when(col("annual_income") < 40000, "Low Income")
    .when((col("annual_income") >= 40000) & (col("annual_income") < 80000), "Mid Income")
    .otherwise("High Income")
)

display(
    loan_df_inc.groupBy("income_bracket", "loan_status").count().orderBy("income_bracket")
)


# COMMAND ----------

display(
    loan_df.groupBy("loan_status")
    .agg(
        avg("interest_rate").alias("Avg_Interest_Rate"),
        avg("credit_score").alias("Avg_Credit_Score"),
        avg("dti").alias("Avg_DTI"),
        avg("revol_util").alias("Avg_Revolving_Utilization")
    )
)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Define categorical and numeric columns
categorical_cols = ["home_ownership", "purpose"]
numeric_cols = [
    "loan_amount", "term", "interest_rate", "annual_income",
    "credit_score", "employment_length", "dti", "delinq_2yrs",
    "revol_util", "total_acc"
]

# Index and encode categorical variables
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in categorical_cols]
encoders = [OneHotEncoder(inputCols=[c + "_index"], outputCols=[c + "_vec"]) for c in categorical_cols]

# Combine all features into a single vector
assembler = VectorAssembler(
    inputCols=numeric_cols + [c + "_vec" for c in categorical_cols],
    outputCol="features"
)

# Create a pipeline for preprocessing
pipeline = Pipeline(stages=indexers + encoders + [assembler])
processed_df = pipeline.fit(loan_df).transform(loan_df)

# Check the transformed dataset
display(processed_df.select("loan_status", "features").limit(5))


# COMMAND ----------

train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training Records: {train_df.count()} | Test Records: {test_df.count()}")


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="loan_status", featuresCol="features", maxIter=20)
lr_model = lr.fit(train_df)

lr_predictions = lr_model.transform(test_df)
display(lr_predictions.select("loan_status", "prediction", "probability"))


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# AUC
evaluator = BinaryClassificationEvaluator(labelCol="loan_status", rawPredictionCol="rawPrediction")
auc_lr = evaluator.evaluate(lr_predictions)
print(f"✅ Logistic Regression AUC: {auc_lr:.3f}")

# Accuracy, Precision, Recall
accuracy_eval = MulticlassClassificationEvaluator(labelCol="loan_status", metricName="accuracy")
precision_eval = MulticlassClassificationEvaluator(labelCol="loan_status", metricName="weightedPrecision")
recall_eval = MulticlassClassificationEvaluator(labelCol="loan_status", metricName="weightedRecall")

print(f"Accuracy: {accuracy_eval.evaluate(lr_predictions):.3f}")
print(f"Precision: {precision_eval.evaluate(lr_predictions):.3f}")
print(f"Recall: {recall_eval.evaluate(lr_predictions):.3f}")


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="loan_status", featuresCol="features", numTrees=50, maxDepth=8, seed=42)
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# Evaluate Random Forest
auc_rf = evaluator.evaluate(rf_predictions)
print(f"✅ Random Forest AUC: {auc_rf:.3f}")

print(f"Accuracy: {accuracy_eval.evaluate(rf_predictions):.3f}")
print(f"Precision: {precision_eval.evaluate(rf_predictions):.3f}")
print(f"Recall: {recall_eval.evaluate(rf_predictions):.3f}")


# COMMAND ----------

import pandas as pd

feature_imp = list(zip(assembler.getInputCols(), rf_model.featureImportances.toArray()))
feature_imp_df = pd.DataFrame(feature_imp, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
display(feature_imp_df)


# COMMAND ----------

# Confusion Matrix for Random Forest
confusion_rf = (
    rf_predictions.groupBy("loan_status", "prediction")
    .count()
    .orderBy("loan_status", "prediction")
)

print("✅ Confusion Matrix (Random Forest)")
display(confusion_rf)
