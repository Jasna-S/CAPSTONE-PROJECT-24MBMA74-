# Databricks notebook source
# Load dataset from Databricks workspace
fraud_df = spark.read.table("workspace.default.credit_card_fraud_detection")

# Display first few rows
display(fraud_df)

# Check schema
fraud_df.printSchema()

# Register as SQL temporary view
fraud_df.createOrReplaceTempView("credit_card_data")


# COMMAND ----------

print(f"Total Transactions: {fraud_df.count()}")
print(f"Total Columns: {len(fraud_df.columns)}")


# COMMAND ----------

from pyspark.sql.functions import col, sum

null_counts = fraud_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in fraud_df.columns
])
display(null_counts)


# COMMAND ----------

class_dist = spark.sql("""
SELECT 
  `Is Fraud` AS Class,
  COUNT(*) AS Transaction_Count,
  ROUND((COUNT(*) * 100) / (SELECT COUNT(*) FROM credit_card_data), 2) AS Percentage
FROM credit_card_data
GROUP BY `Is Fraud`
ORDER BY `Is Fraud`
""")

display(class_dist)

# COMMAND ----------

amount_stats = spark.sql("""
SELECT 
  `Is Fraud` AS Class,
  ROUND(AVG(`Transaction Amount`), 2) AS Avg_Amount,
  ROUND(MAX(`Transaction Amount`), 2) AS Max_Amount,
  ROUND(MIN(`Transaction Amount`), 2) AS Min_Amount,
  ROUND(STDDEV(`Transaction Amount`), 2) AS Std_Amount
FROM credit_card_data
GROUP BY `Is Fraud`
ORDER BY `Is Fraud`
""")

display(amount_stats)

# COMMAND ----------

from pyspark.sql.functions import col, hour, sum

time_analysis = fraud_df.withColumn(
    "Hour",
    hour(col("Time of Transaction"))
)

time_summary = time_analysis.groupBy("Hour").agg(
    sum((col("Is Fraud") == 1).cast("int")).alias("Fraud_Count"),
    sum((col("Is Fraud") == 0).cast("int")).alias("Non_Fraud_Count")
).orderBy("Hour")

display(time_summary)

# COMMAND ----------

feature_compare = spark.sql("""
SELECT 
  ROUND(AVG(`Is Fraud`), 4) AS Avg_Is_Fraud,
  `Is Fraud`
FROM credit_card_data
GROUP BY `Is Fraud`
""")
display(feature_compare)

# COMMAND ----------

fraud_summary = spark.sql("""
SELECT 
  `Is Fraud` AS Fraud_Flag,
  COUNT(*) AS Transaction_Count,
  ROUND(AVG(`Transaction Amount`), 2) AS Avg_Amount,
  ROUND(SUM(`Transaction Amount`), 2) AS Total_Amount,
  ROUND(STDDEV(`Transaction Amount`), 2) AS Std_Amount
FROM credit_card_data
GROUP BY `Is Fraud`
""")

display(fraud_summary)

fraud_summary.write.mode("overwrite").saveAsTable("workspace.default.fraud_summary")

# COMMAND ----------

from pyspark.sql.functions import when, col

fraud_scored = fraud_df.withColumn(
    "Fraud_Risk_Score",
    when(
        (col("Transaction Amount") > 2000) & (col("Merchant Type") == "Online"), 0.9
    ).when(
        (col("Transaction Amount") > 1000) & (col("Merchant Type") == "Retail"), 0.7
    ).when(
        col("Transaction Amount") > 500, 0.5
    ).otherwise(0.1)
)

display(
    fraud_scored.select(
        "Transaction Amount", "Merchant Type", "Fraud_Risk_Score", "Is Fraud"
    ).limit(10)
)