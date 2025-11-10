# Databricks notebook source
# Load dataset from Databricks
claims_df = spark.read.table("workspace.default.insurance_claim_data")

# Display first few rows
display(claims_df)

# Print schema
claims_df.printSchema()

# Register as temporary SQL view for SQL-based queries
claims_df.createOrReplaceTempView("insurance_claims")


# COMMAND ----------

print(f"Total Records: {claims_df.count()}")
print(f"Total Columns: {len(claims_df.columns)}")


# COMMAND ----------

from pyspark.sql.functions import col, sum

null_counts = claims_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in claims_df.columns
])
display(null_counts)


# COMMAND ----------

from pyspark.sql import functions as F

status_dist = claims_df.groupBy("Claim_Status").agg(
    F.count("*").alias("Claim_Count")
)
display(status_dist)


# COMMAND ----------

amount_stats = claims_df.groupBy("Claim_Status").agg(
    F.round(F.avg("Claim_Amount"), 2).alias("Avg_Amount"),
    F.round(F.max("Claim_Amount"), 2).alias("Max_Amount"),
    F.round(F.min("Claim_Amount"), 2).alias("Min_Amount"),
    F.round(F.stddev("Claim_Amount"), 2).alias("Std_Amount")
)
display(amount_stats)


# COMMAND ----------

claims_df = claims_df.withColumn(
    "Age_Group",
    F.when(F.col("Age") < 30, "Below 30")
     .when((F.col("Age") >= 30) & (F.col("Age") < 50), "30-49")
     .when((F.col("Age") >= 50) & (F.col("Age") < 70), "50-69")
     .otherwise("70+")
)

age_group_summary = claims_df.groupBy("Age_Group").agg(
    F.round(F.avg("Claim_Amount"), 2).alias("Avg_Claim_Amount"),
    F.count("*").alias("Total_Claims")
).orderBy("Age_Group")

display(age_group_summary)


# COMMAND ----------

region_summary = claims_df.groupBy("Region").agg(
    F.count("*").alias("Claim_Count"),
    F.round(F.avg("Claim_Amount"), 2).alias("Avg_Claim_Amount")
).orderBy(F.desc("Claim_Count"))

display(region_summary)


# COMMAND ----------

claims_df = claims_df.withColumn("Month", F.date_format(F.col("Claim_Date"), "yyyy-MM"))

monthly_trend = claims_df.groupBy("Month").agg(
    F.count("*").alias("Total_Claims"),
    F.round(F.avg("Claim_Amount"), 2).alias("Avg_Claim_Amount")
).orderBy("Month")

display(monthly_trend)


# COMMAND ----------

import pyspark.sql.functions as F

claims_df = spark.table("workspace.default.insurance_claim_data")

claims_scored = claims_df.withColumn(
    "Claim_Risk_Score",
    F.when(
        (F.col("Claim_Amount") > 150000) & (F.col("Claim_Status") == "Rejected"), 0.9
    ).when(
        (F.col("Claim_Amount") > 100000) & (F.col("Claim_Status") == "Pending"), 0.7
    ).when(
        F.col("Claim_Amount") > 50000, 0.5
    ).otherwise(0.2)
)

display(
    claims_scored.select(
        "Claim_ID",
        "Claim_Amount",
        "Claim_Status",
        "Claim_Risk_Score"
    ).limit(10)
)

# COMMAND ----------

summary = claims_df.groupBy("Claim_Status").agg(
    F.count("*").alias("Claim_Count"),
    F.round(F.avg("Claim_Amount"), 2).alias("Avg_Claim_Amount"),
    F.round(F.sum("Claim_Amount"), 2).alias("Total_Claim_Value")
)
display(summary)

# Save for reuse or reporting
summary.write.mode("overwrite").saveAsTable("workspace.default.insurance_claim_summary")
