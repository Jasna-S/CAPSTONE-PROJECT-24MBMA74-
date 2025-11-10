# Databricks notebook source
# Load data from Databricks table
branch_df = spark.read.table("workspace.default.branch_performance_data")

# Show sample records
display(branch_df)

# Check schema
branch_df.printSchema()

# Register as SQL temporary view
branch_df.createOrReplaceTempView("branch_performance")


# COMMAND ----------

print(f"Total Records: {branch_df.count()}")


# COMMAND ----------

from pyspark.sql.functions import col, sum

null_counts = branch_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in branch_df.columns
])
display(null_counts)


# COMMAND ----------

# DBTITLE 1,Revenue, Profit, and Satisfaction by Branch
from pyspark.sql import functions as F

branch_kpi = branch_df.groupBy("Branch_Name").agg(
    F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
    F.round(F.sum("Profit"), 2).alias("Total_Profit"),
    F.round(F.avg("Customer_Satisfaction"), 2).alias("Avg_Satisfaction")
).orderBy(F.desc("Total_Profit"))

display(branch_kpi)


# COMMAND ----------

# DBTITLE 1,Regional Performance Summary
region_perf = branch_df.groupBy("Region").agg(
    F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
    F.round(F.sum("Profit"), 2).alias("Total_Profit"),
    F.round(F.avg("Customer_Satisfaction"), 2).alias("Avg_Satisfaction")
).orderBy(F.desc("Total_Profit"))

display(region_perf)


# COMMAND ----------

# DBTITLE 1,Regional Performance Summary
branch_df = branch_df.withColumn(
    "Profit_Margin_Percent",
    F.round((F.col("Profit") / F.col("Revenue")) * 100, 2)
)

display(branch_df.select("Branch_Name", "Revenue", "Profit", "Profit_Margin_Percent").limit(10))


# COMMAND ----------

# DBTITLE 1,Average Profit Margin per Branch
profit_margin_summary = branch_df.groupBy("Branch_Name").agg(
    F.round(F.avg("Profit_Margin_Percent"), 2).alias("Avg_Profit_Margin")
).orderBy(F.desc("Avg_Profit_Margin"))

display(profit_margin_summary)


# COMMAND ----------

# DBTITLE 1,Product Type Analysis
product_perf = branch_df.groupBy("Product_Type").agg(
    F.round(F.avg("Revenue"), 2).alias("Avg_Revenue"),
    F.round(F.avg("Profit"), 2).alias("Avg_Profit"),
    F.round(F.avg("Customer_Satisfaction"), 2).alias("Avg_Satisfaction")
).orderBy(F.desc("Avg_Profit"))

display(product_perf)


# COMMAND ----------

# DBTITLE 1,Top 5 Branches by Profit
top_branches = branch_kpi.orderBy(F.desc("Total_Profit")).limit(5)
display(top_branches)


# COMMAND ----------

# DBTITLE 1,Bottom 5 Branches by Satisfaction
low_satisfaction = branch_kpi.orderBy(F.asc("Avg_Satisfaction")).limit(5)
display(low_satisfaction)


# COMMAND ----------

# DBTITLE 1,Monthly Trend Analysis
monthly_summary = branch_df.groupBy("Month").agg(
    F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
    F.round(F.sum("Profit"), 2).alias("Total_Profit")
).orderBy("Month")

display(monthly_summary)


# COMMAND ----------

# DBTITLE 1,Anomaly Detection – Identifying Underperforming Branches
from pyspark.sql import functions as F

# Compute mean and std deviation for Profit and Satisfaction
stats = branch_kpi.agg(
    F.mean("Total_Profit").alias("mean_profit"),
    F.stddev("Total_Profit").alias("std_profit"),
    F.mean("Avg_Satisfaction").alias("mean_satisfaction"),
    F.stddev("Avg_Satisfaction").alias("std_satisfaction")
).collect()[0]

mean_profit = stats["mean_profit"]
std_profit = stats["std_profit"]
mean_satisfaction = stats["mean_satisfaction"]
std_satisfaction = stats["std_satisfaction"]

# Add Z-score columns
branch_anomaly = branch_kpi.withColumn(
    "Profit_ZScore",
    (F.col("Total_Profit") - mean_profit) / std_profit
).withColumn(
    "Satisfaction_ZScore",
    (F.col("Avg_Satisfaction") - mean_satisfaction) / std_satisfaction
)

display(branch_anomaly)


# COMMAND ----------

# DBTITLE 1,Flag Underperforming Branches
underperformers = branch_anomaly.filter(
    (F.col("Profit_ZScore") < -1.5) | (F.col("Satisfaction_ZScore") < -1.0)
).withColumn("Performance_Status", F.lit("Underperforming"))

display(underperformers)


# COMMAND ----------

from pyspark.sql import DataFrame

all_branches_status = (
    branch_anomaly
    .withColumn(
        "Performance_Status",
        F.when((F.col("Profit_ZScore") < -1.5) | (F.col("Satisfaction_ZScore") < -1.0), "Underperforming")
         .when((F.col("Profit_ZScore") > 1.5) & (F.col("Satisfaction_ZScore") > 1.0), "Top Performer")
         .otherwise("Average")
    )
)

display(all_branches_status)
