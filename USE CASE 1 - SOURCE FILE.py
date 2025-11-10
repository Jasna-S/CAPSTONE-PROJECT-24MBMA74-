# Databricks notebook source

# Loading table into a Spark DataFrame
df = spark.table("workspace.default.customer_transaction_analysis")

# Show a few rows
df.show(5)

# Print schema
df.printSchema()

# Total records and columns
print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Count nulls for each column
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()


# COMMAND ----------

df.describe(["amount"]).show()


# COMMAND ----------

df.selectExpr(
    "count(distinct transaction_id) as unique_txns",
    "count(distinct customer_id) as unique_customers",
    "count(distinct merchant_category) as merchant_categories",
    "count(distinct channel) as channels",
    "count(distinct branch_id) as branches"
).show()


# COMMAND ----------

df.groupBy("merchant_category") \
  .count() \
  .orderBy("count", ascending=False) \
  .show(10)

# COMMAND ----------

from pyspark.sql.functions import sum as _sum, round

df.groupBy("channel") \
  .agg(
      round(_sum("amount"),2).alias("total_spend"),
      round((_sum("amount") / df.agg(_sum("amount")).collect()[0][0]) * 100, 2).alias("spend_share_pct")
  ) \
  .orderBy("total_spend", ascending=False) \
  .show()

# COMMAND ----------

df.groupBy("is_international") \
  .agg(round(_sum("amount"),2).alias("total_spend")) \
  .orderBy("is_international") \
  .show()

# COMMAND ----------

from pyspark.sql.functions import avg

df.groupBy("merchant_category") \
  .agg(round(avg("amount"),2).alias("avg_spend")) \
  .orderBy("avg_spend", ascending=False) \
  .show(10)

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, col


# COMMAND ----------

# Use to_timestamp with correct format (dd-MM-yyyy HH:mm)
df = df.withColumn(
    "txn_timestamp",
    to_timestamp(col("txn_timestamp"), "dd-MM-yyyy HH:mm")
)

# COMMAND ----------

from pyspark.sql.functions import expr

df = df.withColumn(
    "txn_timestamp",
    expr("try_to_timestamp(txn_timestamp, 'dd-MM-yyyy HH:mm')")
)

# COMMAND ----------

df.select("txn_timestamp").show(5, truncate=False)
df.printSchema()


# COMMAND ----------

from pyspark.sql.functions import hour, dayofweek, avg, round, count, sum as _sum

# Hourly analysis
df.groupBy(hour("txn_timestamp").alias("txn_hour")) \
  .agg(
      round(avg("amount"), 2).alias("avg_amount"),
      count("*").alias("txn_count")
  ) \
  .orderBy("txn_hour") \
  .show()

# Day of week analysis
df.groupBy(dayofweek("txn_timestamp").alias("day_of_week")) \
  .agg(
      round(_sum("amount"), 2).alias("total_spend")
  ) \
  .orderBy("day_of_week") \
  .show()


# COMMAND ----------

df.groupBy("branch_id") \
  .agg(
      round(_sum("amount"),2).alias("total_spend"),
      count("*").alias("txn_count")
  ) \
  .orderBy("total_spend", ascending=False) \
  .show(10)


# COMMAND ----------

df.stat.corr("amount", "is_international")


# COMMAND ----------

# DBTITLE 1,Merchant Category vs Total Spend
merchant_spend = (
    df.groupBy("merchant_category")
      .agg(round(sum("amount"), 2).alias("total_spend"))
      .orderBy(col("total_spend").desc())
)

display(merchant_spend)

# COMMAND ----------

# DBTITLE 1,Channel-wise Spend Share
channel_spend = (
    df.groupBy("channel")
      .agg(round(sum("amount"), 2).alias("total_spend"))
      .orderBy(col("total_spend").desc())
)

display(channel_spend)

# COMMAND ----------

# DBTITLE 1,Transaction Count by Hour of Day
hourly_txns = (
    df.groupBy(hour("txn_timestamp").alias("txn_hour"))
      .agg(count("*").alias("txn_count"))
      .orderBy("txn_hour")
)

display(hourly_txns)