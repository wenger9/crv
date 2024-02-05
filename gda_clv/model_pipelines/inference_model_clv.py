'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2023-12-30
-------
Objective:
- Operationalize the CLV model inference process for incoming data (delta).
- Handle data loading, preprocessing, prediction generation, and storage.

Key Functionality:
- load new data that needs to be scored (inference input).
- preprocess data if necessary (alignment with project_model_clv.py).
- utilize UDFs defined in project_model_clv.py or directly use the models to generate predictions.
- manage the storage of prediction results.
- Note: Notebook #4 from ctosun@estee combined model-application logic and operational inference.
        That notebook was modified and split into two scripts, with distinct objectives.
        This is the 2nd of the two scripts, handling the operational workflow of model inference.
'''

# # --------------------
# #  A1. Import necessary packages
# # --------------------
import mlflow

from mlflow import pyfunc
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from datetime import datetime
from gda_clv.utils.common import JobContext
from gda_clv.utils.pipelines.save_to_blob import save_inference_to_blob_storage, save_meta_to_blob_storage
from gda_clv.utils.logger_utils import get_logger
# from gda_clv.utils.verify_inference import check_model_output_in_blob

# # --------------------
# #  B1. Instantiate JobContext()
# #     - Parametersa in JobContext initialization apply only for notebook runs. They are ignored when a script runs as a Databricks Job
# #  B2. Instantiate get_logger()
# #     - Log runtime events
# # --------------------
job_context = JobContext("../../conf", "dev", "NOAM", "inference_modeling")
_logger = get_logger()
_logger.info("START Job: CLV Inference | inference_model_clv.py | Running...")



# # --------------------
# #  C1. Load inference input
# #  C2. Create timestamp to track date of inference
# # --------------------
inference_input_df = job_context.spark.table("clv.master_full_noam")
inference_date = datetime.now().strftime("%Y-%m-%d")


# Check if UDF is registered
def is_udf_registered(udf_name):
    return udf_name in spark.catalog.listFunctions()

# Register UDFs if they are not available
def register_udfs_if_needed():
    if not is_udf_registered("clv"):
        clv_udf = mlflow.pyfunc.spark_udf(
            spark, 
            f"models:/clv_monetary_final/{'latest'}", 
            result_type=DoubleType()
        )
        _ = spark.udf.register('clv', clv_udf)
        _logger.info("UDF 'clv' registered.")

    if not is_udf_registered("probability_alive"):
        probability_alive_udf = mlflow.pyfunc.spark_udf(
            spark, 
            f"models:/clv_freq_final/{'latest'}", 
            result_type=ArrayType(FloatType())      
        )
        _ = spark.udf.register('probability_alive', probability_alive_udf)
        _logger.info("UDF 'probability_alive' registered.")

# Check and register UDFs if needed
register_udfs_if_needed()

spark.sql("""
--create or replace table clv.results_noam (
create table if not exists clv.results_noam (
    snapshot_dt date,
    CustomerID	string,
    frequency	float,
    recency	float,
    T	float,
    monetary_value	double,
    prob_alive float,
    purch_1y float,
    purch_2y float,
    clv_1y double,
    clv_3y double,
    clv_1y_decile int,
    prob_alive_decile int
    )
    partitioned by (snapshot_dt);
""")


# # --------------------
# #  D1. Perform batch scoring
# # --------------------
# Perform batch scoring
period = '2023-09-01' # Parameterize and implement via argparse
annual_rate = 0.09
monthly_rate = round((1 + annual_rate)**(1/12) - 1, 6)

# # ----------------------
# #  E1. Create result dataframe
# # ----------------------
result_df = spark.sql(f"""
with 
x as (
SELECT    snapshot_dt,
          CustomerID,
          probability_alive(cast(frequency as bigint), cast(recency as bigint), cast(T as bigint)) as prediction,
          clv(frequency, recency, T, monetary_value, 12, {monthly_rate}) as clv_1y,          
          clv(frequency, recency, T, monetary_value, 36, {monthly_rate}) as clv_3y,
          frequency, recency, T, monetary_value
FROM      clv.master_full_noam
WHERE     snapshot_dt = '{period}'
),
dt as (
SELECT    snapshot_dt,
          x.CustomerID, 
          frequency, recency, T, monetary_value,
          x.prediction[0] as prob_alive, 
          x.prediction[2] as purch_1y,
          x.prediction[3] as purch_2y,
          case when clv_1y<0 then 0 else clv_1y end as clv_1y, 
          case when clv_3y<0 then 0 else clv_3y end as clv_3y
FROM x
)
SELECT    dt.*,
          ntile(10) over(order by clv_1y desc) as clv_1y_decile,
          ntile(10) over(order by prob_alive desc) as prob_alive_decile
FROM      dt
""")

# # ---------------
# #  F1. Write to hive_metastore as clv.results_noam
# # ---------------
(result_df
    .write.mode("overwrite")
    .option("replaceWhere", f"snapshot_dt = '{period}'")
    .saveAsTable('clv.results_noam')
)

# # ---------------
# #  G1. Save model inference to blob storage
# #  G2. Save model metadata to blob storage
# # ---------------
save_location = job_context.get_save_location()
table_name = job_context.base_config.inference_output_table
write_dt = datetime.now()

save_inference_to_blob_storage(
    dataframe=result_df,
    dbutils=job_context.dbutils,
    save_location=save_location,
    table_name=table_name,
    write_dt=write_dt
)

# save_meta_to_blob_storage(
#     save_location=save_location,
#     registry_model_name=job_context.model_config.registry_model_name,
#     model_stage=job_context.env_config.model_stage,
#     dbutils=job_context.dbutils,
#     inference_input_dt=inference_date,
#     write_dt=write_dt
# )

# blob_name = job_context.base_config.inference_output_table + "_" + dt_string + '.csv'
# container_name = "ds-gda"

# # Check if model output exists in Blob Storage
# check_model_output_in_blob(
#     container_name=container_name
#     blob_name=blob_name
# )


# _logger.info(f"CLV Model Inference results for {period} saved to hive_metastore and blob storage.")
_logger.info("END Job: CLV Inference | inference_model_clv.py | Successful")