'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2023-12-29
-------
Objective:
- Apply trained CLV models to generate customer lifetime value predictions.
- Utilize models to understand customer engagement and future purchase behavior.

Key Functionality:
- Register user-defined functions (UDFs) to be used for model inference.
- Define and execute SQL queries to apply UDFs on data.
- Generate predictions, i.e. 1) probability of being alive, 2) expected purchases and 3) CLV.
- Note: Notebook #4 from ctosun@estee combined model-application logic and operational inference.
        That notebook was modified and split into two scripts, with distinct objectives.
        This is the 1st of the two scripts, handling model-application logic.
'''

'''
NOTE
-------
Author | ctosun@estee.com
Date   | 2023-09-15
-------
There are numerous ways we might make use of the trained BTYD model. 
We may want to understand the probability a customer is still engaged.
We may want to predict the number of purchases expected from the customer over some number of days.
In order to generate these predictions, all we need is our trained model and values of frequency, recency and age (T).
'''

# # --------------------
# #  A1. Import necessary packages_
# # --------------------
import mlflow

from mlflow import pyfunc
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from gda_clv.utils.common import JobContext
from gda_clv.utils.logger_utils import get_logger
# from gda_clv.utils.verify_udf import verify_and_save_udf_check



# # --------------------
# #  B1. Instantiate JobContext()
# #     - Parameters in JobContext initialization apply only for notebook runs. They are ignored when a script runs as a Databricks Job
# #  B2. Instantiate get_logger()
# #     - Log runtime events
# # --------------------
job_context = JobContext("../../conf", "dev", "NOAM", "clv_projection")
_logger = get_logger()
_logger.info("START Job: CLV Projection | project_model_clv.py | Running...")

# # --------------------
# #  C1. Create and register user-defined functions (UDFs)
# #     - UDF 1 is named clv_udf
# #     - Define the schema of values returned by clv_udf
# #     - Define clv_udf based on the latest clv_monetary_final model registered in MLflow
# #     - Register clv_udf for use in SQL
# # --------------------
# result_schema = DoubleType()

clv_udf = mlflow.pyfunc.spark_udf(
  spark, 
  f"models:/clv_monetary_final/{'latest'}", 
  result_type=DoubleType(),
  env_manager="conda"
  )

_ = spark.udf.register('clv', clv_udf)

# # --------------------
# #  D1. Create and register user-defined functions (UDFs)
# #     - UDF 2 is named probability_alive_udf
# #     - Define the schema of values returned by probability_alive_udf
# #     - Define probability_alive_udf based on the latest clv_freq_final model registered in MLflow
# #     - Register probability_alive_udf for use in SQL
# # --------------------
# result_schema = ArrayType(FloatType())

probability_alive_udf = mlflow.pyfunc.spark_udf(
  spark, 
  f"models:/clv_freq_final/{'latest'}", 
  result_type=ArrayType(FloatType()),
  env_manager="conda"  
  )

_ = spark.udf.register('probability_alive', probability_alive_udf)

# # verify UDFs have been registered and save the list to Hive
# verify_and_save_udf_check(spark)

# _logger.info("UDFs verified and list of functions saved to Hive.")
_logger.info("END Job: CLV Projection | project_model_clv.py | Successful")
