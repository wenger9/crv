'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2023-12-29
-------
Objective:
- Train a model to predict the "monetary" value aspect of Customer Lifetime Value (CLV).

Key Functionality:
- Load preprocessed data suitable for monetary value prediction.
- Define and configure the monetary value prediction model.
- Train the model on the dataset.
- Evaluate model performance using appropriate metrics.
- Save the trained model via MLflow for later use in inference script/pipeline.
'''



# STEP 01 --------------------
#
#          Dependencies
#
# STEP 01 --------------------

# Import necessary packages
import pandas as pd
import numpy as np
from sklearn import metrics
import mlflow 
import mlflow.pyfunc
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, STATUS_FAIL, space_eval
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter
from gda_clv.utils.common import JobContext
from gda_clv.utils.common import JobContext
from gda_clv.utils.logger_utils import get_logger


# Parameters in JobContext initialization apply only for notebook run.
# They are ignored when script runs as a Databricks Job
job_context = JobContext("../../conf", "dev", "NOAM", "monetary_modeling")

# _logger = get_logger()
# _logger.info()

# Retrieve the sample size from the environment configuration
sample_size = job_context.env_config.sample_size


# STEP 02 --------------------
#
#          Data Preparation
#
# STEP 02 --------------------

period = '2023-05-01'

query = f"""
SELECT *
FROM clv.master_train_noam
WHERE snapshot_dt='{period}'
    AND substr(CustomerID, -2) in('1a', '1b', '1c')
    AND frequency_cal<50
"""

# Retrieve sample_size from environments/{job_context.env_config.env}.yml
# If running in dev, table query is limited to sample_size of 1000
if job_context.env_config.env == 'dev' and sample_size != -1:
    query += f" LIMIT {sample_size}"

df = spark.sql(query).toPandas()

# remove customers with no repeats in calibration period
filtered_cal = df[df.frequency_cal>0].reset_index(drop=True).copy()


# STEP 03 --------------------
#
#          Train Spend Model
#
# STEP 03 --------------------

# ctosun@estee:
# With our metrics data in place, we can now train a model to estimate the monetary value to be derived from a future transactional event.
# The model used is referred to as the Gamma-Gamma model in that it fits the gamma-distribution of an individual customer's spend against a gamma-distributed parameter that's derived from the customer population's spending distribution.
# The math is complex but the implementation is fairly straightforward using the lifetimes library.
# That said, we must first determine the best value for the L2-regularization parameter used by the model.
# For this, we will return to hyperopt.

# define search space
search_space = hp.uniform('l2', 0.0, 1.0)

# define function for model evaluation
def evaluate_model(param):
  # accesss replicated input_pd dataframe
  data = filtered_cal  
  # retrieve incoming parameters
  l2_reg = param
  # instantiate and configure the model
  model = GammaGammaFitter(penalizer_coef=l2_reg)
  # fit the model
  model.fit(data['frequency_cal'], data['monetary_value_cal'])
  
  # evaluate the model
  monetary_actual = data['monetary_value_holdout']
  monetary_predicted = model.conditional_expected_average_profit(data['frequency_holdout'], data['monetary_value_holdout'])
  try:
      rmse = metrics.mean_squared_error(monetary_actual,
                                        monetary_predicted,
                                        squared=False)
  except:
      rmse = 99
  
  # return score and status
  return {'loss': rmse, 'status': STATUS_OK}

# select optimization algorithm
algo = tpe.suggest

# perform hyperparameter tuning 
best = fmin(
  fn=evaluate_model,
  space=search_space,
  algo=algo,
  max_evals=100,
  rstate=np.random.default_rng(42) # set seed
  )
# The optimal hyperparameter settings observed during the hyperopt iterations are captured in the variable 'best'.
# Using the space_eval function, we can obtain a friendly representation of which settings performed best:

# print optimum hyperparameter settings
print(space_eval(search_space, best))

# Now that we know our best parameter settings, let's train the model with these to enable us to perform some more in-depth model evaluation:
# NOTE: Because of how search spaces are searched, different hyperopt runs may yield slightly different results.


# STEP 04 --------------------
#
#          Train Best Model
#
# STEP 04 --------------------

# ctosun@estee:
# Mlfow does not support lifetimes models out of the box.
# To use mlflow as our experiment tracking and deployment vehicle, a custom wrapper class is used which translates the standard mlflow API calls into logic which can be applied against our model.
# We've implemented a wrapper class for our monetary model which maps the mlflow predict() method to final Customer Lifetime Value prediction.
# CLV prediction is done by combining frequency and monetary models.
# For that purpose we are providing our previous model as a model artifact.

frequency_model = mlflow.pyfunc.load_model(f"models:/clv_freq_best/{'latest'}")

# create wrapper for spend model
class _clvModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, spend_model):
        self.spend_model = spend_model

    def load_context(self, context):
      # load base model fitter from lifetimes library
      from lifetimes.fitters.base_fitter import BaseFitter
      
      # instantiate lifetimes_model
      self.lifetimes_model = BaseFitter()
      
      # load lifetimes_model from mlflow
      self.lifetimes_model.load_model(context.artifacts['lifetimes_model'])

    def predict(self, context, dataframe):    
      # access input series
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]      
      monetary_value = dataframe.iloc[:,3]
      months = int(dataframe.iloc[0,4])
      discount_rate = float(dataframe.iloc[0,5])      
      
      # make CLV prediction
      results = pd.DataFrame(
          self.spend_model.customer_lifetime_value(
            self.lifetimes_model, #the model to use to predict the number of future transactions
            frequency, recency, T, monetary_value, time=months, discount_rate=discount_rate),
          columns=['clv'])
      
      return results[['clv']]

input_schema = Schema(
    [
        ColSpec("double", "frequency"),
        ColSpec("double", "recency"),
        ColSpec("double", "T"),
        ColSpec("double", "monetary_value"),
        ColSpec("double", "time"),
        ColSpec("double", "discount_rate")
    ]
)
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# add lifetimes to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'].append('lifetimes')

# # Set experiment
# experiment_name = "/GDA/ML_Projects/clv-test-new"
# mlflow.set_experiment(experiment_name)

# Set MLFlow experiment
experiment_name = job_context.get_experiment_name()
mlflow.set_experiment(experiment_name=experiment_name)


# save temp copy of pickled/serialized lifetimes model
mlflow_lifetimes_logged_model = mlflow.pyfunc.load_model(f"models:/clv_freq_best/{'latest'}")
model_container = mlflow_lifetimes_logged_model.unwrap_python_model()
lifetimes_model = model_container.lifetimes_model

lifetimes_model_path = '/dbfs/tmp/lifetimes_model.pkl'

# delete any prior copies that may exist
try:
  dbutils.fs.rm(lifetimes_model_path)
except:
  pass

# save the model to the temp location
lifetimes_model.save_model(lifetimes_model_path)

# get hyperparameter settings
param = space_eval(search_space, best)
l2_reg = param

# instantiate and configure model
model = GammaGammaFitter(penalizer_coef=l2_reg)

with mlflow.start_run(run_name=f'monetary_best') as run:
    # Create and train model
    model.fit(filtered_cal['frequency_cal'], filtered_cal['monetary_value_cal'])
    # get predicted monetary during holdout period
    monetary_holdout_predicted = model.conditional_expected_average_profit(filtered_cal['frequency_holdout'],
                                                                           filtered_cal['monetary_value_holdout'])

    # get actual monetary during holdout period
    monetary_holdout_actual = filtered_cal['monetary_value_holdout']

    # Log parameters
    mlflow.log_param("l2_penalizer_coef", l2_reg)

    # Log metrics
    mae = metrics.mean_absolute_error(monetary_holdout_actual, monetary_holdout_predicted)
    mse = metrics.mean_squared_error(monetary_holdout_actual, monetary_holdout_predicted, squared=True)
    rmse = metrics.mean_squared_error(monetary_holdout_actual, monetary_holdout_predicted, squared=False)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    
    # identify lifetime model as an artifact associated with the spend model
    artifacts = {'lifetimes_model': lifetimes_model_path}

    # Log model
    mlflow.pyfunc.log_model(
    'model', 
    python_model=_clvModelWrapper(model),
    signature=signature,
    conda_env=conda_env,
    artifacts=artifacts
    )
    
    # Model uri later registering
    base_uri = run.info.artifact_uri

# Register model to ML flow to make predictions on the data later  
model_full_name = f"clv_monetary_best"
model_uri = base_uri+"/model"
new_model_version = mlflow.register_model(model_uri, model_full_name,  await_registration_for=1200)


# STEP 05 --------------------
#
#          Model Evaluation
#
# STEP 05 --------------------

# score the model
monetary_holdout_actual = filtered_cal['monetary_value_holdout']
monetary_holdout_predicted = model.conditional_expected_average_profit(
    filtered_cal['frequency_holdout'], 
    filtered_cal['monetary_value_holdout'])
rmse = metrics.mean_squared_error(monetary_holdout_actual, monetary_holdout_predicted, squared=False)
mae = metrics.mean_absolute_error(monetary_holdout_actual, monetary_holdout_predicted)
print('RMSE: {0}'.format(rmse))
print('MAE: {0}'.format(mae))

# Check that error is below 20$
assert(rmse<20)

# calculate MAPE for spenders
perf_df = pd.DataFrame({'actual': monetary_holdout_actual, 'pred': monetary_holdout_predicted})
metrics.mean_absolute_percentage_error(perf_df.query("actual>0").actual, perf_df.query("actual>0").pred)


perf_df_dsp = perf_df.query("actual>0 and actual<=250")

r2 = metrics.r2_score(perf_df_dsp.actual, perf_df_dsp.pred)
print(r2)
assert(r2 > 0.70)


# STEP 06 --------------------
#
#          Final Training
#
# STEP 06 --------------------

# After verifying the model is acceptable, we can train it with full dataset

# read full dataset
## sampling based on last 2 characters of consumerid
df_full = spark.sql(f"""
select  * 
from    clv.master_full_noam
where   snapshot_dt='{period}'  
    and substr(CustomerID, -2) in('1a', '1b', '1c')  
    and frequency<50
""").toPandas()

# remove customers with no repeats
filtered = df_full[df_full.frequency>0].reset_index(drop=True).copy()

with mlflow.start_run(run_name=f'monetary_final') as run:
    # Create and train model
    model.fit(filtered['frequency'], filtered['monetary_value'])
    # Log parameters
    mlflow.log_param("l2_penalizer_coef", l2_reg)
    # Log model
    mlflow.pyfunc.log_model(
    'model', 
    python_model=_clvModelWrapper(model), 
    signature=signature,
    conda_env=conda_env,
    artifacts=artifacts
    )
    # Model uri later registering
    base_uri = run.info.artifact_uri

# Register model to ML flow to make predictions on the data later  
model_full_name = f"clv_monetary_final"
model_uri = base_uri+"/model"
new_model_version = mlflow.register_model(model_uri, model_full_name,  await_registration_for=1200)
