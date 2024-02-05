'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2023-12-28
-------
Objective:
- Train a model to predict the "frequency" aspect of Customer Lifetime Value (CLV).

Key Functionality:
- Load preprocessed data suitable for frequency prediction.
- Define and configure the frequency prediction model.
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
from hyperopt import hp, fmin, tpe, STATUS_OK, space_eval
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.fitters.modified_beta_geo_fitter import ModifiedBetaGeoFitter
from gda_clv.utils.common import JobContext
from gda_clv.utils.logger_utils import get_logger
# from gda_clv.utils.pipelines.model_utils import get_estimator_sizes


# Parameters in JobContext initialization apply only for notebook runs.
# They are ignored when a script runs as a Databricks Job
job_context = JobContext("../../conf", "dev", "NOAM", "frequency_modeling")

_logger = get_logger()
_logger.info("START: Frequency Modeling | train_model_frequency.py")

# Retrieve the sample size from the environment configuration
sample_size = job_context.env_config.sample_size

# STEP 02 --------------------
#
#          Data Preparation
#
# STEP 02 --------------------

period = '2023-05-01'

# Isolate SQL query in order to modify it based on environment configurations
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

# # Ensure sufficient data
# assert(df.shape[0] > 3e5)


# Retrieve the environment from the job context
environment = job_context.env_config.env

# Ensure sufficient data only if not in dev environment
if environment != 'dev':
    assert(df.shape[0] > 3e5)


# Keep customers with no repeats in calibration period
filtered_cal = df.copy()

# Create wrapper for lifetimes model
class _lifetimesModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, lifetimes_model):
        self.lifetimes_model = lifetimes_model

    def predict(self, context, dataframe):
      
      # access input series
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]
      
      # calculate probability currently alive
      results = pd.DataFrame( 
                  self.lifetimes_model.conditional_probability_alive(frequency, recency, T),
                  columns=['alive']
                  )
      # calculate expected purchases for provided time period
      results['purch_30day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(30, frequency, recency, T)
      results['purch_365day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(365, frequency, recency, T)
      results['purch_730day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(730, frequency, recency, T)
      
      return results[['alive', 'purch_30day', 'purch_365day', 'purch_730day']]

input_schema = Schema([ColSpec("long", "frequency"),
                       ColSpec("long", "recency"),
                       ColSpec("long", "tenure")])
output_schema = Schema([ColSpec("double", "alive"),
                        ColSpec("double", "purch_30day"),
                        ColSpec("double", "purch_365day"),
                        ColSpec("double", "purch_730day")
                        ])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)    

# Add lifetimes to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'].append('lifetimes')

# Set MLFlow experiment
experiment_name = job_context.get_experiment_name()
mlflow.set_experiment(experiment_name=experiment_name)


with mlflow.start_run(run_name=f'freq_pareto_simple') as run:
    # Create and train model
    l2_penalizer_coef = 0.01
    model = ParetoNBDFitter(penalizer_coef=l2_penalizer_coef)
    model.fit(filtered_cal['frequency_cal'], filtered_cal['recency_cal'], filtered_cal['T_cal'])

    # get predicted frequency during holdout period
    frequency_holdout_predicted = model.predict(filtered_cal['duration_holdout'].to_numpy(),
                                                filtered_cal['frequency_cal'].to_numpy(), 
                                                filtered_cal['recency_cal'].to_numpy(), 
                                                filtered_cal['T_cal'].to_numpy())

    # get actual frequency during holdout period
    frequency_holdout_actual = filtered_cal['frequency_holdout']

    # Log parameters
    mlflow.log_param("l2_penalizer_coef", l2_penalizer_coef)

    # Log metrics
    mae = metrics.mean_absolute_error(frequency_holdout_actual, frequency_holdout_predicted)
    mse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=True)
    rmse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=False)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.pyfunc.log_model(
    'model', 
    python_model=_lifetimesModelWrapper(model), 
    signature=signature,
    conda_env=conda_env
    )
    
    # Model uri later registering
    base_uri = run.info.artifact_uri


# Register model to ML flow to make predictions on the data later  
model_full_name = f"clv_freq_pareto_simple"
model_uri = base_uri+"/model"
_logger.info(f"Register MLflow Model by uri path: {model_uri}")
new_model_version = mlflow.register_model(model_uri, model_full_name,  await_registration_for=1200)

# With our model now fit, let's make some predictions for the holdout period. We'll grab the actuals for that same period to enable comparison in a subsequent step.

# get predicted frequency during holdout period
frequency_holdout_predicted = model.predict(filtered_cal['duration_holdout'].to_numpy(), 
                                            filtered_cal['frequency_cal'].to_numpy(), 
                                            filtered_cal['recency_cal'].to_numpy(), 
                                            filtered_cal['T_cal'].to_numpy())

# get actual frequency during holdout period
frequency_holdout_actual = filtered_cal['frequency_holdout']

print('mae:', metrics.mean_absolute_error(frequency_holdout_actual, frequency_holdout_predicted))
print('rmse:', metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=False))

# ctosun@estee:
# While the internals of the Pareto/NBD model may be quite complex. In a nutshell, the model calculates a double integral of two curves, one which describes the frequency of customer purchases within a population and another which describes customer survivorship following a prior purchase event. All of the calculation logic is thankfully hidden behind a simple method call.
# As simple as training a model may be, we have two models that we could use here: the Pareto/NBD model and the BG/NBD model. The BG/NBD model simplifies the math involved in calculating customer lifetime and is the model that popularized the BTYD approach. Both models work off the same customer features and employ the same constraints. (The primary difference between the two models is that the BG/NBD model maps the survivorship curve to a beta-geometric distribution instead of a Pareto distribution.) To achieve the best fit possible, it is worthwhile to compare the results of both models with our dataset.
# Each model leverages an L2-norm regularization parameter which we've arbitrarily set to 0 in the previous training cycle. In addition to exploring which model works best, we should consider which value (between 0 and 1) works best for this parameter. This gives us a pretty broad search space to explore with some hyperparameter tuning.
# To assist us with this, we will make use of hyperopt. Hyperopt allows us to parallelize the training and evaluation of models against a hyperparameter search space. This can be done leveraging the multiprocessor resources of a single machine or across the broader resources provided by a Spark cluster. With each model iteration, a loss function is calculated. Using various optimization algorithms, hyperopt navigates the search space to locate the best available combination of parameter settings to minimize the value returned by the loss function.
# To make use of hyperopt, lets define our search space and re-write our model training and evaluation logic to provide a single function call which will return a loss function measure:

# define search space
search_space = hp.uniform('l2', 0.0, 1.0)

# define function for model evaluation
def evaluate_model(param):
  # accesss replicated input_pd dataframe
  # data = inputs.value  
  data = filtered_cal
  # retrieve incoming parameters
  l2_reg = param

  # instantiate and configure the model
  model = ModifiedBetaGeoFitter(penalizer_coef=l2_reg)
  
  # fit the model
  model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
  
  # evaluate the model
  frequency_holdout_actual = data['frequency_holdout']
  frequency_holdout_predicted = model.predict(data['duration_holdout'].to_numpy(), 
                                              data['frequency_cal'].to_numpy(), 
                                              data['recency_cal'].to_numpy(), 
                                              data['T_cal'].to_numpy())
  try:
      rmse = metrics.mean_squared_error(frequency_holdout_actual,
                                        frequency_holdout_predicted,
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

# ctosun@estee:
# The optimal hyperparameter settings observed during the hyperopt iterations are captured in the variable named as best.
# Using the space_eval function, we can obtain a friendly representation of which settings performed best:

# print optimum hyperparameter settings
print(space_eval(search_space, best))

# ctosun@estee:
# Now that we know our best parameter settings, let's train the model with these to enable us to perform some more in-depth model evaluation:
# Note: Because of how search spaces are searched, different hyperopt runs may yield slightly different results.


# STEP 05 --------------------
#
#          Train best model
#
# STEP 05 --------------------

# get hyperparameter settings
l2_reg = space_eval(search_space, best)

# instantiate and configure model
model = ModifiedBetaGeoFitter(penalizer_coef=l2_reg)

with mlflow.start_run(run_name=f'freq_best') as run:
    # Create and train model
    model.fit(filtered_cal['frequency_cal'], filtered_cal['recency_cal'], filtered_cal['T_cal'])
    # get predicted frequency during holdout period
    frequency_holdout_predicted = model.predict(filtered_cal['duration_holdout'].to_numpy(),
                                                filtered_cal['frequency_cal'].to_numpy(), 
                                                filtered_cal['recency_cal'].to_numpy(), 
                                                filtered_cal['T_cal'].to_numpy())

    # get actual frequency during holdout period
    frequency_holdout_actual = filtered_cal['frequency_holdout']

    # Log parameters
    mlflow.log_param("model_type", "Modified BG/NBD")    
    mlflow.log_param("l2_penalizer_coef", l2_penalizer_coef)

    # Log metrics
    mae = metrics.mean_absolute_error(frequency_holdout_actual, frequency_holdout_predicted)
    mse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=True)
    rmse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=False)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.pyfunc.log_model(
    'model', 
    python_model=_lifetimesModelWrapper(model), 
    signature=signature,
    conda_env=conda_env
    )
    
    # Model uri later registering
    base_uri = run.info.artifact_uri

# Register model to ML flow to make predictions on the data later  
model_full_name = f"clv_freq_best"
model_uri = base_uri+"/model"
_logger.info(f"Register MLflow Model by uri path: {model_uri}")
new_model_version = mlflow.register_model(model_uri, model_full_name,  await_registration_for=1200)


# STEP 06 --------------------
#
#          Evaluate best model
#
# STEP 06 --------------------

# Let's calculate RMSE and MAE for our newly trained model:
# score the model
frequency_holdout_actual = filtered_cal['frequency_holdout']
frequency_holdout_predicted = model.predict(filtered_cal['duration_holdout'].to_numpy(), 
                                            filtered_cal['frequency_cal'].to_numpy(), 
                                            filtered_cal['recency_cal'].to_numpy(), 
                                            filtered_cal['T_cal'].to_numpy())

# Convert the predicted values to a pandas Series and fill missing values
frequency_holdout_predicted = pd.Series(frequency_holdout_predicted).fillna(0)

# rmse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted.fillna(0), squared=False)
# mae = metrics.mean_absolute_error(frequency_holdout_actual, frequency_holdout_predicted.fillna(0))

rmse = metrics.mean_squared_error(frequency_holdout_actual, frequency_holdout_predicted, squared=False)
mae = metrics.mean_absolute_error(frequency_holdout_actual, frequency_holdout_predicted)

print('RMSE: {0}'.format(rmse))
print('MAE: {0}'.format(mae))

# check that MAE is lower than 0.5
assert(mae<0.5)

# Let's predict purchase count for next year and probility of being alive
filtered_cal['purchases_next365days']=round((
  model.conditional_expected_number_of_purchases_up_to_time(
    365, 
    filtered_cal['frequency_cal'], 
    filtered_cal['recency_cal'], 
    filtered_cal['T_cal']
    )
  ), 2)

filtered_cal['prob_alive']=model.conditional_probability_alive(
    filtered_cal['frequency_cal'], 
    filtered_cal['recency_cal'], 
    filtered_cal['T_cal']
    )

# Value capture by decile
# Make a copy of predictions
decile_df = filtered_cal.sort_values(by='purchases_next365days', ascending=False).copy()

# Compute deciles
decile_df['decile'] = 10 - (pd.qcut(decile_df['purchases_next365days'], 10, labels=False, duplicates='drop'))

# Compute cumulative sum of actual by decile
decile_df['cum_actual'] = decile_df['frequency_holdout'].cumsum()

# Compute cumulative sum of actual for each decile (last value per decile group)
decile_cumsum = decile_df.groupby('decile')['cum_actual'].last()

# Compute total sum of actual
total_actual = decile_df['frequency_holdout'].sum()

# Compute ratio
decile_cumsum_ratio = decile_cumsum / total_actual

# Convert to DataFrame for cleaner display
decile_summary_df = pd.DataFrame({
    'decile': decile_cumsum_ratio.index,
    'ratio_cum_actual_to_total': decile_cumsum_ratio.values
})

# Check that proportion of sales captured at 3rd most valuable decile is higher than 80% of total sales
captured_value_ratio_at_decile_3 = decile_summary_df.iloc[2].ratio_cum_actual_to_total
print(captured_value_ratio_at_decile_3)
assert(captured_value_ratio_at_decile_3>0.8)

# Let's consider consumer with a purchase in the holdout period as alive
filtered_cal['alive'] = np.where(filtered_cal.frequency_holdout>0,1,0)

# Calculate classification performance
metrics.roc_auc_score(filtered_cal.alive, filtered_cal.prob_alive)

# Check that classification performance is at leat 0.80
assert(metrics.roc_auc_score(filtered_cal.alive, filtered_cal.prob_alive)>0.80)

# Let's define an optimal cutoff based on Youden index
def sensivity_specifity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

cutoff = sensivity_specifity_cutoff(filtered_cal.alive, filtered_cal.prob_alive)
print(cutoff)

# Assign predicted label
filtered_cal['pred_alive'] =  np.where(filtered_cal.prob_alive>cutoff,1,0)

print(metrics.classification_report(filtered_cal.alive, filtered_cal.pred_alive))

metrics.f1_score(filtered_cal.alive, filtered_cal.pred_alive)

# Check that F1 score is at least 0.40
assert(metrics.f1_score(filtered_cal.alive, filtered_cal.pred_alive)>0.40)

# Display confusion matrix
pd.crosstab(filtered_cal.alive, filtered_cal.pred_alive, margins=True)


# STEP 07 --------------------
#
#          Final training
#
# STEP 07 --------------------

# After making sure that our model is good enough we can train it with full dataset
# read full dataset
# sampling based on last 2 characters of consumerid
df_full = spark.sql(f"""
select  * 
from    clv.master_full_noam
where   snapshot_dt='{period}'  
    and substr(CustomerID, -2) in('1a', '1b', '1c')  
    and frequency<50
""").toPandas()

print("Row count:", df_full.shape[0])
assert(df_full.shape[0]>3e5)

# make a copy while keeping customers with no repeats
filtered = df_full.copy()
print(filtered.shape)

with mlflow.start_run(run_name=f'freq_final') as run:
    # Create and train model
    model.fit(filtered['frequency'], filtered['recency'], filtered['T'])
    # Log parameters
    mlflow.log_param("model_type", "Modified BG/NBD")    
    mlflow.log_param("l2_penalizer_coef", l2_penalizer_coef)
    # Log model
    mlflow.pyfunc.log_model(
    'model', 
    python_model=_lifetimesModelWrapper(model), 
    signature=signature,
    conda_env=conda_env
    )
    # Model uri later registering
    base_uri = run.info.artifact_uri

# Register model to ML flow to make predictions on the data later  
model_full_name = f"clv_freq_final"
model_uri = base_uri+"/model"
_logger.info(f"Register MLflow Model by uri path: {model_uri}")
new_model_version = mlflow.register_model(model_uri, model_full_name,  await_registration_for=1200)