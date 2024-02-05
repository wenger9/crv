<!-- # Introduction
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started!
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies 
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. _

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore) --> a





### `../gda_clv/utils/common.py`
- Contains: Common utility classes and functions utilized throughout project.
- Key functionalities:
  - Singleton implementation for consistent instances across uses.
  - Data classes such as `FeatureStoreTableConfig` and `TargetsTableConfig` for structured configuration management.
  - `JobContext` class, which is **central in managing and providing context** (i.e. configurations and Spark session) for your jobs.
  - Utility functions for database and feature store interactions.

### `../gda_clv/utils/config_utils.py`
- Contains: Utilities for configuration management, parsing, and loading.
- Key functionalities:
  - Load and convert various configuration files (.yml)
  - Central role in managing environment-specific configurations for the project.

### `../gda_clv/utils/dbx_utils.py`
- Contains: Utilities specifically for handling `dbx` operations.
- Key Functionalities:
  - Allow for detailed job handling, task ordering, and job construction, which are crucial for your CI/CD pipeline setup using `dbx`.
  - Functions such as `reorder_dict`, `remove_task`, and job building logic (`JobFactory`) allow for manipulation and organization of CI/CD tasks.

### `../gda_clv/utils/feature_store_utils.py`
- Contains: Logic dedicated to feature store interactions.
- Key Functionalities:
  - Create feature tables
  - Check for table existence
  - Manage feature update conditions
  - Timeseries aspects of data is handled via the `Timeline` class and related functions.

### `../gda_clv/utils/job_gen.py`
- Contains: Logic for generating job configurations.
- Key Functionalities:
  - Loads templates
  - Configures jobs for different environments (i.e. dev, staging, prod)
  - Dumps these environment configurations to `deployment.yml`.
  - **IMPORTANT**: Using job_gen.py ***aligns*** with the use of `dbx` for deployment, indicating an automated process to generate environment-specific job configurations. **This ***linkage*** is incredibly important to understand when learning how to build Azure CI/CD frameworks. Makes life easier in the long run... trust me.**

### `../gda_clv/utils/job_meta_checker.py`
- Contains: Validation logic for the metadata of jobs (i.e. think of it as a QA check in the CI/CD pipeline, ***prior*** to deployment in any environment)
- Key Functionalities:
  - Functions to ensure the presence and accuracy of commit IDs in the job configurations.

### `../gda_clv/utils/job_templates_scheduler_gen.py`
- Contains: Logic to generate schedules for various jobs, dynamically adjusting them based on the current time.
- Key Functionalities:
  - Modifies `templates.yml`, keeping the job schedules up-to-date and syncd with with real-time requirements.

### `../gda_clv/utils/logger_utils.py`
- Contains: Basic utility for logging.
- Key Functionalities:
  - Sets up a logger with a specific format and level.
  - Allows us to track and debug.


##


### `../gda_clv/utils/pipelines/model_utils.py`
- Contains: Utilities related to machine learning models.
- Key functionalities:
  - **`GetDummies` Class**: Custom transformer for one-hot encoding, leveraging `pandas.get_dummies()`.
  - **`ModelConfig` Data Class**: Manages configuration details for models, including target column, feature columns, and categorical/numeric columns. Supports integration with `mlflow`.
  - **Utility Functions**: 
    - `tree_nbytes` for calculating memory usage of decision trees.
    - `get_estimator_size` for analyzing the size of estimators in a pipeline, particularly forest-based models.
  - Enhances model management and evaluation within the machine learning pipeline.

### `../gda_clv/utils/pipelines/save_to_blob.py`
- Contains: Script dedicated to saving results and metadata to Azure Blob Storage.
- Key functionalities:
  - **`get_commit_id` Function**: Extracts the commit ID from the notebook, aiding in version tracking.
  - **`save_inference_to_blob_storage` Function**: Handles the storage of model inference results in Azure Blob Storage, including file naming and cleanup.
  - **`save_meta_to_blob_storage` Function**: Captures and stores metadata related to the model and execution environment, important for model lineage tracking.
  - Crucial for operational aspects of ML pipelines, ensuring efficient data handling and tracking.


##


### Dependencies Table
| Script Filename               | Dependencies                                       | Description                                                                                                                                                                                                                                                |
|-------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `01_data_preparation.py`      | `mlflow`, `pandas`, `numpy`, `scipy`, `lifetimes`  | - `mlflow` for experiment tracking and model logging.<br> - `pandas` and `numpy` for data manipulation.<br> - `scipy` for statistical functions.<br> - `lifetimes` for BTYD models. |
| `model_training_frequency.py`    | `mlflow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `hyperopt`, `lifetimes` | - `mlflow` for model logging and experiment tracking.<br> - `pandas` and `numpy` for data handling.<br> - `matplotlib` and `seaborn` for plotting (remove later).<br> - `sklearn` for metrics.<br> - `hyperopt` for hyperparameter tuning.<br> - `lifetimes` for BTYD modeling.    |
| `model_training_monetary.py`     | `mlflow`, `pandas`, `numpy`, `lifetimes`           | - `mlflow` for model logging.<br> - `pandas` and `numpy` for data handling.<br> - `lifetimes` for customer lifetime value modeling.                                                                                                                         |
| `model_projection_clv.py`        | `mlflow`, `pandas`, `numpy`, `lifetimes`, `pyspark`| - `mlflow` for model loading and execution.<br> - `pandas` and `numpy` for data processing.<br> - `lifetimes` for CLV calculations.<br> - `pyspark` for Spark SQL and DataFrame operations.                                                                   |

### Configurations Table
| Script Filename               | Configurations                                    | Description                                                                                                                                                                                                                                                |
|-------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `01_data_preparation.py`      | Database connections, Spark SQL queries, MLflow experiment name | - Database connections for Spark SQL queries.<br> - Spark SQL queries to process and prepare data.<br> - MLflow experiment name for tracking                                                                      |
| `model_training_frequency.py`    | MLflow experiment name, model hyperparameters, lifetimes model parameters | - MLflow experiment name for tracking.<br> - Model hyperparameters for optimization (e.g., l2 regularization).<br> - Parameters specific to the lifetimes models (e.g., penalizer coefficient).                                                           |
| `model_training_monetary.py`     | MLflow experiment name, model hyperparameters, lifetimes model parameters | - MLflow experiment name for tracking.<br> - Model hyperparameters for the Gamma-Gamma model.<br> - Parameters specific to the lifetimes models (e.g., penalizer coefficient).                                                                                         |
| `model_projection_clv.py`        | MLflow model paths, Spark SQL queries              | - Paths to MLflow models for loading the frequency and monetary models.<br> - Spark SQL queries for data processing and batch scoring.                                                                                                                      |

##


### Dependency Matrix
A 4x4 matrix where each row and column represents one of the scripts (`01_data_preparation.py`, `model_training_frequency.py`, `model_training_monetary.py`, `model_projection_clv.py`). A green checkmark (✅) indicates a dependency. Click a checkmark for details.

| From \ To                 | `01_data_preparation.py` | `model_training_frequency.py` | `model_training_monetary.py` | `model_projection_clv.py` |
|---------------------------|:------------------------:|:-------------------------:|:------------------------:|:---------------------:|
| `01_data_preparation.py`  |                          |           [✅][1]          |                          |                       |
| `model_training_frequency.py`|                          |                           |          [✅][2]         |         [✅][3]       |
| `model_training_monetary.py` |                          |                           |                          |         [✅][4]       |
| `model_projection_clv.py`    |                          |                           |                          |                       |

### Dependency Descriptions

#### [1]: Dependency from `01_data_preparation.py` to `model_training_frequency.py`
- `model_training_frequency.py` uses data prepared and shaped by `01_data_preparation.py`, specifically features like `frequency_cal`, `recency_cal`, `T_cal`, and `monetary_value_cal`.

#### [2]: Dependency from `model_training_frequency.py` to `model_training_monetary.py`
- `model_training_monetary.py` utilizes the CLV model trained and logged in MLflow by `model_training_frequency.py` for further profit calculations.

#### [3]: Dependency from `model_training_frequency.py` to `model_projection_clv.py`
- `model_projection_clv.py` employs the models and predictions from `model_training_frequency.py` for projecting CLV-related metrics.

#### [4]: Dependency from `model_training_monetary.py` to `model_projection_clv.py`
- `model_projection_clv.py` uses outputs (models and predictions) from `model_training_monetary.py` for detailed CLV calculations and projections.


##


| **Check Name** | **Description** | **Status** |
| -------------- | --------------- | ---------- |
| **Dependency Verification** | Confirm if `model_training_frequency.py`, `model_training_monetary.py`, and `model_projection_clv.py` correctly access any output or data processed by `01_data_preparation.py`. | 🫡 |
| **Path Updates** | Ensure all file paths and references in the scripts are updated to reflect the new structure. This includes data file paths, import statements, and any script-to-script calls. | 🫡 |
| **Configuration File Updates** | Update config .yml files that contain hardcoded paths or script references to align with the new folder structure. | 🫡 |
| **CI/CD Pipeline Adjustments** | Modify the CICD pipeline configs with updated paths. | 🫡 |
| **Testing Script Execution** | Manually execute the scripts in sequential order (should align with order of dependency as well) to check for any runtime errors or unexpected behavior. | ❓ |
| **Unit Testing** | Run unit tests to ensure refactoring didn’t break existing functionalities. | ❓ |
| **Data Output Validation** | Check data output of each script for structure consistency and accuracy. **IMPORTANT** Especiialy for `01_data_preparation.py` since its output is used by subsequent scripts. | ❓ |
| **Logging and Monitoring** | Ensure logging is still functional. Monitor the system logs for any errors during script execution. | ❓ |


##


Add model info here.
BTYD and purpose of combining frequency and monetary outputs.


##


### Dependency Tracking for `model_training_frequency.py`

| Internal / External | Upstream / Downstream | Dependency Name | Description |
|---------------------|-----------------------|-----------------|-------------|
| Internal            | Upstream              | `filtered_cal`  | Dataset prepared in `01_data_preparation.py`. Used for training models and hyperparameter optimization. **IMPORTANT**: Write unit and integration tests using this. |
| External            | Upstream              | None            | No explicit external upstream data dependencies. |
| External            | Upstream              | None            | No explicit external upstream model artifacts dependencies. |
| Internal            | Upstream              | None            | No internal upstream model artifacts used directly by this script. |

#### Notes
- **Internal Upstream Data (`filtered_cal`)**: Ensure `filtered_cal` is prepared and accessible as expected by `model_training_frequency.py`. It's a transformation from `01_data_preparation.py`. Ensuring its availability and integrity is key. 
- **Other Dependencies**: This is an important consideration in maintaining Environment Specific Behavior (ESB). Ensure all scripts run in any environments they are deployed in. This requires checks on the Azure pipeline config files and the `deployment.yml` called from `/conf`. `Deployment.yml` sets all the context for the jobs containing these scripts. Write a test to ensure the environment created **MATCHES** workflows deployed (i.e., jobs + scripts).


##


### GDA-CLV
CICD Directory Structure 
```
gda-clv/
│
├── conf/
│   ├── base.yml
│   ├── deployment.yml
│   ├── environments/
│   │   ├── dev.yml
│   │   ├── prod.yml
│   │   ├── staging.yml
│   ├── models/
│   │   └── NOAM/
│   │       └── clv.yml
│   └── templates.yml
│
├── data_engineering_pipelines/
│   └── NOAM/
│       └── 01_data_preparation.py
│
├── model_pipelines/
│   ├── model_training_frequency.py
│   ├── model_training_monetary.py
│   └── model_projection_clv.py
│
├── utils/
│   ├── api/
│   │   └── __init__.py
│   │   └── make_ws_dir.py
│   │   └── permissions.py
│   ├── pipelines/
│   │   └── __init__.py
│   │   └── model_utils.py
│   │   └── save_to_blob.py
│   └── __init__.py
│   └── common.py
│   └── config_utils.py
│   └── dbx_utils.py
│   └── feature_store_utils.py
│   └── job_gen.py
│   └── job_meta_checker.py
│   └── job_templates_scheduler_gen.py
│   └── logger_utils.py
│
└── README.MD
└── requirements.txt
└── setup.py
└── unit-requirements.txt
```


# Overview of all Utility Files

| Utility File                | Description |
| --------------------------- | ----------- |
| `make_ws_dir.py`            | Automates the creation of directories within the Databricks Workspace using the REST API. It sets up standardized environments for different stages and regions. |
| `permissions.py`            | Manages permissions for MLflow models within Databricks, interfacing with the REST API to manage model access control lists. |
| `common.py`                 | Contains common utility functions and classes for job context, configurations, and interactions with Databricks' DBUtils and databases. |
| `config_utils.py`           | Handles configuration files and command-line arguments, providing methods for conversion, loading, and parsing configurations. |
| `dbx_utils.py`              | Offers utilities for operations related to Databricks jobs, including string manipulation, dictionary handling, and job configuration building. |
| `feature_store_utils.py`    | Provides functions and classes for working with the Databricks Feature Store, managing feature table creation, and updates. |
| `logger_utils.py`           | Configures and returns a Python logger with a specific format, reducing verbosity from certain PySpark modules. |
| `job_gen.py`                | Generates Databricks jobs configuration files from templates, potentially as part of a CI/CD pipeline for automating deployments. |
| `job_meta_checker.py`       | Checks for specific metadata across model paths in a Databricks workspace, ensuring models are updated in line with codebase changes. |
| `job_templates_scheduler_gen.py` | Generates schedules for Databricks jobs, creating cron job formats and updating schedules based on templates and environment. |
| `model_utils.py`            | Works with ML models, providing tools for encoding variables, estimating model size, and retrieving configurations from MLflow. |
| `save_to_blob.py`           | Saves inference results and metadata to Azure Blob Storage, using Databricks utilities to write files to storage. |



