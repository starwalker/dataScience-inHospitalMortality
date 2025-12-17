# TO RUN: python get_artifacts.py >> get_artifacts-logger.txt

import os
import logging

if __name__ == "__main__":
  try:
    logger.debug('Logger is already defined')
  except:
    logger = logging.getLogger('get_model')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', '%Y-%b-%d %I:%M:%S%p %z')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = 0
    logger.info('Logger is defined')


MODEL_NAME = 'HospitalMortality'

EXPERIMENT_ID = '898660927958573'
RUN_ID = '771597f36ccf4b679381ab0dcc006f21'


# Changes the current working directory to the given path
os.chdir(f'c:/Users/grn201/Development/Projects/ECCP/{MODEL_NAME}')

# Gets the model from the databricks mlflow repo, copies it to artifacts
logger.info(f'Extracting model pipeline and artifacts from MLflow \n  >> Experiment ID: {EXPERIMENT_ID} \n  >> Run ID: {RUN_ID}')
dbricks_cmd = f'databricks fs cp -r --overwrite dbfs:/databricks/mlflow/{EXPERIMENT_ID}/{RUN_ID} ./'
os.system(dbricks_cmd)

# Copies model.pkl from artifacts to resources
os.system('cp ./artifacts/model/model.pkl ./deployment/resources')
logger.debug('Copied pickled model to resources')

# # Copies model.pkl from artifacts to resources
# os.rename('./artifacts/pipeline/model.pkl', './artifacts/pipeline/pipeline.pkl')
# os.system('cp ./artifacts/pipeline/pipeline.pkl ./deployment/resources')
# logger.debug('Copied pickled pipeline to resources')

# Copies train_data.tsv from artifacts to resources
os.system('cp ./artifacts/train_data.tsv ./deployment/resources')
logger.debug('Copied sample data to resources')

# Copies column_map.json from artifacts to resources
os.system('cp ./artifacts/column_map.json ./deployment/resources')
logger.debug('Copied column names to resources')