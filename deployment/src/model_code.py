import os
import json
import pandas as pd
import pickle
import sklearn
import xgboost as xgb
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from parcel import Parcel  # Parcel is an Epic released packaging/formatting helper

print(f'Using xgboost=={xgb.__version__}')
print(f'Using scikit-learn=={sklearn.__version__}')

warnings.simplefilter('ignore', FutureWarning)


def load(cwd):
    """
    :param cwd: current working directory
    :return: dict with column names and sklearn pipeline with xgboost model as last step
    """
    COLUMN_MAP_PATH = '/resources/column_map.json'
    MODEL_PATH = '/resources/model.pkl'
    with open(f'{cwd}{COLUMN_MAP_PATH}', 'r') as f:
        column_map = json.load(f)
    print(f'Column map with {list(column_map.keys())} loaded from .{COLUMN_MAP_PATH}')
    with open(f'{cwd}{MODEL_PATH}', 'rb') as f:
        model = pickle.load(f)
    print(f'Model {type(model)} loaded from .{MODEL_PATH}')

    return column_map, model


def predict(data, cwd=os.getcwd()):
    """
    :param data: json payload with features and metadata
    :param cwd: current working directory
    :return: scores and contributions to cache for a patient or batch
    """
    MODEL_NAME = 'HospitalMortality'

    # GET/DEFINE RESOURCES
    print(f'Current working directory: {cwd}')
    # Loads column names and model containing column transformer and model
    column_map, model = load(cwd)

    # Defines lists of column names
    KEY_COLS = [col for col in column_map['KEY_COLS'] if col != 'SnapshotHour']
    INPUT_COLS = column_map['INPUT_COLS']
    FEATURE_NAMES = column_map['FEATURE_COLS']
    BP_COLS = [col for col in FEATURE_NAMES if 'BloodPressure' in col]
    print(f'{len(INPUT_COLS)} input names loaded')
    print(f'{len(FEATURE_NAMES)} feature names loaded')

    CATEGORY_COL_MAP = {
      'AVPU': {
        'ALERT': 1,
        'RESPONSIVE TO VOICE': 2,
        'RESPONSIVE TO PAIN': 3,
        'UNRESPONSIVE': 4
      },
      'IV_Device_WDL': {
        'WDL EXCEPT (SEE ASSESSMENT)': 1,
        'WDL': 2
      }
    }
    CATEGORY_COLS = list(CATEGORY_COL_MAP.keys())

    # PREPARE DATA
    # Lists the feature names and their types in order to unpack
    ordered_columns = [(col, "str") for col in INPUT_COLS + KEY_COLS]
    df, chronicles_info = Parcel.unpack_input(data, ordered_columns)  # For testing: df = pd.read_csv(f'{cwd}/resources/train_data.tsv', sep='\t')
    print(f'Data unpacked to {type(df)}')

    # Sets index to keys and limits to feature columns used by model
    df.set_index(KEY_COLS, inplace=True)
    df = df[INPUT_COLS]

    # Changes text values to uppercase and handles unknown categories to standardize for category mapping
    for col in CATEGORY_COLS:
        df[col] = df[col].str.upper()
        allowed_values = list(CATEGORY_COL_MAP[col].keys())
        # Sets unknown categories to empty
        df.loc[~df[col].isin(allowed_values), col] = 'nan'

    # Restores empty values which were converted to string during unpacking and changes NaN to None
    df = df.replace({'nan': None})
    df = df.where(pd.notnull(df), None)
    print('Example input:', json.dumps(df.head(1).to_dict('records'), indent=2))

    # Checks that all required input columns have been unpacked
    missing = (set(INPUT_COLS).difference(set(df.columns)))
    if missing:
        print(f'Missing columns {missing}')
    else:
        print('Input column check passed')

    # PREDICT RISK PROBABILITY
    predictions = model.predict_proba(df)[:, 1]
    print('Predictions:', [p for p in predictions])

    # Puts together dictionary of prediction values
    formatted_predictions = {"Predictions": {MODEL_NAME: [float(p) for p in predictions]}}

    # GET FEATURE CONTRIBUTIONS
    pipeline, booster = Pipeline(model.steps[0:-1]), model.steps[-1][1].get_booster()
    x = pipeline.transform(df)
    contribs = booster.predict(xgb.DMatrix(x), pred_contribs=True)
    contributions, bias = contribs[:, 0:-1], contribs[:, -1:]
    print(f'Contributions of shape {contributions.shape}')

    print('Calculating absolute contribution percents')
    abs_contributions = np.abs(contributions)
    for i, row in enumerate(abs_contributions):
        total = np.sum(row)
        abs_contributions[i, ] = abs_contributions[i, ] / total * 100

    # Checks that number of feature names and contributions match
    if len(FEATURE_NAMES) != abs_contributions.shape[-1]:
        print('Number of contributions columns does not equal number of feature names!')
    else:
        print('Feature contributions check passed')

    # Puts together dictionary of feature contributions
    feature_contributions = {}
    for i, feature in enumerate(FEATURE_NAMES):
        feature_contributions[feature] = {"Contributions": [round(float(c), 1) for c in abs_contributions[:, i]]}

    # PREPARE DISPLAY FEATURES
    feats_df = pd.DataFrame(x, columns=FEATURE_NAMES)
    display_df = df.copy()

    # Adds systolic and diastolic columns to display features
    for col in BP_COLS:
        display_df[col] = feats_df[col].values
    display_df.Sex.replace({1: 'Female', 2: 'Male'}, inplace=True)
    display_df.IcuLevelOfCareFlag.replace({0: 'No', 1: 'Yes'}, inplace=True)

    # Changes missing values from NaN to None
    display_df = display_df.where(pd.notnull(display_df), None)
    display_df = display_df.fillna('unspecified')
    # print('Example display values:', json.dumps(display_df.head(1).to_dict('records'), indent=2))

    # Puts together dictionary of feature values to display in hover bubble
    additional_features = {}
    for feature in FEATURE_NAMES:
        additional_features[feature] = {"Values": [val for val in display_df[feature].tolist()]}

    return Parcel.pack_output(
        mapped_predictions=formatted_predictions,
        score_displayed=MODEL_NAME,
        chronicles_info=chronicles_info,
        feature_contributions=feature_contributions,
        additional_features=additional_features
    )