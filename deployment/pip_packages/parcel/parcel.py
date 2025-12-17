#----------------------------------------------------
# © 2017 - 2019 Epic Systems Corporation.
# Chronicles® is a registered trademark of Epic Systems Corporation.
#---------------------------------------------------

"""
Parcel
=====

Provides
  1. API to unpack non-time series data from Chronicles into a panda's dataframe
  2. API to unpack time series data from Chronicles
  3. API to pack predictions to be sent back to Chronicles
  4. API to pack retrained features to send back to Chronicles

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`Model Development Overview`_.
"""
import logging
import pandas as pd
import numpy as np
import json
from enum import Enum
from .converters import Converter

class Parcel:
    """
    Class used to pack and unpack data for communication with Chronicles
    """
    __invalid_dtypes = [ np.dtype('bool') ]

    @staticmethod
    def unpack_input(data, ordered_columns, converters=None):
        """
        Splits information needed by chronicles from data needed for predictions

        Parameters:
        ---------------------------------------------------------------------------------------
            data:               input json payload
            ordered_columns:    List which determines the order of columns in the returned DataFrame and their types.
                                formatted as [("Feature1", numpy.dtype),("Feature2", numpy.dtype),...]
            converters:         Dictionary of features and converters. a converter is a subclass of parcel.converters.Converter
                                {"Feature1": converter_1, "Feature2": converter_2,...}

        Returns:
        # NOTE: don't modify chronicles_info, just pass it back in pack_output
        dataframe, chronicles_info = unpack_input(data)

            Request (data) schema:
            "Data": {
                "OutputType": "",
                "PredictiveContext": {},
                "EntityId": [
                        { "ID": "", "Type": "" },
                        { "ID": "", "Type": "" },
                        ...
                    ],
                "Feature1": [],
                "Feature2": []
                ...
                }
        """
        if converters is None:
            converters = {}
        else:
            for feature, converter in converters.items():
                if not isinstance(converter,Converter):
                    raise TypeError(f"Unsupported Converter for {feature}: {converter}")

        ordered_columns = list(ordered_columns) # make sure we have list, not just an iterator. We need the iterator twice, and it will be exhausted after the first use.

        # Input payload has a wrapping "Data" node; unwrapping
        data = data.pop("Data")
        # Pop the information from the inbound payload not relevant to prediction and return it separately
        chronicles_info = {}
        chronicles_info["OutputType"] = data.pop("OutputType", None)
        chronicles_info["PredictiveContext"] = data.pop("PredictiveContext", None)
        chronicles_info["EntityId"] = data.pop("EntityId", None)

        column_names, ordered_types = zip(*ordered_columns)
        dataframe = pd.DataFrame(data, columns=column_names, dtype="object")
        for feature, dtype in ordered_columns:
            if np.dtype(dtype) in Parcel.__invalid_dtypes:
                raise TypeError(f"Unsupported dtype for Parcel: {dtype}")

            if feature in converters:
                dataframe.loc[:,feature] = converters[feature].convert(dataframe[feature])

            int_type = (dtype == np.dtype('int64'))
            # Interconnect sends missing/null values as {} which cannot be casted, and need to be converted to NaNs
            dataframe.loc[:, feature] = dataframe[feature].apply(func=lambda x: np.nan if (x == {} or x == '{}') else x)
            # Interconnect also sends all feature values as strings, and thus they need to be coerced to the type trained upon
            if int_type and dataframe[feature].isnull().any():
                # If we don't cast to float, we crash since there a NaN value in an interger column. 
                logging.warning("Provided a missing value to an int data type, will convert to float64.")
                dataframe.loc[:, feature] = dataframe[feature].astype("float64") 
            else:
                dataframe.loc[:, feature] = dataframe[feature].astype(dtype)

        error_rows = pd.Index([]) #start with an empty index
        for feature, converter in converters.items():
            converter.log_issues()
            error_rows = error_rows.union(converter.get_error_index())

        if len(error_rows) > 0:
            # Move error rows to the end of the entity list. 
            good_entity_list = []
            bad_entity_list = []
            for idx, entity in enumerate(chronicles_info["EntityId"]):
                if idx in error_rows:
                    bad_entity_list.append(entity)
                else:
                    good_entity_list.append(entity)
            chronicles_info["EntityId"] = good_entity_list + bad_entity_list

            #now drop the bad rows so they don't get scored
            dataframe.drop(error_rows, inplace=True)
        return dataframe, chronicles_info
        
    @staticmethod
    def pack_output(mapped_predictions, score_displayed, chronicles_info=None, feature_contributions=None, additional_features=None):
        '''
        Adds outputs and their scores to the return payload (e.g. "Probability_Septic", or "Probability_Versicolor") via predictions:
    
        Parameters:
        ---------------------------------------------------------------------------------------
            mapped_predictions: a dictionary of all outputs, where each output contains a map to the vector of predictions for each class probability, the regression,
            or class label for each sample. Note that while the schema supports multiple outputs, they are not well handled by the Predictive Analytics infrastructure.
                    predictions = {
                                    "IrisClassification": {
                                        "Probability_Versicolor": [flower1_pred_prob, flower2_pred_prob, ... ],
                                        "Probability_Setosa" : [flower1_pred_prob, flower2_pred_prob, ... ],
                                        ... 
                                    },
                                    "1YrHeightProjection": {
                                        "ProjectedFlowerHeight": [flower1_height, flower2_height, ...]
                                    }
                                }
            score_displayed: Dictates which value from mapped_predictions[Output1] is displayed most predominantly in Hyperspace (should be the probability of "yes" for binary classification).
                    This is also the display name which will be shown in Hyperspace. 
                    The other outputs are accessible, but for many model types, this is the default display value. If multiclass, score_displayed should be the
                    class of most interest.
            chronicles_info: the json object holding metadata needed for chronicles (returned from unpack_input())
            feature_contributions: (optional) the json object used to determine contributions shown in hyperspace
            additional_features: (optional) a dict of features in addition to input features received from chronicles, such as external data (weather, census, etc.)
                    additional_features = {
                        "feat1" : { "Values: [val1, val2,.., valN] }
                        "feat2" : { "Values: [val1, val2,.., valN] }
                    }
                    // NOTE: This data will be persisted in Chronicles.
                            If this data comes from third party source, you may require legal permission to persist this data

        Response schema:
 
        {
            "OutputType": "",
            "PredictiveContext": { },
            "ScoreDisplayed": "RegressionOutputOrClass1Probabilities",
            "EntityId": [
                {  "ID": "", "Type": ""},
                {  "ID": "", "Type": ""},
                ...
            ],
            ...
            "Outputs": {
                "IrisClassification": {
                    "Scores": {
                        "RegressionOutputOrClass1Probabilities" : {
                            "Values": []
                        },
                        "Class2Probabilities" : {
                            "Values" : []
                        },
                        ...
                    }
                    "Features": {
                        "Feature1": {
                            "Contributions":[]
                    },
                        "Feature2": {
                            "Contributions":[]
                    },
                    ...
                },
                // Data for additional features (external data)
                "Raw": {
                    "feat1" : { "Values: [val1, val2,.., valN] }
                    "feat2" : { "Values: [val1, val2,.., valN] }
                },
                // NOTE: multiple output models are not supported in the Predictive Analytics infrastructure at this time
                "1YrHeightProjection": {
                    "Scores": {
                        "RegressionOutputOrClass1Probabilities" : {
                            "Values": []
                        },
                        "Class2Probabilities" : {
                            "Values" : []
                        },
                        ...
                    }
                    "Features": {
                        "Feature1": {
                            "Contributions":[]
                    },
                        "Feature2": {
                            "Contributions":[]
                    },
                    ...
                },
                ...
            }
        }
        '''

        return_payload = {}
        return_payload["Outputs"] = {}
        return_payload["ScoreDisplayed"] = score_displayed
        
        # merge the mapped_predictions into the return payload
        for output in mapped_predictions:
            return_payload["Outputs"][output] = {}
            return_payload["Outputs"][output]["Scores"] = {}

            # score name might be the name of a class, or name of a regression output
            for score_name in mapped_predictions[output]:
                # since these score_names might be a numpy object (e.g. a class in the training set), 
                # cast to string to ensure they are json serializable 
                # also replace NaN's with nulls since they can throw off JSON serialization between python and node.
                str_score_name = str(score_name)
                return_payload["Outputs"][output]["Scores"][str_score_name] = {}
                return_payload["Outputs"][output]["Scores"][str_score_name]["Values"] = [None if np.isnan(x) else x for x in mapped_predictions[output][score_name]]
                
            # requires feature contributions to be formatted correctly (see "Features" in the output schema above)
            return_payload["Outputs"][output]["Features"] = {}
            if feature_contributions is not None:
                for feature in feature_contributions:
                    # replace NaN's with nulls since they can throw off JSON serialization between python and node.
                    return_payload["Outputs"][output]["Features"][feature]={}
                    return_payload["Outputs"][output]["Features"][feature]["Contributions"]=[None if np.isnan(x) else x for x in feature_contributions[feature]["Contributions"]]
        
        # merge the chronicles info into the return payload if chronicles_info was passed in
        if chronicles_info is not None:
            for meta_info_field in chronicles_info:
                return_payload[meta_info_field] = chronicles_info[meta_info_field]
                
        if additional_features is not None:
            return_payload["Raw"] = additional_features

        return return_payload
    
    @staticmethod
    def unpack_timeseries_input(data, as_type=None, ordered_features=None):
        """
        Splits information needed by chronicles from data needed for predictions.
        
        Parameters:
        ---------------------------------------------------------------------------------------
            data: input json payload
            as_type: name of return type. Defaults to dictionary, but can be:
                        "dict"   - default, dict[feature][entity][time_step]
                        "pandas" - for a pandas.Series in [feature][entity][time_step]
                        "numpy"  - for a numpy multidimensional array[entity][time_step][feature] requires a uniform number of timesteps over all features
            ordered_features: list of feature names to be used to sorting indices, defaults to the order of the keys in data dictionary
        Returns:
            # don't modify chronicles_info--pass it back into package_timeseries_output
            timeseries_data, chronicles_info = Parcel.unpack_timeseries_input(data)

        Request(data) schema:
               "Data": {
                    "EntityId": [
                            { "ID": "", "Type": "" },
                            { "ID": "", "Type": "" },
                            ...
                        ],
                    "BedsOccupied": ["bedsHour1\tbedsHour2\t...", "bedsHour1..."],
                    "HumidityPerHour": [...]
                    "AnotherTimeSeriesNeededForThisPrediction":[...]
                    ...
                }
        Return timeseries_data as:
            as_type=="dict":
                dictionary["feature"] = array[entity][timestep]
            as_type=="panda"
                pandas.Series["feature"][entity][time_step]
            as_type=="numpy"
                numpy.array[entity][time_step][feature] - requires an equal number of timesteps per feature
        """
        
        # Pop the information from the inbound payload not relelvant to prediction and return it separately
        chronicles_info = {}
        data = data["Data"] # taking off the wrapper of the actual data content
        chronicles_info["OutputType"] =  data.pop("OutputType", None)
        chronicles_info["PredictiveContext"] =  data.pop("PredictiveContext", None)
        chronicles_info["EntityId"] = data.pop("EntityId", None)

        timeseries_data = {}
        max_timesteps = 0 # we generally expect the number of timesteps to be uniform for the numpy option. We store the max number of timesteps for all features here

        for key in data:
            timeseries_data[key] = []
            for tab_delim_sample in data[key]:
                if tab_delim_sample != {} :
                    time_samples = tab_delim_sample.split("\t")
                    timeseries_data[key].append(time_samples)
                    if len(time_samples) > max_timesteps:
                        max_timesteps = len(time_samples)
                else:
                    timeseries_data[key].append([])

        if as_type in ["pandas","numpy"]:
            # Convert dict into an pandas series, use index to keep the 
            # keys in specified order, otherwise it will resort to dict_keys
            if ordered_features == None:
                logging.warning("No feature ordering provided. Will sort keys alphabetically.")
                ordered_features = sorted(timeseries_data.keys())
            timeseries_data = pd.Series(timeseries_data, index=ordered_features)
            
                

        if as_type == "numpy":
            # Convert pandas into a rectangular np-array with index [entity][timestep][feature]
            skipped_entities = set()
            for entities in timeseries_data.values:
                for entity_index, timesteps in enumerate(entities):
                    if (len(timesteps) != max_timesteps):
                        skipped_entities.add(entity_index)
                        
            # if there is a mismatch for timesteps for any entity/feature combination, we will remove that entity and all of its corresponding features.
            if len(skipped_entities) != 0:
                if len(skipped_entities) == len(chronicles_info["EntityId"]):
                    raise ValueError("No valid entities for scoring, all entities had at least one feature with mismatched timesteps")
                    
                entity_list = []
                skipped_entity_list = []
                for entity_index, entity in enumerate(chronicles_info["EntityId"]):
                    if entity_index in skipped_entities:
                        logging.warning("Timesteps do not match for entity " + str(entity) + " removing from prediction batch.")
                        skipped_entity_list.append(entity)
                    else:
                        entity_list.append(entity)

                # sort skipped entities to the end of chronicles info
                chronicles_info["EntityId"] = entity_list + skipped_entity_list

                # delete the timeseries data for skipped entities
                for entity_index in sorted(skipped_entities, reverse=True):
                    for feature in timeseries_data.index:
                        del timeseries_data[feature][entity_index]

            # Transpose numpy array to better ordering of indexes
            # from (features, entities, time_steps) 
            # to (entities, time_steps, features) 
            timeseries_data = np.array(timeseries_data.values.tolist()).transpose(1,2,0)

        return timeseries_data, chronicles_info

    @staticmethod
    def pack_timeseries_output(timeseries_map, chronicles_info, additional_features=None):
        """
        Adds each prediction timeseries to the return payload (e.g. "EDBedCountForcast" and "OtherForcast")

        Parameters:
        ---------------------------------------------------------------------------------------
            timeseries: a map of names of each return time series to a list of lists denoting each forcast.
                        That is, timeseries["EDBedCountForcast"][0] would be the first forcast
            chronicles_info: the json object holding chronicles information (returned from unpack_input())
            additional_features: (optional) a dict of features in addition to input features received from chronicles, such as external data (weather, census, etc.)
                    additional_features = {
                        "feat1" : { "Values: ["v1\tv2\v3",.., "v1\tv2\v3"] }
                        "feat2" : { "Values: ["v1\tv2\v3",.., "v1\tv2\v3"] }
                    }
                    // NOTE: This data will be persisted in Chronicles.
                            If this data comes from third party source, you may require legal permission to persist this data

        Response schema (note we're not filling out raw or transformed features for any current use case):

        {
            "OutputType": "",
            "PredictiveContext": {},
            "EntityId": [
                            { "ID": "", "Type": "" },
                            { "ID": "", "Type": "" },
                            ...
                        ],
            "Outputs": {
                "EDBedCountForcast": {
                    "Values": ["forcastHour1\tForcastHour2...","forcastHour1\tforcastHour2...",...]
                    }        
                },
                "TimeseriesOutput2": {
                    "Values": [...]
                        ...
                    }        
                },
            },
            // Data for additional features (external data)
            "Raw": {
                "feat1" : { "Values: ["v1\tv2\v3",.., "v1\tv2\v3"] }
                "feat2" : { "Values: ["v1\tv2\v3",.., "v1\tv2\v3"] }
            }
        }

        """
        chronicles_info["Outputs"] = {}
        for output_name in timeseries_map:
            values = []
            for sample_list in timeseries_map[output_name]:
                # create a tab delimited string for each sample in the list of values
                sample_list = [str(i) for i in sample_list]
                values.append("\t".join(sample_list)) 
            
            chronicles_info["Outputs"][output_name] = {}
            chronicles_info["Outputs"][output_name]["Values"] = values
        
        if additional_features is not None:
            chronicles_info["Raw"] = additional_features

        #return_package = {}
        # return_package["Results"] = chronicles_info
        # this package will be plugged into the "results" node later
        return chronicles_info #return_package
    
    class ModelState(Enum):
        PRIOR = "Prior"
        RETRAINED = "Retrained"
    
    @staticmethod
    def pack_retrain_results(chronicles_info, features, prior_model_details=None, retrained_model_details=None):
        """
        Produce the payload needed to be passed back to chronicles after a train. 
        Needs to be returned in order for a retrain to be considered successful.
        prior_model_details and retrained_model_details are shown to the end user. 
        Only pass statistics and factor relevance that are relevant for making decisions
        about the quality of the retrain by an end user.

        Parameters: 
        ---------------------------------------------------------------------------------------
            chronicles_info:    the json object holding chronicles information 
                                (returned from Parcel.unpack_input()); needs HDAId, output_displayed

            features:   the list of pre-transformed feature names used in the model. This information
                        is used by Cache to determine which "features" (aka mnemonics / factors) are 
                        sent to the model at predict time
            
            prior_model_details: A tuple of two dicts for the prior model:
                                1. Model Statistics (Ex.: {"R2":"0.5","AUC":"0.6"})
                                2. Model Relevance  (Ex.: {"NUM1":"4","CAT1":"6"})
                                These are displayed to the end user

            retrained_model_details: Same as prior_model_details, but with values
                                    for the retrained model
        
        Raises:
        ----------------------------------------------------------------------------------------
        ValueError
            if prior_model_details is not None, retrained_model_details cannot be None
        """
        if prior_model_details is not None and retrained_model_details is None:
            raise ValueError("retrained_model_details cannot be None if prior_model_details is not None")

        Parcel._validate_model_details(prior_model_details)
        Parcel._validate_model_details(retrained_model_details)

        # Data node present in payload, but model code might pop it off
        if "Data" in chronicles_info:
            chronicles_info = chronicles_info["Data"]
        chronicles_info.update({"Features" : features})

        Parcel._update_model_details(chronicles_info, prior_model_details, Parcel.ModelState.PRIOR)
        Parcel._update_model_details(chronicles_info, retrained_model_details, Parcel.ModelState.RETRAINED)

        return chronicles_info
    
    @staticmethod
    def _validate_model_details(model_details):
        if model_details is None:
            return
        if not isinstance(model_details, tuple):
            raise TypeError("model_details arguments must be of type Tuple")
        if not len(model_details) == 2:
            raise ValueError("Tuple model_details must have exactly two elements")
        for elem in model_details:
            if elem is not None and not isinstance(elem, dict):
                raise TypeError("Elements in model_details can be a dict (or None)")

    @staticmethod
    def _update_model_details(chronicles_info, model_details, model_state):
        """
        If model_details are present, update PredictiveContext key in chronicles info with them.

        Parameters: 
        ---------------------------------------------------------------------------------------
            chronicles_info:    the json object holding chronicles information 
            model_details:  A tuple of two dicts for the model (stats, relevance)
            model_state: Parcel.ModelState ENUM (PRIOR OR RETRAINED)
        """
        if model_details is None:
            return
        
        disp_stats, disp_relevance = model_details

        pcontext = {}
        if "PredictiveContext" in chronicles_info:
            pcontext = chronicles_info["PredictiveContext"]

        disp_stats_key = model_state.value + "ModelStats"
        disp_relevance_key = model_state.value + "ModelRelevance"
        
        if disp_stats is not None: 
            pcontext.update({
                disp_stats_key: json.dumps(disp_stats, sort_keys=True),
            })
        
        if disp_relevance is not None:
            pcontext.update({
                disp_relevance_key: json.dumps(disp_relevance, sort_keys=True),
            })
 
        chronicles_info["PredictiveContext"] = pcontext

    @staticmethod
    def validate_prediction_return(data):
        """
        Determines if a model return matches interconnect spec

        {
            "OutputType": //string
            "EntityId": [ { "Type": "EPI", "ID": 0 }, { "Type": "EPI", "ID": 1 }, ... ]
            "ScoreDisplayed": "A <<ScoreNameString>> from Scores dictionary keys",
            "Outputs": { //only supports one key
                "<<AStringKey>>" : {
                    "Scores" : {
                        "<<ScoreNameString1>>": {
                            "Values": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                        },
                        "<<ScoreNameString2>>": {
                            "Values": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                        }
                    },
                    "Features": {
                        "<<FeatureNameString1>>": {
                            "Contributions": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                        },
                        "<<FeatureNameString1>>": {
                            "Contributions": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                        },
                        "<<ExternalFeatureNameString1>>": {
                            "Contributions": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                        }
                    }
                }
            },
            "Raw": {
                "<<ExternalFeatureNameString1>>": {
                    "Values": [ //array of string, float, int, or null (length less than count of EntityIDs) ]
                }
            },
            "PredictiveContext": //dictionary<str,str> or null
        }
        """
        pass

    @staticmethod
    def validate_train_return(data):
        """
        Determines if a model return matches interconnect spec
        {
            "Features":  [ //array of strings (feature name mnemonics) ],
            "PredictiveContext": //dictionary<str,str> or null
        }
        """
        pass