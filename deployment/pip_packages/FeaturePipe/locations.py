import pandas as pd
import logging
from FeaturePipe import setup_logger
from sklearn.base import TransformerMixin
import re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
import pkg_resources
import warnings
logger = logging.getLogger('FeaturePipe')

'''
GeoCoding System that uses csv file located in resources to return lat, lon
from zipcode, city state or state
source file for te original csv "http://docs.gaslamp.media/wp-content/uploads/2013/08/zip_codes_states.csv"

'''


def _lat_long_error_hander(func):
    '''
    wrapper method to convert all errors to key errors so the can be caught, also erros with None output
    '''
    def handler(*args, **kwargs):
        try:
            output =func(*args, **kwargs)
            if None in output:
                msg = 'output contains None Values, key not found {}'.format(func)
                raise KeyError(msg)
            return output
        except:
            msg = 'method {} failed '.format(func)
            raise KeyError(msg)
    return handler



def _na_fill(func, fill_val=(None, None)):
    '''
    wrapper that returns (None, None) if the even that a method fails

    '''
    def handler(*args, **kwargs):
        try:
            output =func(*args, **kwargs)
            if None in output:
                msg = 'output contains None Values, key not found {}'.format(func)
                logger.debug(msg)
                return fill_val
            return output
        except:
            msg = 'method {} failed '.format(func)
            logger.debug(msg)
            return fill_val
    return handler


def _preproccess_string(x):
    '''
    strips extra white spaces and lower cases string input
    param x:
    return str
    '''
    try:
        return int(x)
    except:
        pass
    try:
        x = x.lower()
        x = re.sub('[^a-z0-9]', ' ', x)
        x = re.sub('  ', ' ', x)
        x = x.strip()
        return x
    except AttributeError:
        return None

def _load_state_abbreviations():
    path = 'resources//abbr-name.json'
    stream = pkg_resources.resource_stream('FeaturePipe', path)
    d = json.load(stream)
    d = dict(zip(list(map(_preproccess_string, d.values())), list(map(_preproccess_string, d.keys()))))
    return d


def _load_location_data():
    '''
    loads data from resources with lat lon by zipcode data
    '''
    path = 'resources//us-zip-code-latitude-and-longitude.csv'
    stream = pkg_resources.resource_stream('FeaturePipe', path)
    df = pd.read_csv(stream, sep=";")

    df['zip_code'] = df['Zip'].astype(int)
    df.set_index('zip_code', inplace=True)
    df.loc[:, 'city'] = df.loc[:, 'City'].apply(lambda x: _preproccess_string(x))
    df.loc[:, 'state'] = df.loc[:, 'State'].apply(lambda x: _preproccess_string(x))
    df.loc[:, 'longitude'] = pd.to_numeric( df.loc[:, 'Longitude'] )
    df.loc[:, 'latitude']= pd.to_numeric( df.loc[:, 'Latitude'] )
    df.sort_index(inplace=True)
    return df

_state_abbreviations = _load_state_abbreviations()
_locations = _load_location_data()


@_lat_long_error_hander
def get_loc_from_zip(zipcode, locations=_locations, **kwargs):
    '''
    Gets lat, lon from locations
    param x: int zipcode
    param locations: pd.DataFrame with zip_code as index, city, state, county, lat and lon coluns
    if locations are not specified, trys _locations global variable

    returns: pd.Series with (lat, lon) in degree decimal format
    '''
    try:
        zipcode = int(zipcode)
        vals = locations.loc[zipcode, :]
        output =  vals['latitude'], vals['longitude']
        return output
    except:
        msg = '{0} :not found using loc_from_zip'.format(zipcode)
        raise KeyError(msg)


@_lat_long_error_hander
def get_loc_from_city_state(city_state, locations=_locations, state_abbreviations=_state_abbreviations):
    '''
    Gets lat, lon from locations
    param x: str  "City, State"
    param locations: pd.DataFrame with zip_code as index, city, state, county, lat and lon coluns
    if locations are not specified, trys _locations global variable

    returns: pd.Series with (lat, lon) in degree decimal format
    '''
    try:
        city_state = _preproccess_string(city_state).split(' ')
        city_state = [v for v in city_state if len(v)>1]
        city =  ' '.join(city_state[0:-1])
        state =  city_state[-1]

        msg = ' looking up "{0}" "{1}" with get_loc_from_city_state'.format(city, state)
        logger.debug(msg)
        if len(state) > 2:
            state = state_abbreviations[state]
        df_temp =_locations.loc[locations['state'] == state]
        df_temp = df_temp.loc[locations['city'] == city]
        vals = df_temp.dropna().mean()
        if vals.isna().any(axis=None):
            raise KeyError
        return  vals['latitude'], vals['longitude']
    except:
        msg = '{0} {1} not found using get_loc_from_city_state'.format(city, state)
        raise KeyError(msg)


@_lat_long_error_hander
def get_loc_from_state(state, locations=_locations, state_abbreviations=_state_abbreviations):
    '''
    Gets lat, lon from locations
    param x: str "State"
    param locations: pd.DataFrame with zip_code as index, city, state, county, lat and lon coluns
    if locations are not specified, trys _locations global variable

    returns: pd.Series with (lat, lon) in degree decimal format
    '''
    try:
        state = _preproccess_string(state)
        if len(state) > 2:
            state = state_abbreviations[state]
        df_temp =locations.loc[locations['state'] == state]
        vals = df_temp.dropna().mean()
        if vals.isna().any(axis=None):
            raise KeyError
        return vals['latitude'], vals['longitude']
    except:
        msg = ' {} not found using get_loc_from_state'.format(state)
        raise KeyError(msg)


@_na_fill
@_lat_long_error_hander
def get_loc(x,  locations=_locations, state_abbreviations=_state_abbreviations):
    '''
    Gets lat, lon from locations
    param x: str or int zipcode, string uses "," to split should be "City, State", "State" or "County, State"
    param locations: pd.DataFrame with zip_code as index, city, state, county, lat and lon coluns
    if locations are not specified, trys _locations global variable

    States are two digit codes , words for United States Only

    Uses the following logic to get lat and lon
        trys as zipcode,
        trys as "city, state"
        trys as 'county, state'
        trys as 'state'
        if all else fails returns None,None as lat, lon
    returns: pd.Series with (lat, lon) in degree decimal format
    '''

    if isinstance(x, type(None)):
        return KeyError('None Type input to get_loc')
    else:
        x = _preproccess_string(x)
    msg = 'get_loc "{}" after preprocessing'.format(x)
    logger.debug(msg)

    # case where input is a zipcode
    try:
        msg = 'trying get_loc_from_zip ({}) after preprocessing'.format(x)
        logger.debug(msg)
        output = get_loc_from_zip(x, locations=locations)
        msg = '{} found in  zips ","'.format(x)
        return output

    except KeyError:
        msg = ' get_loc() {} not found in zipcodes'.format(x)
        logger.debug(msg)

    # disables secondary lookup if it's a zipcode
    try:
        x = int(x)
    except:
        pass

    if isinstance(x, int):
        msg = '{} is converable to int, and cannot be a city or state'.format(x)
        logger.debug(msg)
        raise KeyError(msg)

    # case where input is a state
    try:
        msg = 'trying get_loc_from_state ({}) after preprocessing'.format(x)
        logger.debug(msg)
        output = get_loc_from_state(x, locations=locations, state_abbreviations=state_abbreviations)
        msg = '{} found in  zips ","'.format(x)
        return output

    except KeyError:
        msg = ' get_loc() {} not found in zipcodes'.format(x)
        logger.debug(msg)

    # case where input is a city state
    try:
        msg = 'trying get_loc_from_city_state({}) after preprocessing'.format(x)
        logger.debug(msg)
        output = get_loc_from_city_state(x)
        msg = '{} found in city states ","'.format(x)
        logger.debug(msg)
        return  output

    except KeyError:
        msg = '{} not found in city states ","'.format(x)
        logger.debug(msg)

    msg = ' get_loc({}) not found any method'.format(x)
    logger.debug(msg)
    raise KeyError(msg)


class GeoCoder(TransformerMixin):
    '''
    Geocoder for zipcodes, states, city states and y state

    Applies transfrom to an input data frame with a location columns
    returns an pd DataFrame with "lat" and "lon" columns that represent then
    geograpgic center of the zipcode, state (by averaging zipcodes) or city or get_loc_from_county_state

    In City, State mode, assumes that City,State is specified,
    useage:
        df = pd.DataFrame([29412, 49707], columns=['zipcode'])
        g = GeoCoder(input_col='zipcode')
        g.fit_transform(df))

        df = pd.DataFrame(['Alpena, MI', 'Detroit, MI', 'asdfkasdf ff'], columns=['addr'])
        g = GeoCoder(input_col='addr', remainder='passthrough')
        g.fit_transform(df))

    '''
    def __init__(self,
                 input_col=None,
                 output_cols = ('lat', 'lon'),
                 remainder='drop'):
        '''
        param input_col: string input column name of zipcode, or string location data
        param output_cols: tuple of two strings, name of output columns if remained='passthrough'
        param remainded: string in ('drop', 'passthrough'), drop method returns only a numpy array of lat
        lon when the transform method is called, 'passthrough' adds output_cols to the input data frame

        usage:
        list or numpy array input should have remainder='drop'
        pd.Dataframe input, can use either depending on user preferences

        '''

        self.input_col = input_col
        self.output_cols = output_cols
        self._locations = _load_location_data().copy()
        self._state_abbreviations = _load_state_abbreviations().copy()
        self.remainder = remainder



    def fit(self, X, y=None):
        '''
        Geocoder Fit Method
            if remainder = 'passthrough', X must be a pandas data frame
            This runs the entire transform method, however no actually values are learned,
            so can be run on a subset

        param X: numpy array, pd.Dataframe or list
        param y: array, or None (unused for sklearn pipeline compatability)
        returns self

        '''
        X = self.transform(X)
        return self

    def transform(self, X):
        '''
        Extracts Lat and Lon from string or zip code
        param X: array like object
        returns: pd.DataFrame with lat and lon columns in degree decimal format
        '''

        if isinstance(self.input_col, type(None)):
            input_array = X
        else:
            try:
                input_array = X.loc[:, self.input_col].values
            except:
                input_array = X[:, self.input_col]
        coords = np.reshape(list(map(lambda x: get_loc(x, self._locations, self._state_abbreviations),
                                   input_array)), (-1,2))
        msg = 'coords value {}'.format(coords)
        logger.debug(msg)
        if self.remainder == 'passthrough':
            X.loc[:, self.output_cols[0]] = pd.to_numeric(coords[:, 0])
            X.loc[:, self.output_cols[1]] = pd.to_numeric(coords[:, 1])
            return X
        else:
            return coords.astype(float)

    def get_feature_names():
        '''
        retruns self.output_cols (tuple of two strings)
        '''
        return self.output_cols


def _locations_test():
    # tests indiviual lookup functions
    assert get_loc_from_zip(29412) ==  (32.73727, -79.95409000000001)
    assert get_loc_from_state('MI') == (43.458386228595174, -84.73934209226933)
    assert get_loc_from_city_state('Alpena, MI') == (45.08583, -83.46410999999999)

    # tests loc function that combines lookup functions
    logger.debug('testing get loc')
    assert get_loc(29412) ==  (32.73727, -79.95409000000001)
    assert get_loc('MI') ==  (43.458386228595174, -84.73934209226933)
    assert get_loc('Alpena  MI') == (45.08583, -83.46410999999999)
    logger.debug('testing Geocoder transformer ... ')

    # tests data frame input with noisy data will not fail
    df = pd.DataFrame([29412, 49707, 23941, 90210, 9999999, 98105, None, ''], columns=['zipcode'])
    g = GeoCoder(input_col='zipcode', remainder='passthrough')
    logger.debug('tranformer output on only zipcodes ... ')
    logger.debug(g.fit_transform(df))

    # tests inside a pipeline
    pipe = Pipeline(steps=[('g', GeoCoder('zipcode')),('imp', SimpleImputer(strategy='mean'))]).fit(df)
    assert np.sum(np.isnan(pipe.transform(df))) == 0

    # tests pandas data frame input with passthrough
    df = pd.DataFrame(['Alpena, MI', 49707,'Detroit, MI', 29412], columns=['zipcode'])
    g = GeoCoder(input_col='zipcode', remainder='passthrough')
    logger.debug('tranformer output on only zipcodes ... ')
    assert g.fit_transform(df).isna().any().sum() == 0

    #tests when input is pandas series
    g = GeoCoder()
    s = pd.Series(['Alpena, MI', 49707,'Detroit, MI', 29412])
    assert np.sum(np.isnan(g.transform(s))) == 0

    #test when input is a list
    g = GeoCoder()
    s = ['Alpena, MI', 49707,'Detroit, MI', 29412]
    assert np.sum(np.isnan(g.transform(s))) == 0
    logger.info('locations.py module test completed')

#_locations_test()
