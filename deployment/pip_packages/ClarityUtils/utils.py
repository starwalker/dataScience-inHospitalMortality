import json
import logging
import re
import pkg_resources
import pickle
from ClarityUtils import setup_logger
logger = logging.getLogger('ClarityUtils.utils')
logger.setLevel(logging.WARNING)

# loads in dictionaries from package resources
path = 'resources/pharm_class.json'
stream = pkg_resources.resource_stream('ClarityUtils', path)
pharm_dict = json.load(stream)

# creates reverse lookup dictionary
pharm_dict_rev = {v: k for k, v in pharm_dict.items()}

path = 'resources/pharm_cv.pkl'
stream = pkg_resources.resource_stream('ClarityUtils', path)
cv = pickle.load(stream)

# loads neighest neighor to go from text pharma class to index
path = 'resources/pharm_nn.pkl'
stream = pkg_resources.resource_stream('ClarityUtils', path)
nn = pickle.load(stream)

def _string_to_int(x, na_val=''):
    '''
    get a floating number with optional decimal from a string
    :param x: string, float or int
    :param x: string, float int or None, value returned when no digits are found
    :return: string
    '''
    try:
        output = int(x)
    except:
        try:
            pattern = '(?=[^\d])*\d+[.]?\d*'
            v = re.findall(pattern, x)
            output = int(v[0])
        except:
            output = na_val
    return output


def int_to_pharm_class(txt, na_val=''):
    try:
        key = _string_to_int(txt, na_val=na_val)
        val = pharm_dict_rev[key]
    except KeyError:
        logger.debug('missing  {}'.format(txt))
        val = na_val
    except TypeError:
        logging.debug('could not convert {} to int'.format(txt))
        val = na_val
    return val


def pharm_class_to_int(txt, na_val=0):
    try:
        features = cv.transform([txt])
        index = nn.kneighbors(features, 1, return_distance=False)[0][0]
        output = list(pharm_dict.items())[index][1]
    except TypeError:
        output = na_val
    return output


def get_pharm_class(txt, na_val=''):
    try:
        features = cv.transform([txt])
        index = nn.kneighbors(features, 1, return_distance=False)[0][0]
        output = list(pharm_dict.items())[index][0]
    except TypeError:
        output = na_val
    return output

def _utils_test():
    logger.info('testing int_to_pharm_class ... ')
    x = int_to_pharm_class(985)
    assert x == 'ARTV NUCLEOSIDE,NUCLEOTIDE,NON-NUCLEOSIDE RTI COMB'
    logger.info('testing pharm_class_to_int ... ')
    x = pharm_class_to_int('STEROID STRUCTURE, DIETARY SUPPLEMENT, MISC.')
    assert x == 986
    for k, v in pharm_dict.items():
        if pharm_class_to_int(int_to_pharm_class(v)) != v:
            print(k, v)
            print(pharm_dict[k])
            print('"{0}" mistaken for "{1}"'.format(pharm_dict_rev[v], int_to_pharm_class(v)))
    print('utils test completed')