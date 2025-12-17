# -*- coding: utf-8 -*-
#----------------------------------------------------
# © 2019 Epic Systems Corporation.
# Chronicles® is a registered trademark of Epic Systems Corporation.
#---------------------------------------------------
"""
    This module providers common converters to use along with Parcel.unpack_input() in order to
    preprocess input columns into a known format.

    For converters that call `add_warning` a warning message will log which rows produced a warning

    For converters that call `add_error` a warning message will log which rows produced a warning,
    and those rows will be removed from the prediction set.

    Examples
    --------

    The core APIs exposed by this module are:
        TimeseriesConverter(data_truncation_log_level='warn')
    This returns a timeseries converter to be used with PAF columns based on R PAF 82140 and R PAF 82141.
    The converted timeseries will be stored as a list of (timestamp, value) tuples.

        BooleanConverter(truthy=iterrable)
    This returns a boolean converter that can be used to convert string values into True/False values based
    on truthy values. By default, the following values are considered truthy (and all other map to False):
        "1", "True", "true", "t", "Yes", "yes", "y"

        InterconnectMissingValueConverter(default=None, log_level='ignore')
    This returns a converter that can be used for custom handling of missing Interconnect values (passed as {})
    to convert to a specific default value, and potentially log an error if log_level is set to "warn" or "error"

    We use the numpy docstring standard, for questions refer to:
         https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
"""
from abc import ABC, abstractmethod
from collections import Iterable
import pandas as pd
import warnings

class ParcelConversionWarning(RuntimeWarning):
    pass

class ParcelConversionErrorWarning(RuntimeWarning):
    pass

class Converter(ABC):
    """
        Converter abstract base class, inherit from this if you are writing your own class.

        Attributes
        ----------
        warnings : list
            a list of warnings generated during conversion

        Methods
        -------
        convert(series) : series
            converts one pandas.Series into another
    """
    ERROR_LEVELS = ('warn', 'error', 'ignore')
    WARNING_MESSAGE = "{}\n Affected rows:\n {}"
    ERROR_MESSAGE = "{}\n Affected rows will be dropped:\n {}"

    def __init__(self):
        self.warnings = []
        self.errors = []
        super().__init__()

    def add_warning(self, idx, msg):
        """
            used by subclasses to add warnings to self.warnings for use by log_warnings()

            Parameters
            ----------
            idx : pandas.index
                series index of conversion warning
        
            msg : string
                description of conversion warning to log.

        """
        self.warnings.append((idx, msg))

    def add_error(self, idx, msg):
        """
            used by subclasses to add errors to self.errors for use by log_issues()
        
            Parameters
            ----------
            idx : pandas.index
                series index of conversion errors

            msg : string
                description of conversion error to log.

        """
        self.errors.append((idx, msg))
    
    def get_error_index(self):
        """
            Returns an pandas index for all conversion errors, regaurdless of message.

            Returns
            -------
            index
                index locations of values in series whose conversions had errors.
        """
        errors = pd.Index([])
        for idx, msg in self.errors:
            errors = errors.union(idx)
        return errors
    
    def get_warning_index(self):
        """
            Returns a pandas index for conversion warnings, regaurdless of message.

            Returns
            -------
            index
                index locations of values in series whose conversions had warnings.
        """
        errors = pd.Index([])
        for idx, msg in self.warnings:
            errors = errors.union(idx)
        return errors

    def log_issues(self):
        """
            Logs warnings and errors to stout using the warnings.warn function
        """
        for index, message in self.warnings:
            warnings.warn(self.WARNING_MESSAGE.format(message, list(index)), ParcelConversionWarning)

        for index, message in self.errors:
            warnings.warn(self.ERROR_MESSAGE.format(message, list(index)), ParcelConversionErrorWarning)
    
    @classmethod
    def assert_log_level(cls, level):
        if level not in cls.ERROR_LEVELS:
            raise ValueError(f"'level' only accepts values in {cls.ERROR_LEVELS}")

    @abstractmethod
    def convert(self, series):
        """
            Converts a pandas series to a new series, and records issues for use by log_issues().
        
            Parameters
            ----------
            series : pandas.Series
                a pandas series to be converted
        
            Returns
            -------
            pandas.Series
                the converted pandas series
        """
        pass
    

class TimeseriesConverter(Converter):
    """
        Converts timeseries delimited data from PAF columns based on R PAF 82140 and R PAF 82141 into
        [(timestamp, value), (timestamp, value), ...] for use in modeling.

        Methods
        -------
        convert(self, series) : series
            returns a pandas series converted from the original series. logs warnings that should be displayed in Radar.
    """
    INVALID_FORMAT_ERROR = "Invalid format for timeseries."
    DATA_TRUNCATION_ERROR = "Truncated timeseries due to Chronicles max string limit."

    def __init__(self, data_truncation_log_level='error', item_sep='\t', time_sep='\x1d'):
        """
        Parameters
        ----------
        log_level: str, optional
            level at which to log truncated data errors, by default 'error'
            "error"  - a warning message is loged to stdout, and the errored row is removed from the prediction
            "warn"   - a warning mmessge is loged to stdout, but the row is still allowed for prediction
            "ignore" - the chronicles truncation flag is ignored entirely
        item_sep : str, optional
            item delimiter in timeseries string, by default '\t'
        time_sep : str, optional
            time/value delimiter, by default '\x1d'

        Note: This class expects the input string to start with a single 1 or 0 character followed by the item_sep delimiter.
        A leading 1 means the timeseries is complete, while a 0 indicates that the value was truncated by a max string limit on the server.
        """
        self.assert_log_level(data_truncation_log_level)

        self.log_level = data_truncation_log_level
        self.item_sep = item_sep
        self.time_sep = time_sep
        super().__init__()

    def _get_converter(self):
        """
        Returns the default converter for a timeseries column
        """
        import dateutil.parser

        def wrapped_converter(string):
            if not isinstance(string, str):
                return ValueError(f"Invalid Timeseries Data: {string}")
            series = []
            items=string.split(self.item_sep)
            for item in items[1:]: # The 0-th element indicates if data was complete (1) or truncated (0)
                try:
                    split_item = item.split(self.time_sep)
                    split_item[0]=dateutil.parser.parse(split_item[0])
                    series.append(tuple(split_item))
                except Exception as error:
                    return error
            return series
        return wrapped_converter

    def convert(self, series):
        """
        Converts a pandas series into a series of (timestamp, value) pairs.

        Parameters
        ----------
        series : pandas series
            series to be converted

        Returns
        -------
        series
            pandas series of converted timeseries data
        """
        # need to handle interconnect nulls sent at empty dicts {}
        series[series == {}] = "1"

        # check if any timeseres were truncated
        max_string_rows = series.index[series.str.startswith("0")] #index of truncated columns

        if len(max_string_rows) > 0:
            if self.log_level == 'warn':
                self.add_warning(max_string_rows, self.DATA_TRUNCATION_ERROR)
            elif self.log_level == 'error':
                self.add_error(max_string_rows, self.DATA_TRUNCATION_ERROR)

        series = series.apply(self._get_converter())
        
        series_error = series.apply(lambda x : True if isinstance(x, Exception) else False)
        error_rows = series_error.index[series_error == True]
        series[series_error == True] = None
        if len(error_rows) > 0:
            self.add_error(error_rows, self.INVALID_FORMAT_ERROR)
        
        return series


class BooleanConverter(Converter):
    """
        Converts string values to boolean values given a mapping to use for truthy values.

        Methods
        -------
        convert(self, series) : series
            returns a pandas series converted from the original series, missing values are always False.
    """

    def __init__(self, truthy=None):
        """
        Parameters
        ----------
        truthy : list, optional
            list of values to be considered true, by default None
        """
        if truthy is None or not isinstance(truthy, Iterable):
            truthy = [ True, 1, '1', 'TRUE', 'True', 'true', 'T', 't', 'YES', 'Yes', 'yes', 'Y', 'y']
        self.truthy_values = truthy
        super().__init__()

    def _get_converter(self):
        """
        Conversion function to use in pandas.series.apply
        """
        def wrapped_converter(value):
            if value in self.truthy_values:
                return True
            return False

        return wrapped_converter

    def convert(self, series):
        """
        Converts a pandas series into a series of boolean values.

        Parameters
        ----------
        series : pandas series
            series to be converted

        Returns
        -------
        series
            pandas series of converted boolean values
        """
        return series.apply(self._get_converter())


class InterconnectMissingValueConverter(Converter):
    """
        Converts the InterconnectMissing value {} to given default value.
        If logging level is set, additionally logs warnings/errors.
        In the error case, rows with the InterconnectMissing

        Methods
        -------
        convert(self, series) : series
            returns a pandas series converted from the original series
    """

    def __init__(self, default=None, *, level='ignore'):
        """
        Parameters
        ----------
        default : optional
            value to be used in place of interconnect missing values {}
        level : str, optional, default 'warning'
            'ignore' to just replace, 
            'warning' to replace and log warnings, 
            'error' to log errors and remove from prediction set
        """
        Converter.assert_log_level(level)

        self.default = default
        self.log_level = level
        super().__init__()

    def convert(self, series):
        """
        Converts a pandas series into a new series with interconnect missing values replaced

        Parameters
        ----------
        series : pandas series
            series to be converted

        Returns
        -------
        series
            pandas series of updated values
        """
        ic_missing = series.index[series == {}]
        series[ic_missing] = self.default
        if self.log_level == 'warn':
            self.add_warning(ic_missing, f"Value for {series.name} replaced by {self.default}")
        elif self.log_level == 'error':
            self.add_error(ic_missing, f"Values missing for {series.name}")
        return series