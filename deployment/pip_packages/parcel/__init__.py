#----------------------------------------------------
# © 2017 - 2019 Epic Systems Corporation.
# Chronicles® is a registered trademark of Epic Systems Corporation.
#----------------------------------------------------
'''
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
'''
from .parcel import Parcel
__version__ = "0.2.10"

