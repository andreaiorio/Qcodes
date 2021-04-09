import os
import glob
import pandas as pd
import numpy as np
import logging

from qcodes import (
    Instrument,
    ManualParameter,
    validators as vals
)
from time import time
from datetime import datetime

log = logging.getLogger(__name__)

class LeidenFridge(Instrument):

    def __init__(self, name, log_folder='C:\\LC Data\Data\\', log_name='LogAVS', log_dateformat='%Y-%m-%d %H:%M:%S', update_interval=3, **kwargs):
        super().__init__(name, **kwargs)
        self.log_folder = log_folder
        self.log_name = log_name
        self.log_dateformat = log_dateformat
        
        self.add_parameter('log_file', 
                           parameter_class=ManualParameter)
        
        self.add_parameter('log_file_last_update', 
                           parameter_class=ManualParameter)
                 
        self.add_parameter('update_interval', 
                           unit='s',
                           initial_value=update_interval,
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(min_value=0))
        
        self._generate_dataframes()

        for _i, _ch in enumerate(['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9']):
            self.add_parameter(_ch, 
                               unit='mK',
                               label=self.T_names[_i],
                               get_cmd=self.get_T(_ch),
                               get_parser=float)
 
        super().snapshot(update=True)
        self.connect_message()

    def get_T(self, channel):
        def get_cmd():
            self._update()
            return self.T[channel].tail(1).values
        
        return get_cmd

    def get_idn(self):
        return {'vendor': 'Leiden Cryogenics',
                'model': self.name,
                'serial': None, 'firmware': '0.1'}
    
    def snapshot(self, update=False):
        _time_since_update = time() - self._last_temp_update
        if _time_since_update > (self.update_interval()):
            return super().snapshot(update=True)
        else:
            return super().snapshot(update=update)

    
    def _update(self):
        _time_since_update = time() - self._last_temp_update
        if _time_since_update > self.update_interval():
            self._generate_dataframes()

    def _generate_dataframes(self):
        self._last_temp_update = time()
        self._last_file = max(glob.glob(self.log_folder+'/'+self.log_name+'*.dat'), key=os.path.getmtime)
        self.df = pd.read_csv(self._last_file, 
                              skiprows=[0,1], 
                              delimiter='\t', 
                              index_col=0,  
                              header=[0,1,2], 
                              date_parser=lambda date: datetime.strptime(date, self.log_dateformat))
        
        self.T = self.df[['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9']]
        self.R = self.df[['R0','R1','R2','R3','R4','R5','R6','R7','R8','R9']]
        self.heaters = self.df[['I0', 'I1', 'I2', 'I3']]
        self.T_names = self.T.columns.get_level_values(level=1).to_list()
        self.T_calibrations = self.T.columns.get_level_values(level=2).to_list()

        self.log_file(self._last_file)
        self.log_file_last_update(self.df.tail(1).index.strftime(self.log_dateformat).values[0])
