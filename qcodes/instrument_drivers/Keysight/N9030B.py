import numpy as np
from typing import Any, Tuple, Dict, Union

from qcodes import (VisaInstrument, InstrumentChannel, Parameter,
                    ParameterWithSetpoints, ChannelList)
from qcodes.instrument.parameter import ParamRawDataType
from qcodes.utils.validators import Enum, Numbers, Arrays, Ints
from qcodes.utils.helpers import create_on_off_val_mapping


class FrequencyAxis(Parameter):

    def __init__(self, start: float, stop: float, npts: int, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start: float = start
        self._stop: float = stop
        self._npts: int = npts

    def get_raw(self) -> ParamRawDataType:
        return np.linspace(self._start, self._stop, self._npts)


class Trace(ParameterWithSetpoints):

    def __init__(self, number: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instrument: Union["SpectrumAnalyzerMode", "PhaseNoiseMode"]
        self.root_instrument: "N9030B"

        self.number = number

    def get_raw(self) -> ParamRawDataType:
        return self.instrument._get_data(trace_num=self.number)


class SpectrumAnalyzerMode(InstrumentChannel):
    """
    Spectrum Analyzer Mode for Keysight N9030B instrument.
    """
    def __init__(self, parent: "N9030B", name: str, *arg: Any, **kwargs: Any):
        super().__init__(parent, name, *arg, **kwargs)

        self.mode("SA")

        self._min_freq = 2
        self._max_freq = 50e9

        self.add_parameter(
            name="start",
            get_cmd=":SENSe:FREQuency:STARt?",
            set_cmd=self._set_start,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10),
            docstring="start frequency for the sweep"
        )

        self.add_parameter(
            name="stop",
            get_cmd=":SENSe:FREQuency:STOP?",
            set_cmd=self._set_stop,
            get_parser=float,
            vals=Numbers(self._min_freq + 10, self._max_freq),
            docstring="stop frequency for the sweep"
        )

        self.add_parameter(
            name="center",
            get_cmd=":SENSe:FREQuency:CENTer?",
            set_cmd=self._set_center,
            get_parser=float,
            vals=Numbers(self._min_freq + 5, self._max_freq - 5),
            docstring="Sets and gets center frequency"
        )

        self.add_parameter(
            name="span",
            get_cmd=":SENSe:FREQuency:SPAN?",
            set_cmd=self._set_span,
            get_parser=float,
            vals=Numbers(10, self._max_freq - self._min_freq),
            docstring="Changes span of frequency"
        )

        self.add_parameter(
            name="npts",
            get_cmd=":SENSe:SWEep:POINts?",
            set_cmd=self._set_npts,
            get_parser=int,
            vals=Ints(1, 20001),
            docstring="Number of points for the sweep"
        )

        self.add_parameter(
            name="sweep_time",
            label="Sweep time",
            get_cmd=":SENSe:SWEep:TIME?",
            get_parser=float,
            unit="s",
            docstring="gets sweep time"
        )

        self.add_parameter(
            name="auto_sweep_time_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TIME:AUTO?",
            set_cmd=self._enable_auto_sweep_time,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="enables auto sweep time"
        )

        self.add_parameter(
            name="auto_sweep_type_enabled",
            initial_value=False,
            get_cmd=":SENSe:SWEep:TYPE:AUTO?",
            set_cmd=self._enable_auto_sweep_type,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="enables auto sweep type"
        )

        self.add_parameter(
            name="sweep_type",
            get_cmd=":SENSe:SWEep:TYPE?",
            set_cmd=self._set_sweep_type,
            val_mapping={
                "fft": "FFT",
                "sweep": "SWE",
            },
            docstring="Sets up sweep type. Possible options are 'fft' and "
                      "'sweep'."
        )

        self.add_parameter(
            name="freq_axis",
            label="Frequency",
            unit="Hz",
            start=self.start,
            stop=self.stop,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis,
            docstring="Sets frequency axis for the sweep."
        )

        self.add_parameter(
            name="trace",
            label="Trace",
            unit="dB",
            number=1,
            vals=Arrays(shape=(self.npts.get_latest,)),
            setpoints=(self.freq_axis,),
            parameter_class=Trace,
            docstring="Gets trace data."
        )

    def _set_start(self, val: float) -> None:
        """
        Sets start frequency
        """
        stop = self.stop()
        if val >= stop:
            raise ValueError(f"Start frequency must be smaller than stop "
                             f"frequency. Provided start freq is: {val} Hz and "
                             f"set stop freq is: {stop} Hz")

        self.write(f":SENSe:FREQuency:STARt {val}")

        start = self.start()
        if abs(val - start) >= 1:
            self.log.warning(
                f"Could not set start to {val} setting it to {start}"
            )

    def _set_stop(self, val: float) -> None:
        """
        Sets stop frequency
        """
        start = self.start()
        if val <= start:
            raise ValueError(f"Stop frequency must be larger than start "
                             f"frequency. Provided stop freq is: {val} Hz and "
                             f"set start freq is: {start} Hz")

        self.write(f":SENSe:FREQuency:STOP {val}")

        stop = self.stop()
        if abs(val - stop) >= 1:
            self.log.warning(
                f"Could not set stop to {val} setting it to {stop}"
            )

    def _set_center(self, val: float) -> None:
        """
        Sets center frequency and updates start and stop frequencies if they
        change.
        """
        self.write(f":SENSe:FREQuency:CENTer {val}")
        self.update_trace()

    def _set_span(self, val: float) -> None:
        """
        Sets frequency span and updates start and stop frequencies if they
        change.
        """
        self.write(f":SENSe:FREQuency:SPAN {val}")
        self.update_trace()

    def _set_npts(self, val: int) -> None:
        """
        Sets number of points for sweep
        """
        self.write(f":SENSe:SWEep:POINts {val}")

    def _enable_auto_sweep_time(self, val: str) -> None:
        """
        Enables auto sweep time
        """
        self.write(f":SENSe:SWEep:TIME:AUTO {val}")

    def _enable_auto_sweep_type(self, val: str) -> None:
        """
        Enables auto sweep type
        """
        self.write(f":SENSe:SWEep:TYPE:AUTO {val}")

    def _set_sweep_type(self, val: str) -> None:
        """
        Sets sweep type
        """
        self.write(f":SENSe:SWEep:TYPE {val}")

    def _get_data(self, trace_num: int) -> ParamRawDataType:
        """
        Gets data from the measurement.
        """
        self.cont_meas("OFF")
        try:
            timeout = self.sweep_time() + self._additional_wait
            with self.timeout.set_to(timeout):
                data_str = self.ask(f":READ:{self.measurement}{trace_num}?")
                data = np.array(data_str.rstrip()).astype("float64")
        finally:
            self.cont_meas("ON")

        return data

    def update_trace(self) -> None:
        """
        Updates start and stop frequencies whenever span of/or center frequency
        is updated.
        """
        self.start()
        self.stop()

    def setup_swept_sa_sweep(self,
                             start: float,
                             stop: float,
                             npts: int) -> None:
        """
        Sets up the Swept SA measurement sweep for Spectrum Analyzer Mode.
        """
        if "SANalyzer" in self._available_meas():
            self.measurement("SANalyzer")
        else:
            raise RuntimeError("Swept SA measurement is not available on your "
                               "Keysight N9030B instrument with Spectrum "
                               "Analyzer mode.")
        self.start(start)
        self.stop(stop)
        self.npts(npts)

    def autotune(self) -> None:
        """
        Autotunes frequency
        """
        self.write(":SENS:FREQuency:TUNE:IMMediate")


class PhaseNoiseMode(InstrumentChannel):
    """
    Phase Noise Mode for Keysight N9030B instrument.
    """

    def __init__(self, parent: "N9030B", name: str, *arg: Any, **kwargs: Any):
        super().__init__(parent, name, *arg, **kwargs)

        self.mode("PNOISE")

        self._min_freq = 1
        self._valid_max_freq: Dict[str, float] = {"503": 3699999995,
                                                  "508": 8499999995,
                                                  "513": 13799999995,
                                                  "526": 26999999995}
        opt: str
        for hw_opt_for_max_freq in self._valid_max_freq.keys():
            if hw_opt_for_max_freq in self._options():
                opt = hw_opt_for_max_freq
        self._max_freq = self._valid_max_freq[opt]

        self.add_parameter(
            name="npts",
            get_cmd=":SENSe:LPLot:SWEep:POINts?",
            set_cmd=":SENSe:LPLot:SWEep:POINts {}",
            get_parser=int,
            vals=Ints(601, 20001),
            docstring="Number of points for the sweep"
        )

        self.add_parameter(
            name="start_offset",
            get_cmd=":SENSe:LPLot:FREQuency:OFFSet:STARt?",
            set_cmd=self._set_start_offset,
            get_parser=float,
            vals=Numbers(self._min_freq, self._max_freq - 10),
            docstring="start frequency offset for the plot"
        )

        self.add_parameter(
            name="stop_offset",
            get_cmd=":SENSe:LPLot:FREQuency:OFFSet:STOP?",
            set_cmd=self._set_stop_offset,
            get_parser=float,
            vals=Numbers(self._min_freq + 100, self._max_freq),
            docstring="stop frequency offset for the plot"
        )

        self.add_parameter(
            name="freq_axis",
            label="Frequency",
            unit="Hz",
            start=self.start_offset,
            stop=self.stop_offset,
            npts=self.npts,
            vals=Arrays(shape=(self.npts.get_latest,)),
            parameter_class=FrequencyAxis,
            docstring="Sets frequency axis for the sweep."
        )

        self.add_parameter(
            name="trace",
            label="Trace",
            unit="dB",
            number=1,
            vals=Arrays(shape=(self.npts.get_latest,)),
            setpoints=(self.freq_axis,),
            parameter_class=Trace,
            docstring="Gets trace data."
        )

    def _set_start_offset(self, val: float) -> None:
        """
        Sets start offset for frequency in the plot
        """
        stop_offset = self.stop_offset()
        self.write(f":SENSe:LPLot:FREQuency:OFFSet:STARt {val}")
        start_offset = self.start_offset()

        if abs(val - start_offset) >= 1:
            self.log.warning(
                f"Could not set start offset to {val} setting it to "
                f"{start_offset}"
            )
        if val >= stop_offset or abs(val - stop_offset) < 10:
            self.log.warning(f"Provided start frequency offset {val} Hz was "
                             f"greater than preset stop frequency offset "
                             f"{stop_offset} Hz. Provided start frequency "
                             f"offset {val} Hz is set and new stop freq offset"
                             f" is: {self.stop_offset()} Hz.")

    def _set_stop_offset(self, val: float) -> None:
        """
        Sets stop offset for frequency in the plot
        """
        start_offset = self.start_offset()
        self.write(f":SENSe:LPLot:FREQuency:OFFSet:STOP {val}")
        stop_offset = self.stop_offset()

        if abs(val - stop_offset) >= 1:
            self.log.warning(
                f"Could not set stop offset to {val} setting it to "
                f"{stop_offset}"
            )

        if val <= start_offset or abs(val-start_offset) < 10:
            self.log.warning(f"Provided stop frequency offset {val} Hz was "
                             f"less than preset start frequency offset "
                             f"{start_offset} Hz. Provided stop frequency "
                             f"offset {val} Hz is set and new start freq offset"
                             f" is: {self.start_offset()} Hz.")

    def _get_data(self, trace_num: int) -> ParamRawDataType:
        """
        Gets data from the measurement.
        """
        self.cont_meas("OFF")
        try:
            data_str = self.ask(f":READ:{self.measurement}{trace_num}?")
            data = np.array(data_str.rstrip()).astype("float64")
        finally:
            self.cont_meas("ON")

        return data

    def setup_log_plot_sweep(self,
                             start_offset: float,
                             stop_offset: float,
                             npts: int
                             ) -> None:
        """
        Sets up the Log Plot measurement sweep for Phase Noise Mode.
        """
        if "LPLot" in self._available_meas():
            self.measurement("LPLot")
        else:
            raise RuntimeError("Log Plot measurement is not available on your "
                               "Keysight N9030B instrument with Phase Noise "
                               "mode.")

        self.start_offset(start_offset)
        self.stop_offset(stop_offset)
        self.npts(npts)

    def autotune(self) -> None:
        """
        Autotunes frequency
        """
        self.write(":SENSe:FREQuency:CARRier:SEARch")


class N9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer.
    """

    CHANNEL_CLASS_1 = SpectrumAnalyzerMode
    CHANNEL_CLASS_2 = PhaseNoiseMode

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self._min_freq: float
        self._max_freq: float
        self._additional_wait: float = 1

        self.add_parameter(
            name="mode",
            get_cmd=":INSTrument:SELect?",
            set_cmd=":INSTrument:SELect {}",
            vals=Enum(*self._available_modes()),
            docstring="Allows setting of different modes present and licensed "
                      "for the instrument."
        )

        self.add_parameter(
            name="measurement",
            get_cmd=":CONFigure?",
            set_cmd=":CONFigure:{}",
            vals=Enum(*self._available_meas()),
            docstring="Sets measurement type from among the available "
                      "measurement types."
        )

        self.add_parameter(
            name="cont_meas",
            initial_value=False,
            get_cmd=":INITiate:CONTinuous?",
            set_cmd=self._enable_cont_meas,
            val_mapping=create_on_off_val_mapping(on_val="ON", off_val="OFF"),
            docstring="Enables or disables continuous measurement."
        )

        self.add_parameter(
            name="format",
            get_cmd=":FORMat:TRACe:DATA?",
            set_cmd=":FORMat:TRACe:DATA {}",
            val_mapping={
                "ascii": "ASCii",
                "int32": "INTeger,32",
                "real32": "REAL,32",
                "real64": "REAL,64"
            },
            docstring="Sets up format of data received"
        )

        if "SA" in self._available_modes():
            sa_mode = ChannelList(
                self, "sa", self.CHANNEL_CLASS_1, snapshotable=True
            )
            self.add_submodule("sa", sa_mode)
        else:
            self.log.info("Spectrum Analyzer mode is not available on this "
                          "instrument.")

        if "PNOISE" in self._available_modes():
            pnoise_mode = ChannelList(
                self, "pn", self.CHANNEL_CLASS_2, snapshotable=True
            )
            self.add_submodule("pn", pnoise_mode)
        else:
            self.log.info("Phase Noise mode is not available on this "
                          "instrument.")
        self.reset()
        self.connect_message()

    def _available_modes(self) -> Tuple[str, ...]:
        """
        Returns present and licensed modes for the instrument.
        """
        available_modes = self.ask(":INSTrument:CATalog?")
        return tuple(available_modes.split(','))

    def _available_meas(self) -> Tuple[str, ...]:
        """
        Gives available measurement with a given mode for the instrument
        """
        available_meas = self.ask(":CONFigure:CATalog?")
        return tuple(available_meas.split(','))

    def _enable_cont_meas(self, val: str) -> None:
        """
        Sets continuous measurement to ON or OFF.
        """
        self.write(f":INITiate:CONTinuous {val}")

    def _options(self) -> Tuple[str, ...]:
        """
        Returns installed options numbers.
        """
        options_raw = self.ask('*OPT?')
        return tuple(options_raw.split(','))

    def reset(self) -> None:
        """
        Reset the instrument by sending the RST command
        """
        self.write("*RST")

    def abort(self) -> None:
        """
        Aborts the measurement
        """
        self.write(":ABORt")
