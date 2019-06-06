import pytest
import timeit

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.tektronix.DPO7200xx import TektronixDPO7000xx

visalib = sims.__file__.replace('__init__.py', 'Tektronix_DPO7200xx.yaml@sim')


@pytest.fixture(scope='module')
def tektronix_dpo():
    """
    A six channel-per-relay instrument
    """
    driver = TektronixDPO7000xx(
        'dpo', address='TCPIP0::0.0.0.0::inst0::INSTR', visalib=visalib)

    yield driver
    driver.close()


def test_adjust_timer(tektronix_dpo):
    """
    After adjusting the type of the measurement or the source of the
    measurement, we need wait at least 0.1 seconds
    ('minimum_adjustment_time') before a measurement value can be
    retrieved. Test this.
    """

    timer = timeit.Timer(
        'tektronix_dpo.measurement[0].source1("CH1"),'
        'tektronix_dpo.measurement[0].amplitude()',
        globals=locals(),
    )
    min_time = tektronix_dpo.measurement[0]._minimum_adjustment_time
    repeats = timer.repeat(repeat=10, number=1)

    # The minimum time should be at least 95% of the 'minimum_adjustment_time'
    assert all(t > min_time * 0.95 for t in repeats)
    # To see why this fudge factor is necessary, try the following:
    # >>> import time
    # >>> import timeit
    # >>> timer = timeit.Timer("time.sleep(1E-3)")
    # >>> print(any(t < 1E-3 for t in timer.repeat(repeat=100, number=1)))
    # ... True
    # Conclusion: the command 'time.sleep(1E-3)' sometimes takes less
    # than 1E-3 seconds to return. Since the sleep time is not critical
    # to the microsecond, we don't care that we sometimes retrieve
    # measurements slightly sooner then 'minimum_adjustment_time'
