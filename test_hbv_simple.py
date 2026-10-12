"""Hourly HBV integration, unit conversion and workflow regression checks."""
import numpy as np
import pandas as pd
import pytest

from src.app import dash_analysis as analysis
from src.hydro_calc import calc_nse
from src.models.registry import ModelRegistry


def forcing():
    return np.tile([0., 12., 3., 0., 0., 1.], 8), np.full(48, .1)


def test_hourly_discharge_conversion_preserves_runoff_volume():
    model = ModelRegistry.get_model('HBV Model')
    rain, evap = forcing()
    daily = model.run(rain, evap, model.default_params, {'area': 100, 'timestep': 'daily'})
    hourly = model.run(rain, evap, model.default_params, {'area': 100, 'timestep': 'hourly'})
    assert model.supports_hourly
    assert np.any(hourly > 0)
    np.testing.assert_allclose(hourly, daily * 24)
    np.testing.assert_allclose(hourly.sum() * 3600, daily.sum() * 86400)
    # Independent first-step result from the existing soil / dual-reservoir equations.
    p = model.default_params
    soil = .5 * p['fc'] - .1 * min(.5 / p['lp'], 1)
    recharge = 0.  # First precipitation value is zero.
    assert hourly[0] == recharge
    rain_one = np.array([12.])
    soil += 12
    recharge = 12 * (soil / p['fc']) ** p['beta']
    expected = recharge * (.7 * p['k0'] + .3 * p['k1']) * 100 * 1000 / 3600
    assert model.run(rain_one, np.array([.1]), p, {'area':100,'timestep':'hourly'})[0] == pytest.approx(expected)


@pytest.mark.parametrize('parameter_mode', ['import', 'calibrate'])
def test_hourly_dash_pipeline_uses_simple_hbv(parameter_mode):
    model = ModelRegistry.get_model('HBV Model')
    rain, evap = forcing()
    observed = model.run(rain, evap, model.default_params, {'area':100,'timestep':'hourly'})
    frame = pd.DataFrame({'date':pd.date_range('2020-01-01',periods=len(rain),freq='h'),
                          'precip':rain,'evap':evap,'flow':observed})
    config = dict(mode=analysis.UPLOAD_MODES[0],area=100,timestep='hourly',warmup_hours=3,
                  parameter_mode=parameter_mode,models=[model.name],algorithm='PSO',max_iter=5,
                  algo_params={'n_particles':10},routing=False,
                  mapping={key:key for key in ('date','precip','evap','flow','upstream')})
    prepared = analysis.preprocess([('hourly.csv',frame.to_csv(index=False).encode())],config)
    params = {model.name:pd.DataFrame([model.default_params]).to_csv(index=False).encode()}
    simulation = analysis.simulate(prepared,config,params)
    assert not simulation['warnings']
    assert set(simulation['parameters'][model.name]) == {'fc','beta','k0','k1','lp'}
    assert simulation['metrics'][0]['模型'] == model.name
    values = simulation['file_results'][model.name]['hourly.csv']['simulated']
    assert len(values) == len(observed)
    assert np.isfinite(values).all() and np.any(values > 0)
    assert simulation['metrics'][0]['NSE'] == pytest.approx(calc_nse(observed[3:],values[3:]))
    if parameter_mode == 'import':
        np.testing.assert_allclose(values, observed)
        assert simulation['metrics'][0]['NSE'] == pytest.approx(1)


def test_registry_exposes_one_hourly_capable_hbv():
    assert [name for name in ModelRegistry.list_models() if 'HBV' in name] == ['HBV Model']
    model = ModelRegistry.get_model('HBV Model')
    assert model.supports_hourly
    assert set(model.param_bounds) == {'fc','beta','k0','k1','lp'}
    assert model.get_timestep_hours({'timestep':'hourly'}) == 1
