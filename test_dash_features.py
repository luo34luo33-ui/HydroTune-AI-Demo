"""Behavior checks for the Dash service and its offline artifacts."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from src.app import dash_analysis
from src.app.basin_overview import build_basin_overview, prepared_overview_image
from src.app.html_snapshot import build_snapshot
from src.llm_api import get_llm_config, llm_settings
from src.models.registry import ModelRegistry


def _analysis_inputs():
    n = 32
    files = []
    for index in range(4):
        frame = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n),
            'precip': np.where(np.arange(n) % 7 < 3, 5, 0),
            'evap': np.full(n, .2),
            'flow': np.linspace(2, 15, n) + index,
        })
        files.append((f'event{index}.csv', frame.to_csv(index=False).encode()))
    params = pd.DataFrame([ModelRegistry.get_model('Tank Model').default_params])
    config = {'area': 150.7944, 'timestep': 'daily', 'warmup_hours': 0,
              'parameter_mode': 'import', 'routing': False, 'xgb': False,
              'mapping': {name: name for name in ('date', 'precip', 'evap', 'flow', 'upstream')},
              'llm_settings': {'provider': 'minimax', 'api_key': 'SECRET-FOR-TEST'}}
    return files, {'Tank Model': params.to_csv(index=False).encode()}, config


@pytest.mark.parametrize('mode,count', [(dash_analysis.UPLOAD_MODES[0], 1),
                                        (dash_analysis.UPLOAD_MODES[1], 1),
                                        (dash_analysis.UPLOAD_MODES[2], 4)])
def test_uploaded_modes_keep_real_results(mode, count):
    files, parameters, config = _analysis_inputs()
    with patch.object(dash_analysis, 'call_minimax', return_value='[ERROR] offline'):
        result = dash_analysis.analyze(files[:count], {**config, 'mode': mode}, parameters)
    assert len(result['charts']) == count
    assert len([row for row in result['metrics'] if row['模型'] == 'Tank Model']) == count
    assert result['parameters']['Tank Model']
    assert 'SECRET-FOR-TEST' not in build_snapshot(result)


def _basin_files():
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        base = Path(directory, 'basin.shp')
        gpd.GeoDataFrame(geometry=[box(.2, .2, .8, .8)], crs='EPSG:4326').to_file(base)
        packed = BytesIO()
        with ZipFile(packed, 'w') as archive:
            for item in Path(directory).glob('basin.*'):
                archive.write(item, item.name)
        boundary = packed.getvalue()
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=10, height=10, count=1,
                         dtype='float32', crs='EPSG:4326',
                         transform=from_origin(0, 1, .1, .1), nodata=-9999) as dem:
            dem.write(np.arange(100, dtype='float32').reshape(10, 10), 1)
        raster = memory.read()
    return boundary, raster


def test_basin_overview_and_reserved_image():
    boundary, raster = _basin_files()
    result = build_basin_overview(boundary, raster)
    assert result['area_km2'] > 0
    assert result['min_m'] == 22
    assert result['max_m'] == 77
    assert len(result['figure']['data']) == 2
    picture = BytesIO()
    Image.new('RGB', (2, 2)).save(picture, format='PNG')
    assert prepared_overview_image(picture.getvalue(), 'image/png')['src'].startswith('data:image/png;base64,')


def test_basin_requires_shapefile_components():
    boundary, raster = _basin_files()
    without_projection = BytesIO()
    with ZipFile(BytesIO(boundary)) as source, ZipFile(without_projection, 'w') as target:
        for name in source.namelist():
            if not name.endswith('.prj'):
                target.writestr(name, source.read(name))
    with pytest.raises(ValueError, match='配套文件'):
        build_basin_overview(without_projection.getvalue(), raster)


def test_llm_settings_are_scoped_to_one_analysis():
    with llm_settings({'provider': 'minimax', 'api_key': 'visitor-one'}):
        assert get_llm_config()['api_key'] == 'visitor-one'
        with llm_settings({'provider': 'minimax', 'api_key': 'visitor-two'}):
            assert get_llm_config()['api_key'] == 'visitor-two'
        assert get_llm_config()['api_key'] == 'visitor-one'


def test_imported_routing_and_xgb_use_existing_backend():
    files, parameters, config = _analysis_inputs()
    frame = pd.read_csv(BytesIO(files[0][1]))
    frame['upstream'] = 1.0
    event = ('event0.csv', frame.to_csv(index=False).encode())
    config.update(mode=dash_analysis.UPLOAD_MODES[0], routing=True,
                  routing_parameters={'k_routing': 2.5, 'x_routing': .25},
                  xgb=True, xgb_estimators=50)
    with patch.object(dash_analysis, 'call_minimax', return_value='[ERROR] offline'):
        result = dash_analysis.analyze([event], config, parameters)
    assert 'k_routing' in result['parameters']['Tank Model']
    assert any(row['模型'] == 'Tank Model-XGB' for row in result['metrics'])
    assert len(result['corrected_flow']['event0.csv']['corrected']) == len(frame)


@pytest.mark.parametrize('transform,crs,values,problem', [
    (from_origin(10, 11, .1, .1), 'EPSG:4326', np.ones((10, 10)), '不相交'),
    (from_origin(0, 1, .1, .1), None, np.ones((10, 10)), '坐标系'),
    (from_origin(0, 1, .1, .1), 'EPSG:4326', np.full((10, 10), -9999), '有效'),
])
def test_basin_reports_input_problems(transform, crs, values, problem):
    boundary, _ = _basin_files()
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=10, height=10, count=1,
                         dtype='float32', crs=crs, transform=transform, nodata=-9999) as dem:
            dem.write(values.astype('float32'), 1)
        raster = memory.read()
    with pytest.raises(ValueError, match=problem):
        build_basin_overview(boundary, raster)
