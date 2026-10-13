"""Basin overview from a zipped Shapefile and a GeoTIFF DEM."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import plotly.graph_objects as go

TERRAIN_COLORS = [[0,'#2B783D'],[.2,'#79AB5A'],[.4,'#C6D596'],[.6,'#E8DBAD'],
                  [.78,'#BD9063'],[.9,'#8D6045'],[1,'#F8F8F8']]


def map_layout(figure, title, latitude):
    figure.update_layout(template='plotly_white',title=title,
                         xaxis=dict(title='经度 (°E)',tickformat='.3f',constrain='domain'),
                         yaxis=dict(title='纬度 (°N)',tickformat='.3f',scaleanchor='x',
                                    scaleratio=1/np.cos(np.deg2rad(latitude)),constrain='domain'),
                         showlegend=False,margin=dict(l=55,r=45,t=40,b=50))
    figure.add_annotation(x=.94,y=.89,xref='paper',yref='paper',text='',showarrow=True,
                          ax=0,ay=42,arrowhead=3,arrowsize=1.2,arrowwidth=2,arrowcolor='#111827')
    figure.add_annotation(x=.94,y=.99,xref='paper',yref='paper',text='<b>N</b>',showarrow=False,
                          xanchor='center',font=dict(size=15,color='#111827'))


def prepared_overview_image(content: bytes, mime_type: str) -> dict:
    """Reserved integration point for an existing basin overview image.

    The Dash upload form intentionally does not call this function yet.
    """
    if mime_type not in {'image/png', 'image/jpeg', 'image/webp'}:
        raise ValueError('概况图须为 PNG、JPEG 或 WebP。')
    if not content:
        raise ValueError('概况图为空。')
    from PIL import Image
    expected = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}[mime_type]
    try:
        with Image.open(BytesIO(content)) as picture:
            if picture.format != expected:
                raise ValueError('概况图内容与所选格式不符。')
            picture.verify()
    except OSError as exc:
        raise ValueError('概况图无法读取。') from exc
    return {'src': f'data:{mime_type};base64,{b64encode(content).decode("ascii")}',
            'mime_type': mime_type}


def build_basin_overview(shapefile_zip: bytes, dem_tiff: bytes = None) -> dict:
    import geopandas as gpd
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.mask import mask
    from rasterio.transform import array_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with TemporaryDirectory(prefix='.basin-', dir=Path(__file__).resolve().parents[2]) as directory, ZipFile(BytesIO(shapefile_zip)) as archive:
        entries = [name for name in archive.namelist() if not name.endswith('/')]
        if len(entries) > 100 or sum(item.file_size for item in archive.infolist()) > 250 * 1024 * 1024:
            raise ValueError('空间压缩包过大，最多 100 个文件 / 解压后 250 MB。')
        if any(PurePosixPath(name.replace(chr(92), '/')).is_absolute() or '..' in PurePosixPath(name.replace(chr(92), '/')).parts for name in entries):
            raise ValueError('空间压缩包包含不安全路径。')
        # Only extract the components used by the selected Shapefile.
        shapefiles = [name for name in entries if name.lower().endswith('.shp')]
        if len(shapefiles) != 1:
            raise ValueError('ZIP 中须包含一套 SHP 流域边界。')
        stem = shapefiles[0][:-4].lower()
        components = {PurePosixPath(name).suffix.lower(): name for name in entries
                      if name[:-len(PurePosixPath(name).suffix)].lower() == stem}
        missing = {'.shp', '.shx', '.dbf', '.prj'} - components.keys()
        if missing:
            raise ValueError('SHP 缺少配套文件：' + ', '.join(sorted(missing)))
        for suffix, name in components.items():
            Path(directory, 'basin' + suffix).write_bytes(archive.read(name))
        basin = gpd.read_file(Path(directory, 'basin.shp'))
        if basin.empty or basin.crs is None or basin.geometry.is_empty.all():
            raise ValueError('流域边界为空或缺少坐标系。')
        basin = basin[basin.geometry.notna() & ~basin.geometry.is_empty]
        if not basin.geometry.is_valid.all():
            raise ValueError('流域边界包含无效几何。')
        geom = basin.geometry.union_all()
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError('流域边界须为面要素。')
        area_km2 = float(gpd.GeoSeries([geom], crs=basin.crs).to_crs('EPSG:6933').area.iloc[0] / 1e6)

        if dem_tiff is None:
            boundary = basin.to_crs('EPSG:4326').geometry.union_all()
            polygons = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
            fig = go.Figure()
            for polygon in polygons:
                x, y = polygon.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', fill='toself',
                                         line={'color':'#111111','width':2},fillcolor='rgba(8,166,173,.08)', name='流域边界'))
            map_layout(fig,'流域边界',boundary.centroid.y)
            return {'figure': fig.to_plotly_json(), 'area_km2': area_km2, 'crs': str(basin.crs)}

        with MemoryFile(dem_tiff) as memory, memory.open() as dem:
            if dem.crs is None:
                raise ValueError('DEM 缺少坐标系。')
            local = basin.to_crs(dem.crs).geometry.union_all()
            try:
                clipped, transform = mask(dem, [local], crop=True, filled=False)
            except ValueError as exc:
                raise ValueError('流域边界与 DEM 不相交。') from exc
            elevation = clipped[0]
            valid = elevation.compressed()
            if valid.size == 0:
                raise ValueError('流域范围内没有有效的 DEM 高程像元。')
            # The full clipped raster is used for statistics; only the browser view is sampled.
            stride = max(1, int(np.ceil(max(elevation.shape) / 750)))
            view = elevation[::stride, ::stride].astype(float).filled(np.nan)
            rows, cols = view.shape
            source_transform=transform*transform.scale(stride,stride)
            bounds=array_bounds(rows,cols,source_transform)
            geographic_transform,width,height=calculate_default_transform(dem.crs,'EPSG:4326',cols,rows,*bounds)
            reduction=max(1,width/750,height/750)
            geographic_transform=geographic_transform*geographic_transform.scale(reduction,reduction)
            width,height=int(np.ceil(width/reduction)),int(np.ceil(height/reduction))
            geographic=np.full((height,width),np.nan)
            reproject(view,geographic,src_transform=source_transform,src_crs=dem.crs,src_nodata=np.nan,
                      dst_transform=geographic_transform,dst_crs='EPSG:4326',dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            x=geographic_transform.c+(np.arange(width)+.5)*geographic_transform.a
            y=geographic_transform.f+(np.arange(height)+.5)*geographic_transform.e
            fig = go.Figure(go.Heatmap(x=x, y=y, z=geographic, colorscale=TERRAIN_COLORS, name='高程',
                                       colorbar={'title': '高程 (m)','thickness':13,'len':.75},
                                       hovertemplate='经度 %{x:.4f}°E<br>纬度 %{y:.4f}°N<br>高程 %{z:.1f} m<extra></extra>'))
            boundary=basin.to_crs('EPSG:4326').geometry.union_all()
            polygons = list(boundary.geoms) if hasattr(boundary, 'geoms') else [boundary]
            for polygon in polygons:
                px, py = polygon.exterior.xy
                fig.add_trace(go.Scatter(x=list(px), y=list(py), mode='lines',
                                         line={'color': '#111111', 'width': 2}, name='流域边界'))
            map_layout(fig,'流域地形概况',boundary.centroid.y)
            return {'figure': fig.to_plotly_json(), 'area_km2': area_km2,
                    'min_m': float(valid.min()), 'max_m': float(valid.max()),
                    'mean_m': float(valid.mean()), 'crs': str(dem.crs)}
