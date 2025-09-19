import pandas as pd
import geopandas as gpd

def drop_points_inside_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    points = gdf[gdf.geometry.type == "Point"]

    if polys.empty or points.empty:
        return gdf.reset_index(drop=True).copy()

    union_poly = polys.union_all()
    points_outside = points[~points.geometry.within(union_poly)]
    cleaned = pd.concat([polys, points_outside], ignore_index=True)
    return cleaned.reset_index(drop=True).copy()
