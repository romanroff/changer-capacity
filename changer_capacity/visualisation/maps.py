import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonPopup
from branca.colormap import LinearColormap

def make_blocks_capacity(
    buildings_blocks: gpd.GeoDataFrame,
    prov_df_old: pd.DataFrame,
    prov_df_new: pd.DataFrame,
    *,
    id_col: str | None = None,           # id в buildings_blocks (если None — берем индекс)
    prov_id_col_old: str | None = None,  # id в prov_df_old (если None — индекс)
    prov_id_col_new: str | None = None,  # id в prov_df_new (если None — индекс)
    capacity_col: str = "capacity",
    drop_invalid: bool = True
) -> folium.Map:
    # --- геометрия -> WGS84
    gdf = buildings_blocks.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(3857, allow_override=True)
    gdf = gdf.to_crs(4326)

    if drop_invalid:
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        try:
            gdf = gdf[gdf.geometry.is_valid]
        except Exception:
            pass

    # --- id в гео
    if id_col is None:
        id_col = "_poly_id_"
        gdf[id_col] = gdf.index

    # --- нормализуем prov_old/new
    def _norm(df, prov_id_col):
        if prov_id_col is None:
            df = df.rename_axis(id_col).reset_index()
        else:
            df = df.rename(columns={prov_id_col: id_col})
        return df

    old = _norm(prov_df_old, prov_id_col_old)[[id_col, capacity_col]].rename(columns={capacity_col: "capacity_old"})
    new = _norm(prov_df_new, prov_id_col_new)[[id_col, capacity_col]].rename(columns={capacity_col: "capacity_new"})

    # --- merge и метрики
    gdf = gdf.merge(old, on=id_col, how="left").merge(new, on=id_col, how="left")
    gdf["capacity_old"] = pd.to_numeric(gdf["capacity_old"], errors="coerce")
    gdf["capacity_new"] = pd.to_numeric(gdf["capacity_new"], errors="coerce")
    gdf["capacity_delta"] = gdf["capacity_new"] - gdf["capacity_old"]

    # --- карта
    m = folium.Map(tiles="cartodbpositron", zoom_start=12)
    minx, miny, maxx, maxy = gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # === 1) Общая шкала для old & new ===
    both = pd.concat([gdf["capacity_old"], gdf["capacity_new"]], axis=0)
    both = pd.to_numeric(both, errors="coerce")
    if both.notna().any():
        vmin_common = float(both.min(skipna=True))
        vmax_common = float(both.max(skipna=True))
        if vmin_common == vmax_common:
            vmax_common = vmin_common + 1.0
    else:
        vmin_common, vmax_common = 0.0, 1.0

    cmap_common = LinearColormap(colors=["#f7fbff", "#10ae01"], vmin=vmin_common, vmax=vmax_common)
    cmap_common.caption = "Capacity (old/new): low → high"
    cmap_common.add_to(m)

    # === 2) Дивергентная шкала для Δ ===
    delta_vals = pd.to_numeric(gdf["capacity_delta"], errors="coerce")
    if delta_vals.notna().any():
        vmin_d = float(delta_vals.min(skipna=True))
        vmax_d = float(delta_vals.max(skipna=True))
        if vmin_d == vmax_d:
            # немного расширим диапазон, чтобы была видна легенда
            vmin_d, vmax_d = vmin_d - 1.0, vmax_d + 1.0
    else:
        vmin_d, vmax_d = -1.0, 1.0

    cmap_delta = LinearColormap(colors=["#ffffff", "#fff200", "#07c300"], vmin=vmin_d, vmax=vmax_d)
    cmap_delta.caption = "Δ Capacity: negative → positive"
    cmap_delta.add_to(m)

    # --- утилита добавления слоя с POPUP при клике
    def add_layer(df, color_col, name, cmap, show=True, outline="#222", weight=0.7, fill_opacity=0.9):
        vals = pd.to_numeric(df[color_col], errors="coerce")
        if not vals.notna().any():
            return

        def style(feat):
            v = feat["properties"].get(color_col, None)
            try:
                vv = float(v) if v is not None else None
            except Exception:
                vv = None
            return {
                "color": outline,
                "weight": weight,
                "fillOpacity": fill_opacity,
                "fillColor": cmap(vv) if vv is not None else "#ffffff",
            }

        # в payload включаем сразу все три значения, чтобы popup всегда их показывал
        payload = df[[id_col, "capacity_old", "capacity_new", "capacity_delta", "geometry"]]

        popup = GeoJsonPopup(
            fields=[id_col, "capacity_old", "capacity_new", "capacity_delta"],
            aliases=["ID", "Capacity (old)", "Capacity (new)", "Δ Capacity"],
            localize=True,
            labels=True
        )

        layer = folium.FeatureGroup(name=name, show=show, overlay=True)
        folium.GeoJson(
            payload,
            name=name,
            style_function=style,
            popup=popup
        ).add_to(layer)
        layer.add_to(m)

    # --- три слоя: оба по общей шкале + дельта по дивергентной
    add_layer(gdf, "capacity_old", "Capacity (old)", cmap_common, show=False)
    add_layer(gdf, "capacity_new", "Capacity (new)", cmap_common, show=True)
    add_layer(gdf, "capacity_delta", "Δ Capacity (abs)", cmap_delta, show=True)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
