from __future__ import annotations

import math
import numpy as np
import pandas as pd
import geopandas as gpd

# ===== Константы / имена колонок =====
CAP_COL = "capacity"
CAP_TYPE_COL = "cap_type"          # 'base' | 'real'
GEOM_TYPE_COL = "geom_type"        # 'Point' | 'Polygon'
BLOCK_ID_COL = "block_id"
DEMAND_COL = "demand"              # спрос квартала (в людях)
UNMET_COL = "unmet_block_demand"
SANPIN_COL = "sanpin_cap"
CAP_MAX_COL = "cap_max"
NEW_CAP_COL = "new_capacity"
ADD_CAP_COL = "added_capacity"
SAT_COL = "saturated"
KEEP_COL = "keep"                  # объект сохранён (True) или удалён из-за совпадения (False)
DROP_REASON_COL = "drop_reason"    # причина удаления

# ===== Утилиты =====
def _ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS (e.g. EPSG:3857).")
    if gdf.crs.to_epsg() != epsg:
        return gdf.to_crs(epsg)
    return gdf

def _geom_type_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    return gdf.geometry.geom_type

def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol

# ===== 0) Подготовка входа =====
def prepare_blocks_with_demand(
    blocks_gdf: gpd.GeoDataFrame,
    demand_per_1000: float,
    population_col: str = "population",
    epsg: int = 3857,
) -> gpd.GeoDataFrame:
    """
    Добавляет спрос DEMAND_COL = population * demand_per_1000 / 1000.
    Создаёт block_id, если нет.
    """
    gb = _ensure_crs(blocks_gdf.copy(), epsg)
    if population_col not in gb.columns:
        raise ValueError(f"blocks_gdf must contain '{population_col}'.")
    gb[DEMAND_COL] = (gb[population_col].fillna(0).astype(float) * float(demand_per_1000) / 1000.0)
    if BLOCK_ID_COL not in gb.columns:
        gb = gb.reset_index(drop=True)
        gb[BLOCK_ID_COL] = gb.index
    return gb

def prepare_services_cap_types(
    service_gdf: gpd.GeoDataFrame,
    base_count: float,
    epsg: int = 3857,
) -> gpd.GeoDataFrame:
    """
    Добавляет CAP_TYPE_COL ('base'|'real') на основе BASE_COUNT.
    """
    gf = _ensure_crs(service_gdf.copy(), epsg)
    if CAP_COL not in gf.columns:
        raise ValueError(f"schools_gdf must contain '{CAP_COL}'.")
    if GEOM_TYPE_COL not in gf.columns:
        gf[GEOM_TYPE_COL] = _geom_type_series(gf)

    def _cap_type(v):
        if pd.isna(v):  # пустые трактуем как базовые, при желании можно иначе
            return "base"
        return "base" if _isclose(float(v), float(base_count)) else "real"

    gf[CAP_TYPE_COL] = gf[CAP_COL].apply(_cap_type)
    return gf

# ===== 1) Совпадения Point↔Polygon: оставить полигоны, перенести real с точки на полигон =====
def merge_points_into_polygons_keep_polys(
    fac_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    gdf = fac_gdf.copy()
    if GEOM_TYPE_COL not in gdf.columns:
        gdf[GEOM_TYPE_COL] = _geom_type_series(gdf)
    gdf[KEEP_COL] = True
    gdf[DROP_REASON_COL] = pd.NA

    polys = gdf[gdf[GEOM_TYPE_COL] == "Polygon"].copy()
    points = gdf[gdf[GEOM_TYPE_COL] == "Point"].copy()
    if points.empty or polys.empty:
        return gdf

    j = gpd.sjoin(points, polys[["geometry"]], how="left", predicate="within")
    covered = j[~j.index_right.isna()]
    if covered.empty:
        return gdf

    pts_idx = covered.index.unique()

    # переносим real-точки на полигоны
    real_pts = points.loc[pts_idx][points.loc[pts_idx, CAP_TYPE_COL] == "real"]
    if not real_pts.empty:
        join_real = gpd.sjoin(real_pts, polys[["geometry"]], how="left", predicate="within")
        # берём максимум capacity, если несколько real-точек попали в один полигон
        per_poly = real_pts[[CAP_COL]].groupby(join_real["index_right"]).max()[CAP_COL]
        polys.loc[per_poly.index, CAP_COL] = per_poly.values
        polys.loc[per_poly.index, CAP_TYPE_COL] = "real"

    # помечаем покрытые точки как удалённые
    gdf.loc[pts_idx, KEEP_COL] = False
    gdf.loc[pts_idx, DROP_REASON_COL] = "covered_by_polygon"

    # собрать обратно: полигоны (обновлённые) + оставшиеся точки
    remaining_points = points.drop(index=pts_idx)
    out = pd.concat([polys, remaining_points, gdf[(gdf[GEOM_TYPE_COL] != "Polygon") & (gdf[GEOM_TYPE_COL] != "Point")]], ignore_index=True)
    if GEOM_TYPE_COL not in out.columns:
        out[GEOM_TYPE_COL] = _geom_type_series(out)
    out = out.merge(
        gdf[[KEEP_COL, DROP_REASON_COL]].reset_index().rename(columns={"index": "_orig_idx"}),
        left_index=True, right_index=True, how="left"
    )
    return gpd.GeoDataFrame(out.drop(columns=["_orig_idx"]), geometry="geometry", crs=fac_gdf.crs)

# ===== 2) СанПиН потолок (для полигонов с base) =====
def add_sanpin_ceiling(
    fac_gdf: gpd.GeoDataFrame,
    m2_per_person: float = 5.0
) -> gpd.GeoDataFrame:
    gdf = fac_gdf.copy()
    if GEOM_TYPE_COL not in gdf.columns:
        gdf[GEOM_TYPE_COL] = _geom_type_series(gdf)
    gdf[SANPIN_COL] = np.inf
    areas = gdf.geometry.area
    mask = (gdf[GEOM_TYPE_COL] == "Polygon") & (gdf[CAP_TYPE_COL] == "base")
    gdf.loc[mask, SANPIN_COL] = np.floor(areas.loc[mask] / float(m2_per_person)).astype(float)
    return gdf

# ===== 3) Привязка к кварталам =====
def attach_blocks(
    fac_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    gf = fac_gdf.copy()
    gb = blocks_gdf.copy()
    if GEOM_TYPE_COL not in gf.columns:
        gf[GEOM_TYPE_COL] = _geom_type_series(gf)

    # точки — как есть; полигоны — центроиды
    anchors = gf.geometry.copy()
    poly_mask = (gf[GEOM_TYPE_COL] == "Polygon")
    anchors.loc[poly_mask] = gf.loc[poly_mask].geometry.centroid
    pts = gpd.GeoDataFrame(gf.drop(columns=["geometry"]), geometry=anchors, crs=gf.crs)

    sj = gpd.sjoin(pts, gb[[BLOCK_ID_COL, DEMAND_COL, "geometry"]], how="left", predicate="within")
    gf[BLOCK_ID_COL] = sj[BLOCK_ID_COL].values
    gf[DEMAND_COL] = sj[DEMAND_COL].values
    gf[BLOCK_ID_COL] = gf[BLOCK_ID_COL].fillna(-1).astype(int)
    gf[DEMAND_COL] = gf[DEMAND_COL].fillna(0.0).astype(float)
    return gf

# ===== 4) cap_max для базовых =====
def add_cap_max(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    base = gdf[CAP_COL].astype(float).fillna(0.0)
    triple = 3.0 * base
    sanpin = gdf.get(SANPIN_COL, pd.Series(np.inf, index=gdf.index)).astype(float)
    cap_max = base.copy()
    mask_base = (gdf[CAP_TYPE_COL] == "base")
    cap_max.loc[mask_base] = np.minimum(triple.loc[mask_base], sanpin.loc[mask_base])
    gdf[CAP_MAX_COL] = cap_max
    return gdf

# ===== 5) Нелинейное распределение относительно максимального спроса =====
def allocate_relative_to_max(
    gdf: gpd.GeoDataFrame,
    k: float = 2.0,
) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out[NEW_CAP_COL] = out[CAP_COL].astype(float).values
    out[ADD_CAP_COL] = 0.0
    out[SAT_COL] = False
    out[UNMET_COL] = 0.0

    if CAP_MAX_COL not in out.columns:
        out = add_cap_max(out)

    dmax_global = max(out[DEMAND_COL].max(), 1.0)
    w_max = math.exp(k * 1.0)

    for bid, bdf in out.groupby(BLOCK_ID_COL, group_keys=False):
        demand_b = float(bdf[DEMAND_COL].iloc[0])
        if demand_b <= 0:
            continue

        idx_base = bdf.index[bdf[CAP_TYPE_COL] == "base"]
        if len(idx_base) == 0:
            continue

        head = (out.loc[idx_base, CAP_MAX_COL] - out.loc[idx_base, CAP_COL]).clip(lower=0).astype(float)
        total_head = head.sum()
        if total_head <= 0:
            continue

        # вес блока по спросу
        w_block = math.exp(k * (demand_b / dmax_global))
        head_max_local = head.max()

        # бюджет блока: хотим, чтобы блок с максимальным спросом получал head_max_local
        # множитель len(idx_base) слегка выравнивает блоки с множеством объектов
        T = head_max_local * (w_block / w_max) * len(idx_base)
        add_total = min(T, total_head)
        if add_total <= 0:
            continue

        # старт: пропорционально headroom (мягко)
        add = (head / head.sum()) * add_total

        # водораздел: клип по headroom и перераспределение излишка
        for _ in range(12):
            over = add - head
            if (over <= 1e-9).all():
                break
            surplus = over.clip(lower=0).sum()
            add = np.minimum(add, head)
            free = (add < head - 1e-12)
            if not free.any():
                break
            add[free] += surplus * (head[free] / head[free].sum())

        # применяем: меняем только base
        out.loc[idx_base, ADD_CAP_COL] = add
        out.loc[idx_base, NEW_CAP_COL] = round(out.loc[idx_base, CAP_COL] + add, 0)
        out.loc[idx_base, SAT_COL] = np.isclose(out.loc[idx_base, NEW_CAP_COL], out.loc[idx_base, CAP_MAX_COL])

    return out

# ===== 6) Главный пайплайн =====
def recompute(
    blocks_gdf: gpd.GeoDataFrame,
    service_gdf: gpd.GeoDataFrame,
    *,
    demand_per_1000: float,
    base_count: float,
    m2_per_person: float,
    epsg: int = 3857,
    k: float = 2.0,
) -> gpd.GeoDataFrame:
    """
    Возвращает КОПИЮ service_gdf (с учётом правил совпадений: точки, покрытые полигонами, удаляются),
    где у базовых геометрий рассчитаны новые capacity.
    Поля на выходе: capacity, cap_type, sanpin_cap, cap_max, added_capacity, new_capacity, saturated, keep, drop_reason, block_id, demand
    """
    # 1) подготовить блоки с реальным спросом
    gb = prepare_blocks_with_demand(blocks_gdf, demand_per_1000=demand_per_1000, epsg=epsg)

    # 2) подготовить сервис: cap_type
    gf0 = prepare_services_cap_types(service_gdf, base_count=base_count, epsg=epsg)

    # 3) совпадения point↔polygon (оставляем полигоны)
    gf1 = merge_points_into_polygons_keep_polys(gf0)

    # 4) санпин потолок
    gf2 = add_sanpin_ceiling(gf1, m2_per_person=m2_per_person)

    # 5) привязка к кварталам
    gf3 = attach_blocks(gf2, gb)

    # 6) cap_max
    gf4 = add_cap_max(gf3)

    # 7) нелинейная аллокация относительно максимального спроса
    out = allocate_relative_to_max(gf4, k=k)
    return out
