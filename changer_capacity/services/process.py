from ..utils.geometry import drop_points_inside_polygons
from ..utils.facilities_capacity import recompute
from blocksnet.preprocessing.imputing import impute_services
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.analysis.provision import competitive_provision
from blocksnet.config import service_types_config

SERVICES = ['school', 'kindergarten', 'hospital', 'polyclinic', 'pitch', 'swimming_pool', 'stadium', 'theatre', 'museum', 'cinema', 'mall', 'convenience', 'supermarket', 'cemetery', 'religion', 'market', 'university', 'playground', 'pharmacy', 'fuel', 'beach', 'train_building', 'bank', 'lawyer', 'cafe',
 'subway_entrance', 'multifunctional_center', 'hairdresser', 'restaurant', 'bar', 'park', 'government', 'recruitment', 'hotel', 'zoo', 'circus', 'post', 'police', 'dog_park', 'hostel', 'bakery', 'parking', 'guest_house', 'reserve', 'sanatorium', 'embankment', 'machine_building_plant', 'brewery', 'woodworking_plant', 'oil_refinery', 'plant_of_building_materials', 'wastewater_plant', 'water_works', 'substation', 'train_station', 'bus_station',
 'bus_stop', 'pier', 'animal_shelter', 'prison', 'landfill', 'plant_nursery', 'greenhouse_complex', 'warehouse', 'farmland', 'livestock', 'nursing_home', 'library', 'gallery', 'monastery', 'diplomatic', 'court_house', 'veterinary', 'notary', 'houseware', 'car_wash', 'golf_course', 'plant_gas_oil', 'railway_roundhouse', 'aeroway_terminal', 'crematorium']

def process(service_type: str, service, buildings_blocks, acc_mx, local_crs, demand_per_1000, base_count, m2_per_person, k):
    """
    Process service data and compute accessibility metrics.

    This function processes service data by imputing missing values, dropping points inside polygons,
    recomputing service capacity, and calculating competitive provision and accessibility metrics.

    Args:
        service_type (str): Type of service being processed. Must be one of the predefined SERVICES.
        service: Input service data to be processed.
        buildings_blocks: GeoDataFrame containing building blocks with population data.
        acc_mx: Accessibility matrix used for competitive provision calculations.
        local_crs: Coordinate reference system (EPSG code) for local operations.
        demand_per_1000: Demand per 1000 people for the service.
        base_count: Base count for service capacity calculation.
        m2_per_person: Square meters per person for service capacity calculation.
        k: Parameter for spatial analysis operations.

    Returns:
        tuple: A tuple containing:
            - result: DataFrame with recomputed service capacity and other metrics
            - prov_df: DataFrame with competitive provision metrics for original service data
            - prov_df_new: DataFrame with competitive provision metrics for updated service data

    Note:
        If service_type is not found in predefined SERVICES, prints an error message and returns None.
    """
    if service_type not in SERVICES:
        print(f"{service_type} was not found. Possible services: {SERVICES}")
        return None
    
    service_gdf = impute_services(service, service_type)

    
    service_gdf = drop_points_inside_polygons(service_gdf)
    blocks_gdf = buildings_blocks[["geometry", "population"]]

    result = recompute(
        blocks_gdf=blocks_gdf,
        service_gdf=service_gdf,
        demand_per_1000=demand_per_1000,
        base_count=base_count,
        m2_per_person=m2_per_person,
        epsg=local_crs,
        k=k
    )

    service_new = result[["geometry", "new_capacity"]].rename(
        {"new_capacity": "capacity"}, axis=1
    )


    service_blocks = aggregate_objects(buildings_blocks, service_gdf)[0]
    service_blocks_new = aggregate_objects(buildings_blocks, service_new)[0]

    df = buildings_blocks[["population"]].join(service_blocks[["capacity"]])
    df_new = buildings_blocks[["population"]].join(service_blocks_new[["capacity"]])

    _, demand, accessibility = service_types_config[service_type].values()

    prov_df, _ = competitive_provision(df, acc_mx, accessibility, demand)
    prov_df_new, _ = competitive_provision(df_new, acc_mx, accessibility, demand)

    return result, prov_df, prov_df_new
