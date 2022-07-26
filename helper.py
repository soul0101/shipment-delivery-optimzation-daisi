def sanitize(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities):
    if depot_ids is None:
        depot_ids = list(range(len(depot_locations)))
    if drop_ids is None:
        drop_ids = list(range(len(drop_locations)))
    if depot_capacities is None:
        depot_capacities = len(depot_locations) * ["Unknown"]

    return depot_ids, drop_ids, depot_capacities