# Shipment Delivery Optimzation Daisi - [Live App](https://app.daisi.io/daisies/soul0101/Shipment%20Delivery%20Optimization/app)

<p align="center">
    <img src="https://user-images.githubusercontent.com/53980340/192162625-dc3a7ac9-1df6-4e45-b4ab-ba165da3f4bf.png" alt="Logo" width="500">        
</p>


**_NOTE:_** This daisi aims to demonstrate daisi-chaining and also the parallel executions of expensive daisi calls. 

Consider a scenario where a delivery company wants to deliver packages (__drops__) all across the city with the help of 
fulfillment centres (__depots__) at strategic locations. <br>
Each depot has a pre-defined __number of vehicles__ with certain package carrying __capacity__. <br>
Our job is to __allocate__ the drops to a service centre and generate __planned routes__ for each vehicle with the objective of __minimizing cost__ (distance travelled).

## Test API Call

```python
import pydaisi as pyd
import math 

shipment_delivery_optimization = pyd.Daisi("soul0101/Shipment Delivery Optimization")

# Load dummy data
depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities = shipment_delivery_optimization.get_dummy_data().value

# Plot locations of Drops and Depots
before_fig = shipment_delivery_optimization.get_locations_plot_plotly(depot_locations, drop_locations, 
                                                        depot_ids=depot_ids, drop_ids=drop_ids, 
                                                        depot_capacities=depot_capacities).value
before_fig.show()

# Get Drop allocations for each Depot
allocation_results = shipment_delivery_optimization.get_allocations(depot_locations, drop_locations, 
                                                        depot_ids, drop_ids, depot_capacities).value

allocation_fig = shipment_delivery_optimization.get_allocations_plot_plotly(allocation_results).value

# Generating mock vehicle_capacities_list
num_vehicles = 5
vehicle_capacities_list = []
for depot_id, depot_info in allocation_results.items():
    num_drops = len(depot_info["drops"])
    num_vehicles = math.ceil(num_drops / 4)
    vehicle_capacities = num_vehicles * [5]
    vehicle_capacities_list.append(vehicle_capacities)

# Calculation Parameters
search_timeout = 10
sb_first_sol = "AUTOMATIC"
sb_local_mh = "AUTOMATIC"

# Parallel Computation
routing_results = shipment_delivery_optimization.run_parallel_route_solver(allocation_results, 
                                                                vehicle_capacities_list, 
                                                                search_timeout=search_timeout, 
                                                                first_sol_strategy=sb_first_sol, 
                                                                ls_metaheuristic=sb_local_mh).value
                   
# Serial Computation
# routing_results = shipment_delivery_optimization.run_serial_route_solver(allocation_results, vehicle_capacities_list, search_timeout=search_timeout, first_sol_strategy=sb_first_sol, ls_metaheuristic=sb_local_mh).value

# Generate result plots    
route_fig = None
for route in routing_results:
    route_fig = shipment_delivery_optimization.get_route_plot_plotly(depot_locations, 
                                                                drop_locations, route, 
                                                                fig=route_fig).value

allocation_fig.show()
route_fig.show()
```
