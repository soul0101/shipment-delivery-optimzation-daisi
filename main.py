import time
import math 
import random
import helper
import numpy as np
import pandas as pd 
import pydaisi as pyd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

multi_depot_package_allocation = pyd.Daisi("soul0101/Multi-Depot Package Allocation")
vehicle_routing_problem = pyd.Daisi("soul0101/Vehicle Routing Problem ")

################################## Solvers ##############################################

@st.cache
def get_allocations(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities):
    """
    Calls 'soul0101/Multi-Depot Package Allocation' daisi to get the drops allocated to each depot

    Parameters
    ----------
    depot_locations: 
        A list of tuples containing the (latitude, longitude) of each depot.
    drop_locations: 
        A list of tuples containing the (latitude, longitude) of each drop location.
    depot_ids: 
        A list of integers containing the id of each depot.
    drop_ids: 
        A list of integers containing the id of each drop location.
    depot_capacities: 
        A list of integers representing the maximum number of packages that can be allocated to each depot.

    Returns
    -------
    Dict containing the allocation information for each depot:
        {
            <depot_id> : {
                "depot_location": <array(latitude, longitude)>,
                "drops" : {
                            <drop_id1> : "drop_location": <array(latitude, longitude)>,
                            <drop_id2> : "drop_location": <array(latitude, longitude)>,
                        }
                "depot_capacity": int>
            }, ...
        }
    """
    return multi_depot_package_allocation.allocate_packages(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities).value
    
@st.cache
def data_modeller(allocation_result):
    """
    Models the result from the 'Allocation Problem' for the 'Routing Problem'
    Parameters
    ----------
    allocation result: 
        Dict containing the allocation information for each depot:
            {
                <depot_id> : {
                    "depot_location": <array(latitude, longitude)>,
                    "drops" : {
                                <drop_id1> : "drop_location": <array(latitude, longitude)>,
                                <drop_id2> : "drop_location": <array(latitude, longitude)>,
                            }
                    "depot_capacity": int>
                }, ...
            }

    Returns
    -------
    routing_tasks: 
        List of (lists with its first element as the location array of source and rest elements being location arrays of drops)
        Eg: [ [(lat_source, long_source), (lat_drop1, long_drop1)], ...]
    """
    routing_tasks = []

    for depot_id, depot_info in allocation_result.items():
        input_location_arr = []
        input_location_arr.append(depot_info["depot_location"])
        drops = depot_info["drops"]

        for drop_id, drop_info in drops.items():
            input_location_arr.append(drop_info["drop_location"])
        
        routing_tasks.append(input_location_arr)
    
    return routing_tasks

def run_serial_route_solver(allocation_results, vehicle_capacities_list, st_progress_bar=None, **kwargs):
    """
    Runs the Vehicle Routing Problem solver for each routing task 'serially'
    Parameters
    ----------
    allocation result: 
        Dict containing the allocation information for each depot:
            {
                <depot_id> : {
                    "depot_location": <array(latitude, longitude)>,
                    "drops" : {
                                <drop_id1> : "drop_location": <array(latitude, longitude)>,
                                <drop_id2> : "drop_location": <array(latitude, longitude)>,
                            }
                    "depot_capacity": int>
                }, ...
            }
    vehicle_capacities_list:
        List of vehicle_capacities (List of capacity of each vehicle at a particular depot)
    
    st_progress_bar(optional): 
        Streamlit progress bar object to show progress

    Returns
    -------

    Dict containing routing results for each depot: 
        {
            <depot_id> : {
                "route_locs_x": List of latitudes of points in route (in order),
                "route_locs_y": List of longitudes of points in route (in order),
                "route_node_index": List of nodes in route (in order)
            }, 
            ...
        }
    """
    routing_tasks = data_modeller(allocation_results)
    routing_results = []
    for index, task in enumerate(routing_tasks):  
        if st_progress_bar is not None:
            st_progress_bar.progress((index + 1) / len(routing_tasks))

        routing_results.append(get_route(task, vehicle_capacities_list[index], **kwargs))

    return routing_results

@st.cache
def get_route(task, vehicle_capacities, **kwargs):
    """
    Calls 'soul0101/Vehicle Routing Problem' daisi to run the Optimal Route Calculator

    Parameters
    ----------
    input_locations: 
        A list with its first element as the location array of source and rest elements being location arrays of drops
        Eg: [(lat_source, long_source), (lat_drop1, long_drop1), (lat_drop2, long_drop2), ...]
    vehicle_capacities: 
        A list containing the number of drops each vehicle can visit. (Should have atleast one vehicle)
    search_timeout: 
        Maximum time to find a solution
    first_sol_strategy: string
        A first solution strategy. Reference: https://developers.google.com/optimization/routing/routing_options#first_sol_options
    ls_metaheuristic: string
        Local Search Option Metaheuristic. Reference: https://developers.google.com/optimization/routing/routing_options#local_search_options

    Returns
    -------
    Dict containing route information for each vehicle:
        {\n
            <vehicle_id>: {\n
                "route_locs_x": List containing latitudes of drops in route (in order)\n
                "route_locs_y" : List containing longitudes of drops in route (in order)\n
                "route_node_index" : List containing drop indexes in route (in order)\n
            }\n
        }\n
    """
    return vehicle_routing_problem.vrp_calculator(task, vehicle_capacities, **kwargs).value

def run_parallel_route_solver(allocation_results, vehicle_capacities_list, st_progress_bar=None, **kwargs):
    """
    Runs the Vehicle Routing Problem solver for each routing task 'parallely'
    Parameters
    ----------
    allocation result: 
        Dict containing the allocation information for each depot:
            {
                <depot_id> : {
                    "depot_location": <array(latitude, longitude)>,
                    "drops" : {
                                <drop_id1> : "drop_location": <array(latitude, longitude)>,
                                <drop_id2> : "drop_location": <array(latitude, longitude)>,
                            }
                    "depot_capacity": int>
                }, ...
            }
    vehicle_capacities_list:
        List of vehicle_capacities (List of capacity of each vehicle at a particular depot)
    
    st_progress_bar(optional): 
        Streamlit progress bar object to show progress

    Returns
    -------

    Dict containing routing results for each depot: 
        {
            <depot_id> : {
                "route_locs_x": List of latitudes of points in route (in order),
                "route_locs_y": List of longitudes of points in route (in order),
                "route_node_index": List of nodes in route (in order)
            }, 
            ...
        }
    """
    routing_tasks = data_modeller(allocation_results)
    vehicle_routing_problem.workers.set(100)
    keyword_args = {}
    for arg, value in kwargs.items():
        keyword_args[arg] = value
    
    args_list = [{**keyword_args, **{"input_locations": task, "vehicle_capacities": vehicle_capacities}} for task, vehicle_capacities in zip(routing_tasks, vehicle_capacities_list)]
    vrp_solver_instance = vehicle_routing_problem.map(func="vrp_calculator", args_list=args_list)
    vrp_solver_instance.start()
    
    check_parallel_solver_status(vrp_solver_instance, st_progress_bar=st_progress_bar, timeout = 600)
    routing_results = [result for id, result in vrp_solver_instance.value.items()]

    return routing_results

def check_parallel_solver_status(vrp_solver_instance, st_progress_bar=None, timeout = 600):   
    """
    Checks the status of a running parallel execution

    Parameters
    ----------
    vrp_solver_instance: 
        The parallel solver instance
    st_progress_bar (optional):
        The streamlit progress bar object to track progress
    timeout (optional):
        Timeout before killing the solver
    """
    st = time.time()

    if st_progress_bar is not None:
        total_computations = len(vrp_solver_instance.value)
        while True: 
            completed_computations = 0
            et = time.time()
            if (et - st > timeout):
                return
            is_running = False
            for id, result in vrp_solver_instance.value.items():
                if result is None:
                    is_running = True
                else:
                    completed_computations += 1
            st_progress_bar.progress(completed_computations / total_computations)
            if not is_running:
                return
            time.sleep(5)
    else:
        while True: 
            is_running = False
            et = time.time()
            if (et - st > timeout):
                return
            for id, result in vrp_solver_instance.value.items():
                if result is None:
                    is_running = True
                    break
            if not is_running:
                return
            time.sleep(5)

def schedule_deliveries(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities, vehicle_capacities_list, solver="serial", **kwargs):
    """
    All in one function to generate allocations and routes for each depot

    Parameters
    ----------
    depot_locations: 
        A list of tuples containing the (latitude, longitude) of each depot.
    drop_locations: 
        A list of tuples containing the (latitude, longitude) of each drop location.
    depot_ids: 
        A list of integers containing the id of each depot.
    drop_ids: 
        A list of integers containing the id of each drop location.
    depot_capacities: 
        A list of integers representing the maximum number of packages that can be allocated to each depot.
    vehicle_capacities_list:
        List of vehicle_capacities (List of capacity of each vehicle at a particular depot)
    solver: 
        Select the solver to be used. "serial" or "parallel" (alpha stage)
    Returns
    -------

    (allocation_results, routing_results):
        (
            Dict containing the allocation information for each depot:
                {
                    <depot_id> : {
                        "depot_location": <array(latitude, longitude)>,
                        "drops" : {
                                    <drop_id1> : "drop_location": <array(latitude, longitude)>,
                                    <drop_id2> : "drop_location": <array(latitude, longitude)>,
                                }
                        "depot_capacity": int>
                    }, ...
                }, 
            Dict containing routing results for each depot: 
                {
                    <depot_id> : {
                        "route_locs_x": List of latitudes of points in route (in order),
                        "route_locs_y": List of longitudes of points in route (in order),
                        "route_node_index": List of nodes in route (in order)
                    }, 
                    ...
                }
        )
    """

    allocation_results =  get_allocations(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities)
    if solver=="parallel":
        routing_results = run_parallel_route_solver(allocation_results, vehicle_capacities_list, **kwargs)
    else:
        routing_results = run_serial_route_solver(allocation_results, vehicle_capacities_list, **kwargs)

    return (allocation_results, routing_results)

################################## Plots ##############################################

def get_allocations_plot_plotly(allocation_results):
    """
    Plot the allocations for each depot.

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    return multi_depot_package_allocation.get_allocations_plot_plotly(allocation_results).value

def get_route_plot_plotly(depot_locations, drop_locations, final_route, fig=None):
    """
    Plot the planned route for each depot.

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    if fig is None:
        fig = get_locations_plot_plotly(depot_locations, drop_locations)

    color = px.colors.sequential.Inferno
    for vehicle_id, route in final_route.items():
        connector_color = random.choice(color)
        fig.add_trace(go.Scatter(x=route["route_locs_x"], y=route["route_locs_y"],
                        mode='lines+markers', 
                        name="Vehicle #%s"%(vehicle_id),
                        showlegend=False,
                        line_color=connector_color, 
                        marker=dict(opacity=0)))


    fig.update_layout(
        width=700,
        height=500,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),
        title={
        'text': "Delivery Plan",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        paper_bgcolor="#D3D3D3",
        plot_bgcolor="#C0C0C0",
        font=dict(
            family="monospace",
            size=18,
            color="black"
        )
    )
    return fig

def get_locations_plot_plotly(depot_locations, drop_locations, depot_ids = None, drop_ids=None, depot_capacities=None, fig=None):
    """
    Returns a Plotly Figure for Source and Drop locations
    
    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    if fig is None:
        fig = go.Figure()
    
    depot_ids, drop_ids, depot_capacities = helper.sanitize(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities)
    depot_info = ["Depot ID: %s \n Depot Capacity: %s" % (d_id, d_cap) for (d_id, d_cap) in zip(depot_ids, depot_capacities)]
    
    fig.add_trace(go.Scatter(x = drop_locations[:, 0], y = drop_locations[:, 1],
                    mode='markers',
                    name='Drops', 
                    hovertext=drop_ids,
                    marker=dict(color='#848ff0', size=6, 
                    line=dict(width=1,color='DarkSlateGrey'))))

    fig.add_trace(go.Scatter(x = depot_locations[:, 0], y = depot_locations[:, 1],
                    mode='markers',
                    name='Depots',
                    hovertext=depot_info,
                    marker=dict(color='red', size=12, 
                    line=dict(width=1,color='DarkSlateGrey'))
                    ))

    fig.update_layout(
        width=700,
        height=500,
        margin=dict(l=50,r=50,b=100,t=100,pad=4),
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        paper_bgcolor="#D3D3D3",
        plot_bgcolor="#C0C0C0",
        font=dict(
            family="monospace",
            size=18,
            color="black"
        )
    )
    return fig    

################################## UI ##############################################

def get_dummy_data():
    df_depots = pd.read_csv('./data/city_depots1.csv')
    df_drops = pd.read_csv('./data/city_drops1.csv')

    depots = np.column_stack((df_depots['Latitude'], df_depots['Longitude']))
    drops = np.column_stack((df_drops['Latitude'], df_drops['Longitude']))
    depot_capacities = df_depots['Depot Capacity']
    depot_ids = df_depots['Depot ID']
    drop_ids = df_drops['Drop ID']
    return [depots, drops, depot_ids, drop_ids, depot_capacities]

def st_sidebar():
    st.sidebar.markdown("""
                    # Computation Settings
                    ------------------------
                    """)
    run_parallel = st.sidebar.checkbox("Enable Parallel Computations (Alpha Stage)", help="Run the expensive route calculations parallely with multiple workers")

    st.sidebar.markdown("""
                        # Vehicle Routing Settings
                        ------------------------
                        """)
    st.sidebar.header("Local search options")
    #Local Search Option
    sb_local_mh = st.sidebar.selectbox("Select a local search option", 
                        options=["AUTOMATIC", "GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH", 
                        "SIMULATED_ANNEALING", "TABU_SEARCH"], 
                        help="""
                            AUTOMATIC             - Lets the solver select the metaheuristic.\n
                            GREEDY_DESCENT        - Accepts improving (cost-reducing) local search neighbors until a local minimum is reached.\n
                            GUIDED_LOCAL_SEARCH	  - Uses guided local search to escape local minima (cf. http://en.wikipedia.org/wiki/Guided_Local_Search); this is generally the most efficient metaheuristic for vehicle routing.\n
                            SIMULATED_ANNEALING	  - Uses simulated annealing to escape local minima (cf. http://en.wikipedia.org/wiki/Simulated_annealing).\n
                            TABU_SEARCH	          - Uses tabu search to escape local minima (cf. http://en.wikipedia.org/wiki/Tabu_search).\n
                        """)

    st.sidebar.header("First Solution Strategy")
    #First Solution Strategy
    sb_first_sol= st.sidebar.selectbox("Select a first solution strategy", 
                        options=["AUTOMATIC", "PATH_CHEAPEST_ARC", "PATH_MOST_CONSTRAINED_ARC", 
                        "EVALUATOR_STRATEGY", "SAVINGS", "SWEEP", "CHRISTOFIDES", "ALL_UNPERFORMED",
                        "BEST_INSERTION", "PARALLEL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION",
                        "GLOBAL_CHEAPEST_ARC", "LOCAL_CHEAPEST_ARC", "FIRST_UNBOUND_MIN_VALUE"], 
                        
                        help="""
                            AUTOMATIC - Lets the solver detect which strategy to use according to the model being solved. \n
                            PATH_CHEAPEST_ARC - Starting from a route "start" node, connect it to the node which produces the cheapest route segment, then extend the route by iterating on the last node added to the route.\n
                            PATH_MOST_CONSTRAINED_ARC - Similar to PATH_CHEAPEST_ARC, but arcs are evaluated with a comparison-based selector which will favor the most constrained arc first. To assign a selector to the routing model, use the method ArcIsMoreConstrainedThanArc(). \n
                            EVALUATOR_STRATEGY - Similar to PATH_CHEAPEST_ARC, except that arc costs are evaluated using the function passed to SetFirstSolutionEvaluator(). \n
                            SAVINGS - Savings algorithm (Clarke & Wright).\n
                            SWEEP - Sweep algorithm (Wren & Holliday). \n
                            ALL_UNPERFORMED - Make all nodes inactive. Only finds a solution if nodes are optional (are element of a disjunction constraint with a finite penalty cost).\n
                            BEST_INSERTION - Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the global cost function of the routing model. As of 2/2012, only works on models with optional nodes (with finite penalty costs).\n
                            PARALLEL_CHEAPEST_INSERTION - Iteratively build a solution by inserting the cheapest node at its cheapest position; the cost of insertion is based on the arc cost function. Is faster than BEST_INSERTION.\n
                            LOCAL_CHEAPEST_INSERTION - Iteratively build a solution by inserting each node at its cheapest position; the cost of insertion is based on the arc cost function. Differs from PARALLEL_CHEAPEST_INSERTION by the node selected for insertion; here nodes are considered in their order of creation. Is faster than PARALLEL_CHEAPEST_INSERTION.\n
                            GLOBAL_CHEAPEST_ARC - Iteratively connect two nodes which produce the cheapest route segment.\n
                            LOCAL_CHEAPEST_ARC - Select the first node with an unbound successor and connect it to the node which produces the cheapest route segment.\n
                            FIRST_UNBOUND_MIN_VALUE - Select the first node with an unbound successor and connect it to the first available node. This is equivalent to the CHOOSE_FIRST_UNBOUND strategy combined with ASSIGN_MIN_VALUE (cf. constraint_solver.h).\n
                        """)

    st.sidebar.header("Search Timeout")
    search_timeout = st.sidebar.slider(
            'Select a timeout for searching an optimal solution', 10, 600, 10, step=5, help='Increase the time in-case the solution is not satisfactory')
    
    return (sb_local_mh, sb_first_sol, search_timeout, run_parallel)

def st_ui():
    sb_local_mh, sb_first_sol, search_timeout, run_parallel = st_sidebar()
    st.write("# Welcome to the Delivery Planner Daisi! 👋")
    st.markdown(
        """
        Consider a scenario where a delivery company wants to deliver packages (__drops__) all across the city with the help of 
        fulfillment centres (__depots__) at strategic locations. \n
        Each depot has a pre-defined __number of vehicles__ with certain package carrying __capacity__. \n
        Our job is to __allocate__ the drops to a service centre and generate __planned routes__ for each vehicle with the objective of __minimizing cost__ (distance travelled).
        """
    )
    
    [depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities] = get_dummy_data()

    st.header("Locations")
    before_fig = get_locations_plot_plotly(depot_locations, drop_locations, depot_ids=depot_ids, drop_ids=drop_ids, depot_capacities=depot_capacities)
    st.plotly_chart(before_fig)

    generate_btn = st.button("Generate Plan")
    my_bar = st.empty()

    if generate_btn: 
        print("number of workers: ", vehicle_routing_problem.workers.number)
        with st.spinner("Generating allocations..."):
            allocation_results = get_allocations(depot_locations, drop_locations, depot_ids, drop_ids, depot_capacities)
            allocation_fig = get_allocations_plot_plotly(allocation_results)

            # Generating mock vehicle_capacities_list
            num_vehicles = 5
            vehicle_capacities_list = []
            for depot_id, depot_info in allocation_results.items():
                num_drops = len(depot_info["drops"])
                num_vehicles = math.ceil(num_drops / 4)
                vehicle_capacities = num_vehicles * [5]
                vehicle_capacities_list.append(vehicle_capacities)

        with st.spinner("Generating routes..."):
            my_bar.progress(0)
            if run_parallel:
                routing_results = run_parallel_route_solver(allocation_results, vehicle_capacities_list, st_progress_bar=my_bar, search_timeout=search_timeout, first_sol_strategy=sb_first_sol, ls_metaheuristic=sb_local_mh)                    
            else:
                routing_results = run_serial_route_solver(allocation_results, vehicle_capacities_list, st_progress_bar=my_bar, search_timeout=search_timeout, first_sol_strategy=sb_first_sol, ls_metaheuristic=sb_local_mh)
                         
            route_fig = None
            for route in routing_results:
                route_fig = get_route_plot_plotly(depot_locations, drop_locations, route, fig=route_fig)
            
            st.plotly_chart(allocation_fig)
            st.plotly_chart(route_fig)

if __name__ == '__main__':
    st_ui()
