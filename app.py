import streamlit as st
import pydeck as pdk
import json
import pandas as pd
import requests

# ------------------------------------------
# SETUP VARS AND DATA
st.set_page_config(layout="wide")  # Use wide layout

with open('sd_roads.json') as f:
    json_data = json.load(f)

df = pd.DataFrame([
    {k: elem[k] for k in ["id", "type", "nodes", "tags"] if k in elem}
    for elem in json_data['elements']
])


with open('osm_sd_roads.geojson') as f:
    geojson_data = json.load(f)

# Build a lookup from way ID to nodes using the DataFrame (nodes are not in the GeoJSON)
nodes_lookup = {}
for _, row in df.iterrows():
    way_id = row.get('id')
    if way_id is not None:
        nodes_lookup[str(way_id)] = row.get('nodes', [])
        nodes_lookup[f"way/{way_id}"] = row.get('nodes', [])

for feat in geojson_data.get('features', []):
    props = feat.get('properties', {})
    name = props.get('name', 'Road')
    way_id = props.get('@id', '')
    highway = props.get('highway', '')
    nodes = nodes_lookup.get(way_id, [])
    props['tooltip_html'] = f"<b>{name}</b><br/>Way ID: {way_id}<br/>Highway: {highway}<br/>Node IDs: {nodes}"

# DEFINE FUNCTIONS
def extract_node_ids(response):
    """Extract all unique node IDs from API response"""
    nodes = set()
    for matching in response.get('matchings', []):
        for leg in matching.get('legs', []):
            nodes.update(leg.get('annotation', {}).get('nodes', []))
    return nodes

def prepare_dataframe_nodes(df):
    """One-time conversion of nodes lists to sets - stores in session state"""
    if 'df_with_sets' not in st.session_state or not st.session_state.nodes_sets_computed:
        df_copy = df.copy()
        df_copy['nodes_set'] = df_copy['nodes'].apply(set)
        
        # Store in session state
        st.session_state.df_with_sets = df_copy
        st.session_state.nodes_sets_computed = True
        
        return df_copy
    else:
        return st.session_state.df_with_sets

def filter_by_nodes(df, target_nodes):
    """
    Filter DataFrame to rows where nodes column contains any of the target node IDs.
    
    Args:
        df: DataFrame with 'nodes' column (list of node IDs)
        target_nodes: set or list of node IDs to search for
    
    Returns:
        matched_df: DataFrame of local roads that match the route
    """
    if not isinstance(target_nodes, set):
        target_nodes = set(target_nodes)
    df_with_sets = prepare_dataframe_nodes(df)
    
    # create dataframe with matched nodes
    matched_df = df_with_sets[df_with_sets['nodes_set'].apply(lambda x: bool(x & target_nodes))]
    
    st.write(f"Matched DataFrame ({len(matched_df)} rows):")
    matched_df
    
    matched_df = matched_df.drop('nodes_set', axis=1)
    
    return matched_df



# STREAMLIT APP
if 'nodes_sets_computed' not in st.session_state:
    st.session_state.nodes_sets_computed = False

if 'df' not in st.session_state:
    st.session_state.df = df

df = st.session_state.df


st.title("Match Mapbox Directions API Response to the OSM Road Network")


mapbox_token = st.text_input("Mapbox Access Token", value="pk.eyJ1IjoibWJ4c29sdXRpb25zIiwiYSI6ImNqeWhpandmazAyYmYzYnBtZzJxM3hlM2EifQ.ZLKIqBxG97_HklFj0_1RBQ", type="password")

# Input origin and destination
col1, col2 = st.columns(2)

with col1:
    st.subheader("Origin")
    origin_lon = st.number_input("Origin Longitude", value=-117.154738, format="%.6f")
    origin_lat = st.number_input("Origin Latitude", value=32.712152, format="%.6f")

with col2:
    st.subheader("Destination")
    dest_lon = st.number_input("Destination Longitude", value=-117.127995, format="%.6f")
    dest_lat = st.number_input("Destination Latitude", value=32.734151, format="%.6f")

# Profile selection
profile = st.selectbox(
    "Travel Profile",
    ["driving", "driving-traffic", "walking", "cycling"],
    index=0
)

if st.button("Get Route"):
    if not mapbox_token:
        st.error("Please enter your Mapbox access token")
    else:
        coordinates = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coordinates}"
        
        params = {
            "access_token": mapbox_token,
            "geometries": "geojson",  
            "overview": "full",
            "steps": "true" 
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                st.session_state['mapbox_data'] = data
                st.session_state['full_route_geometry'] = route["geometry"]["coordinates"]
                
                st.success("Route found! Configure extraction options below.")
                
            else:
                st.error(f"No route found. Error: {data.get('message', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Route processing & display
if 'mapbox_data' in st.session_state:
    data = st.session_state['mapbox_data']
    route = data["routes"][0]
    
    st.subheader("Route Information")
    st.write(f"**Distance:** {route['distance'] / 1000:.2f} km")
    st.write(f"**Duration:** {route['duration'] / 60:.1f} minutes")

    st.divider()
    st.subheader("1. Extract Coordinates")
    
    extraction_mode = st.radio(
        "Coordinate Source",
        ["Routeline Geometry (Full)", "Step Maneuvers", "Intersections"],
        help="Choose which coordinates to extract from the Mapbox response for matching."
    )
    
    extracted_coords = []
    
    if extraction_mode == "Routeline Geometry (Full)":
        extracted_coords = route["geometry"]["coordinates"]
        
    elif extraction_mode == "Step Maneuvers":
        for leg in route.get('legs', []):
            for step in leg.get('steps', []):
                if 'maneuver' in step and 'location' in step['maneuver']:
                    extracted_coords.append(step['maneuver']['location'])
                    
    elif extraction_mode == "Intersections":
        for leg in route.get('legs', []):
            for step in leg.get('steps', []):
                for intersection in step.get('intersections', []):
                    if 'location' in intersection:
                        extracted_coords.append(intersection['location'])

    # Update session state with extracted coords
    st.session_state['route_coordinates'] = extracted_coords

    st.write(f"**Extracted Points:** {len(extracted_coords)}")
    
    with st.expander("View Full Mapbox API Response"):
        st.json(data)

    coords_string = ";".join([f"{c[0]},{c[1]}" for c in extracted_coords])
    
    st.info("Copy coordinates for OSRM API request:")
    st.code(coords_string, language="text")

st.divider()

layers = []
highlight_layer = None
selected_way_ids = []

map_col, table_col = st.columns([3, 2])


# --- DATA PREPARATION FOR LAYERS ---

if 'matched_df' in st.session_state and not st.session_state['matched_df'].empty:
    with table_col:
        st.subheader("2. Map Matching with OSRM")
        if 'route_coordinates' in st.session_state:
            st.info(f"Route has ${len(st.session_state['route_coordinates'])} coordinate points. OSRM limits to ~100 points.")

            max_coords = st.number_input("Max coordinates to send to OSRM", min_value=1, max_value=len(st.session_state['route_coordinates']), value=min(100, len(st.session_state['route_coordinates'])))

            if st.button("Get OSM IDs via OSRM Map Matching"):
              if len(st.session_state['route_coordinates']) > max_coords:
                step = len(st.session_state['route_coordinates']) // max_coords
                sampled_coords = st.session_state['route_coordinates'][::step][:max_coords]
              else:
                sampled_coords = st.session_state['route_coordinates']

              coord_string = ";".join([f"{lon},{lat}" for lon, lat in sampled_coords])
            #   st.write(len(sampled_coords))

              osrm_url = f"https://router.project-osrm.org/match/v1/driving/{coord_string}?steps=false&geometries=geojson&overview=full&annotations=nodes"

              try:
                with st.spinner("Matching route to OSM road network..."):
                  print(f"Sending request to: {osrm_url}")
                  osrm_response = requests.get(osrm_url)
                  print(f"Response status: {osrm_response.status_code}")
                  osrm_response.raise_for_status()
                  osrm_data = osrm_response.json()
                  st.session_state['osrm_data'] = osrm_data
                  st.session_state['sampled_coords'] = sampled_coords

                  tracepoints = osrm_data.get('tracepoints', [])
                  unmatched_rows = []
                  for i, (tp, coord) in enumerate(zip(tracepoints, sampled_coords)):
                      if tp is None:
                          unmatched_rows.append({
                              "original_index": i, 
                              "longitude": coord[0], 
                              "latitude": coord[1]
                          })

                  not_matched_df = pd.DataFrame(unmatched_rows)
                  st.write(f"Not Matched Coordinates ({len(not_matched_df)} rows):")
                  if not not_matched_df.empty:
                      st.dataframe(not_matched_df)
                  else:
                      st.info("All coordinates matched successfully!")

                  st.session_state['not_matched_df'] = not_matched_df

                  target_nodes = extract_node_ids(osrm_data)
                  st.write(f"Found {len(target_nodes)} unique node IDs from API")

                  with st.spinner("Filtering rows..."):
                    matched_df = filter_by_nodes(df, target_nodes)

                  st.success(f"Found {len(matched_df)} matching rows in local DB")
                  st.session_state['matched_df'] = matched_df
                  st.session_state['target_nodes'] = target_nodes

              except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {str(e)}")
              except Exception as e:
                st.error(f"Error: {str(e)}")

        st.divider()
        st.subheader("Matched Nodes Table")
        node_filter = st.text_input("Filter by Node/Way ID", "", help="Type an ID here to filter the table. You can read IDs from the map tooltip.")

        table_df = st.session_state['matched_df'][['id', 'nodes']].copy()
        display_df = table_df.copy()
        display_df['id_str'] = display_df['id'].astype(str)

        if node_filter.strip():
            # Filter logic: Check if filter string is in ID or any Node ID
            mask_id = display_df['id_str'].str.contains(node_filter)
            mask_nodes = table_df['nodes'].apply(lambda ns: any(node_filter in str(n) for n in ns))
            display_df = display_df[mask_id | mask_nodes]

        st.data_editor(
            display_df[['id_str', 'nodes']],
            hide_index=True,
            key="matched_table",
            use_container_width=True,
            num_rows="dynamic",
            disabled=True,
            column_config={
                "id_str": st.column_config.TextColumn("Way ID"),
                "nodes": st.column_config.ListColumn("Node IDs")
            }
        )

        selected_way_ids = st.multiselect(
            "Highlight specific ways",
            options=table_df['id'].tolist(),
            default=[]
        )

    # Add highlight layer based on multiselect
    if selected_way_ids:
        highlight_features = []
        for feat in geojson_data.get('features', []):
            props = feat.get('properties', {})
            way_id = props.get('@id', '')
            if any(way_id == f"way/{sid}" or way_id == str(sid) for sid in selected_way_ids):
                highlight_features.append(feat)

        if highlight_features:
            highlight_layer = pdk.Layer(
                'GeoJsonLayer',
                data={"type": "FeatureCollection", "features": highlight_features},
                opacity=1.0,
                stroked=True,
                filled=False,
                extruded=False,
                get_line_color=[0, 0, 255],
                get_line_width=50,
                line_width_min_pixels=4,
                pickable=True,
                auto_highlight=True
            )
            layers.append(highlight_layer)


# Base layer: all roads
if st.session_state.get('show_roads', True):
    base_layer = pdk.Layer(
        'GeoJsonLayer',
        data=geojson_data,
        opacity=0.6,
        stroked=True,
        filled=False,
        extruded=False,
        wireframe=True,
        get_line_color=[242, 251, 224],
        get_line_width=20,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True
    )
    layers.append(base_layer)

        # Mapbox route line
    if 'full_route_geometry' in st.session_state:
        route_line_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": st.session_state['full_route_geometry']}],
            pickable=True,
            get_color=[0, 0, 255, 100],
            width_scale=1,
            width_min_pixels=3,
            get_path="path",
            get_width=10
        )
        layers.append(route_line_layer)

    # Mapbox extracted coordinates
    if 'route_coordinates' in st.session_state:
        points_data = [{"pos": c} for c in st.session_state['route_coordinates']]
        extracted_points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=points_data,
            get_position="pos",
            get_color=[0, 100, 255, 200], 
            get_radius=8,
            radius_min_pixels=3,
            radius_max_pixels=8,
            pickable=True,
            auto_highlight=True
        )
        layers.append(extracted_points_layer)


# Tracepoints layer (if OSRM data available)
if st.session_state.get('show_tracepoints', True) and 'osrm_data' in st.session_state:
    tracepoints = st.session_state['osrm_data'].get('tracepoints', [])
    matchings = st.session_state['osrm_data'].get('matchings', [])
    sampled_coords = st.session_state.get('sampled_coords', [])
    
    # Add node ids from the matchings 'legs' annotation object
    legs = matchings[0].get('legs', []) if matchings else []
    
    if tracepoints and sampled_coords:
        
        tracepoints_geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        # Iterate over both the original input coordinates and the returned tracepoints
        for i, (tp, coord) in enumerate(zip(tracepoints, sampled_coords)):
            is_match = tp is not None
            
            props = {
                "name": f"Point {i}",
                "waypoint_index": str(i),
                "distance": "N/A",
                "nodes": [],
                "is_match": is_match,
                "tooltip_html": f"<b>Point {i}</b><br/>Status: {'Matched' if is_match else 'Unmatched'}<br/>Coord: [{coord[0]:.5f}, {coord[1]:.5f}]"
            }
            
            # If matched, update properties with OSRM data
            if is_match:
                waypoint_idx = tp.get('waypoint_index', -1)
                relevant_nodes = []
                if waypoint_idx >= 0 and waypoint_idx < len(legs):
                    relevant_nodes = legs[waypoint_idx].get('annotation', {}).get('nodes', [])
                    relevant_nodes = sorted(list(set([int(n) for n in relevant_nodes])))
                    print(f"Relevant nodes: {relevant_nodes}")
                distance_str = f"{tp.get('distance', 0):.1f}m"
                props.update({
                    "name": tp.get('name', 'Unknown'),
                    "distance": distance_str,
                    "nodes": relevant_nodes,
                    "tooltip_html": f"<b>{tp.get('name', 'Unknown')}</b><br/>Status: Matched<br/>Waypoint Index: {waypoint_idx}<br/>Distance: {distance_str}<br/>Node IDs: {relevant_nodes}"
                })
                # Use the SNAPPED location for matched points
                location = tp['location']
            else:
                # Use the ORIGINAL input location for unmatched points
                location = list(coord)

            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": location
                }
            }
            tracepoints_geojson['features'].append(feature)
        
        tracepoints_data = tracepoints_geojson
        
        tracepoints_layer = pdk.Layer(
            'GeoJsonLayer',
            data=tracepoints_data,
            opacity=0.9,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color="[properties.is_match ? 0 : 255, properties.is_match ? 255 : 0, 0]",
            get_line_color=[0, 0, 0],
            get_line_width=2, 
            point_radius_min_pixels=6,
            point_radius_max_pixels=12,
            pickable=True,
            auto_highlight=True
        )
        layers.append(tracepoints_layer)
        
        st.info(f"Showing {len(tracepoints_geojson['features'])} tracepoints (Green=Matched, Red=Unmatched)")



# --- MAP COLUMN ---

with map_col:
    view_state = pdk.ViewState(
        latitude=32.73,
        longitude=-117.14,
        zoom=12,
        pitch=0,
        bearing=0
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,
        tooltip={
            "html": "{tooltip_html}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
            }
        },

    )

    st.pydeck_chart(r, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.show_roads = st.checkbox("Show OSM Roads", value=st.session_state.get('show_roads', True))
    with col2:
        st.session_state.show_tracepoints = st.checkbox("Show OSRM Tracepoints", value=st.session_state.get('show_tracepoints', True))


if st.button("Reset cache"):
  st.session_state.nodes_sets_computed = False
  if 'df_with_sets' in st.session_state:
    del st.session_state.df_with_sets
  st.rerun()
