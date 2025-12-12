import streamlit as st
import pydeck as pdk
import json
import pandas as pd
import requests

# SETUP VARS AND DATA
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
        Filtered DataFrame
    """
    if not isinstance(target_nodes, set):
        target_nodes = set(target_nodes)
    

    df_with_sets = prepare_dataframe_nodes(df)
    

    mask = df_with_sets['nodes_set'].apply(lambda x: bool(x & target_nodes))
    filtered_df = df_with_sets[mask].copy()
    
    # Drop the helper column from result
    if 'nodes_set' in filtered_df.columns:
        filtered_df = filtered_df.drop('nodes_set', axis=1)
    
    return filtered_df



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
            "overview": "full" 
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                
                # Extract the coordinates from the route geometry
                coordinates = route["geometry"]["coordinates"]
                
                st.success(f"Route found! {len(coordinates)} coordinate points extracted.")
                
                st.subheader("Route Information")
                st.write(f"**Distance:** {route['distance'] / 1000:.2f} km")
                st.write(f"**Duration:** {route['duration'] / 60:.1f} minutes")
                
                st.subheader("Extracted Coordinates")
                st.write(f"Total points: {len(coordinates)}")
                
                with st.expander("Preview Coordinates (first 10)"):
                    for i, coord in enumerate(coordinates[:10]):
                        st.write(f"Point {i+1}: [{coord[0]:.6f}, {coord[1]:.6f}]")
                
                st.session_state['route_coordinates'] = coordinates
                

                with st.expander("Full Coordinates (JSON format)"):
                    st.json(coordinates)
                
                coords_json = json.dumps(coordinates, indent=2)
                st.download_button(
                    label="Download Coordinates as JSON",
                    data=coords_json,
                    file_name="route_coordinates.json",
                    mime="application/json"
                )
                
                st.info("✓ Coordinates saved to session state as 'route_coordinates'")
                st.code("coordinates = st.session_state['route_coordinates']")
                st.write(st.session_state['route_coordinates'])
                
            else:
                st.error(f"No route found. Error: {data.get('message', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Add Map Matching OSRM Section
st.divider()
st.subheader("Map Matching with OSRM")

if 'route_coordinates' in st.session_state:
  st.info(f"Route has ${len(st.session_state['route_coordinates'])} coordinate points. OSRM may limit to ~100 points.")

  max_coords = st.number_input("Max coordinates to send to OSRM", min_value=1, max_value=len(st.session_state['route_coordinates']), value=min(100, len(st.session_state['route_coordinates'])))

  if st.button("Get Way IDs via OSRM Map Matching"):
    if len(st.session_state['route_coordinates']) > max_coords:
      step = len(st.session_state['route_coordinates']) // max_coords
      sampled_coords = st.session_state['route_coordinates'][::step][:max_coords]
    else:
      sampled_coords = st.session_state['route_coordinates']

    coord_string = ";".join([f"{lon},{lat}" for lon, lat in sampled_coords])
    st.write(len(sampled_coords))
    st.write(coord_string)
  
    osrm_url = f"https://router.project-osrm.org/match/v1/driving/{coord_string}?steps=false&geometries=geojson&overview=full&annotations=nodes"
  
    try:
      with st.spinner("Matching route to OSM road network..."):
        print(f"Sending request to: {osrm_url}")
        osrm_response = requests.get(osrm_url)
        print(f"Response status: {osrm_response.status_code}")
        osrm_response.raise_for_status()
        osrm_data = osrm_response.json()
        st.session_state['osrm_data'] = osrm_data
        target_nodes = extract_node_ids(osrm_data)
        st.write(f"Found {len(target_nodes)} unique node IDs from API")

        with st.spinner("Filtering rows..."):
          filtered_df = filter_by_nodes(df, target_nodes)
        st.success(f"Found {len(filtered_df)} matching rows")
        st.session_state['filtered_df'] = filtered_df
        st.session_state['target_nodes'] = target_nodes
        filtered_df
    except requests.exceptions.RequestException as e:
      st.error(f"API request failed: {str(e)}")
    except Exception as e:
      st.error(f"Error: {str(e)}")

st.subheader("Layer Controls")
col1, col2 = st.columns(2)
with col1:
    show_roads = st.checkbox("Show OSM Roads", value=True)
with col2:
    show_tracepoints = st.checkbox("Show OSRM Tracepoints", value=True)


layers = []
highlight_layer = None
selected_way_ids = []
if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
    st.subheader("Matched ways (select rows to highlight on map)")
    node_filter = st.text_input("Filter table by Node ID (paste from tooltip)", "")

    table_df = st.session_state['filtered_df'][['id', 'nodes']].copy()
    display_df = table_df.copy()
    display_df['id_str'] = display_df['id'].astype(str)
    display_df['nodes_str'] = display_df['nodes'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, tuple, set)) else str(x))

    if node_filter.strip():
        table_df = table_df[table_df['nodes'].apply(lambda ns: any(node_filter in str(n) for n in ns))]
        display_df = display_df.loc[table_df.index]

    st.data_editor(
        display_df[['id_str', 'nodes_str']],
        hide_index=True,
        key="matched_table",
        use_container_width=True,
        num_rows="dynamic",
        disabled=True,
        column_config={
            "id_str": st.column_config.TextColumn("Way ID"),
            "nodes_str": st.column_config.TextColumn("Node IDs")
        }
    )

    selected_way_ids = st.multiselect(
        "Highlight way IDs",
        options=table_df['id'].tolist(),
        default=[]
    )

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
                get_line_width=60,
                line_width_min_pixels=4,
                pickable=True,
                auto_highlight=True
            )
            layers.append(highlight_layer)
            st.info(f"Highlighting {len(highlight_features)} selected way(s)")
        else:
            st.info("No matching features found for selected IDs")
    else:
        st.caption("Select one or more rows to highlight.")

# Base layer: all roads
if show_roads:
    base_layer = pdk.Layer(
        'GeoJsonLayer',
        data=geojson_data,
        opacity=0.6,
        stroked=True,
        filled=False,
        extruded=False,
        wireframe=True,
        get_line_color=[255, 0, 0],
        get_line_width=20,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True
    )
    layers.append(base_layer)

# Tracepoints layer (if OSRM data available)
if show_tracepoints and 'osrm_data' in st.session_state:
    tracepoints = st.session_state['osrm_data'].get('tracepoints', [])
    matchings = st.session_state['osrm_data'].get('matchings', [])
    # add node ids from the matchings 'legs' annotation object
    legs = matchings[0].get('legs', [])
    print(legs)
    if tracepoints:
        tracepoints_geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for tp in tracepoints:

            if tp is not None: 
                waypoint_idx = tp.get('waypoint_index', -1)
                relevant_nodes = []
                if waypoint_idx >= 0 and waypoint_idx < len(legs):
                  relevant_nodes = legs[waypoint_idx].get('annotation', {}).get('nodes', [])

                waypoint_idx_str = str(tp.get('waypoint_index', -1))
                distance_str = f"{tp.get('distance', 0):.1f}m"
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": tp.get('name', 'Unknown'),
                        "waypoint_index": waypoint_idx_str,
                        "distance": distance_str,
                        "nodes": relevant_nodes,
                        "tooltip_html": f"<b>{tp.get('name', 'Unknown')}</b><br/>Waypoint Index: {waypoint_idx_str}<br/>Distance: {distance_str}<br/>Node IDs: {relevant_nodes}"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": tp['location']
                    }
                }
                tracepoints_geojson['features'].append(feature)
        
        print(tracepoints_geojson)

        with st.expander("Debug: Tracepoints GeoJSON"):
            st.json(tracepoints_geojson)
        
        tracepoints_layer = pdk.Layer(
            'GeoJsonLayer',
            data=tracepoints_geojson,
            opacity=0.9,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color=[0, 255, 0], 
            get_line_color=[0, 200, 0],
            point_radius_min_pixels=6,
            point_radius_max_pixels=12,
            pickable=True,
            auto_highlight=True
        )
        layers.append(tracepoints_layer)
        
        st.info(f"Showing {len(tracepoints_geojson['features'])} tracepoints")





view_state = pdk.ViewState(
    latitude=32.73,
    longitude=-117.14,
    zoom=12,
    pitch=0,
    bearing=0
)


# Render with selected layers
r = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=None,
    tooltip={
        "html": "{tooltip_html}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
)

st.pydeck_chart(r)

if st.button("Reset cache"):
  st.session_state.nodes_sets_computed = False
  if 'df_with_sets' in st.session_state:
    del st.session_state.df_with_sets
  st.rerun()