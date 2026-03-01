"""
Vakaros Atlas 2 VMG Analyzer - Streamlit Web App
Interactive web interface for analyzing VMG from Atlas 2 data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================================
# GEOGRAPHIC CALCULATIONS
# ============================================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate initial bearing from point 1 to point 2"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    
    return bearing

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in nautical miles"""
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def normalize_angle(angle):
    """Normalize angle to -180 to +180 range"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def calculate_vmg(df, waypoint_lat, waypoint_lon):
    """Calculate VMG to waypoint for each data point"""
    
    df['bearing_to_waypoint'] = df.apply(
        lambda row: calculate_bearing(
            row['latitude'], 
            row['longitude'], 
            waypoint_lat, 
            waypoint_lon
        ), 
        axis=1
    )
    
    df['distance_to_waypoint'] = df.apply(
        lambda row: calculate_distance(
            row['latitude'], 
            row['longitude'], 
            waypoint_lat, 
            waypoint_lon
        ), 
        axis=1
    )
    
    df['angle_to_waypoint'] = df.apply(
        lambda row: normalize_angle(row['bearing_to_waypoint'] - row['COG']), 
        axis=1
    )
    
    df['VMG'] = df['SOG'] * np.cos(np.radians(df['angle_to_waypoint']))
    
    return df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_clean_data(uploaded_file):
    """Load CSV and standardize column names"""
    df = pd.read_csv(uploaded_file)
    
    column_mappings = {
        'lat': 'latitude', 
        'Lat': 'latitude', 
        'Latitude': 'latitude', 
        'LAT': 'latitude',
        'lon': 'longitude', 
        'Lon': 'longitude', 
        'Longitude': 'longitude', 
        'LON': 'longitude', 
        'lng': 'longitude',
        'sog': 'SOG', 
        'SOG': 'SOG',
        'sog_kts': 'SOG',
        'speed': 'SOG', 
        'Speed': 'SOG', 
        'speed_knots': 'SOG',
        'cog': 'COG', 
        'COG': 'COG',
        'course': 'COG', 
        'Course': 'COG', 
        'heading': 'COG',
        'heel': 'heel',
        'Heel': 'heel',
        'heel_angle': 'heel',
        'heel_deg': 'heel',
        'time': 'timestamp', 
        'Time': 'timestamp', 
        'timestamp': 'timestamp', 
        'Timestamp': 'timestamp',
    }
    
    df.rename(columns=column_mappings, inplace=True)
    
    required_columns = ['latitude', 'longitude', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return None, missing_columns, df.columns.tolist()
    
    df = df.dropna(subset=['latitude', 'longitude', 'SOG', 'COG'])
    
    return df, None, None

def filter_from_start(df, start_time_str):
    """Filter dataframe from race start time onwards"""
    if not start_time_str:
        return df, None
    
    if 'timestamp' not in df.columns:
        return df, "No timestamp column found - cannot filter by time"
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_time = pd.to_datetime(start_time_str)
        df = df[df['timestamp'] >= start_time]
        
        if len(df) == 0:
            return None, "No data points after the start time"
        
        return df, None
    
    except Exception as e:
        return None, f"Error parsing time: {str(e)}. Use format: YYYY-MM-DD HH:MM:SS"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Vakaros VMG Analyzer",
    page_icon="⛵",
    layout="wide"
)

st.title("Vakaros Atlas 2 VMG Analyzer")
st.markdown("**Analyze Velocity Made Good (VMG) from your Atlas 2 GPS sailing data**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Atlas 2 CSV export",
        type=['csv'],
        help="Export your session from the Vakaros Connect app"
    )
    
    st.markdown("---")
    st.header("Waypoint")
    st.markdown("Enter the coordinates of your target waypoint:")
    
    waypoint_lat = st.number_input(
        "Latitude",
        value=42.3601,
        format="%.6f",
        help="Decimal degrees (North is positive)"
    )
    
    waypoint_lon = st.number_input(
        "Longitude",
        value=-71.0589,
        format="%.6f",
        help="Decimal degrees (West is negative)"
    )
    
    st.markdown("---")
    st.header("Race Start Time")
    
    race1_start = st.text_input(
        "Race 1 Start Time",
        placeholder="YYYY-MM-DD HH:MM:SS",
        help="Enter the start time of Race 1. Data before this time will be filtered out."
    )
    
    st.markdown("---")
    st.markdown("**Tip:** Get coordinates from Google Maps by right-clicking a location")

# Main content
if uploaded_file is None:
    st.info("Upload your Atlas 2 CSV file to get started")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. **Export data** from Vakaros Connect app (Sessions - Export - CSV)
    2. **Upload the CSV** using the sidebar
    3. **Enter waypoint coordinates** (your destination or mark)
    4. **Enter Race 1 start time** (optional) to filter pre-race data
    5. **View instant analysis** with charts and statistics
    6. **Download results** as a new CSV file
    """)
    
    st.markdown("### What is VMG?")
    st.markdown("""
    **Velocity Made Good (VMG)** measures how fast you're moving toward a waypoint.
    
    - **Positive VMG** = Moving toward the waypoint
    - **Negative VMG** = Moving away from the waypoint
    - **VMG = SOG x cos(angle)** where angle = difference between your course and bearing to waypoint
    """)

else:
    with st.spinner("Loading data..."):
        df, missing_cols, available_cols = load_and_clean_data(uploaded_file)
    
    if df is None:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("**Available columns in your CSV:**")
        st.write(available_cols)
        st.info("Please ensure your CSV has columns for latitude, longitude, SOG, and COG")
    else:
        st.success(f"Loaded {len(df)} data points")
        
        # Apply race start time filter
        if race1_start:
            df, filter_error = filter_from_start(df, race1_start)
            
            if df is None:
                st.error(f"{filter_error}")
                st.stop()
            else:
                st.success(f"Filtered to {len(df)} data points from race start")
        
        # Show time range
        if 'timestamp' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Start: {df['timestamp'].min()}")
            with col2:
                st.info(f"End: {df['timestamp'].max()}")
        
        # Calculate VMG
        with st.spinner("Calculating VMG..."):
            df = calculate_vmg(df, waypoint_lat, waypoint_lon)
        
        # Statistics
        st.markdown("## Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average VMG", f"{df['VMG'].mean():.2f} kts")
        
        with col2:
            st.metric("Max VMG", f"{df['VMG'].max():.2f} kts")
        
        with col3:
            st.metric("Average SOG", f"{df['SOG'].mean():.2f} kts")
        
        with col4:
            distance_gained = df['distance_to_waypoint'].iloc[0] - df['distance_to_waypoint'].iloc[-1]
            st.metric("Distance Gained", f"{distance_gained:.2f} NM")
        
        col5, col6, col7, col8 = st.columns(4)
        
        positive_vmg = df[df['VMG'] > 0]
        negative_vmg = df[df['VMG'] < 0]
        
        with col5:
            st.metric("Starting Distance", f"{df['distance_to_waypoint'].iloc[0]:.2f} NM")
        
        with col6:
            st.metric("Ending Distance", f"{df['distance_to_waypoint'].iloc[-1]:.2f} NM")
        
        with col7:
            pct_positive = len(positive_vmg) / len(df) * 100
            st.metric("Positive VMG", f"{pct_positive:.1f}%")
        
        with col8:
            if len(positive_vmg) > 0:
                st.metric("Avg Positive VMG", f"{positive_vmg['VMG'].mean():.2f} kts")
        
        st.markdown("---")
        
        # Charts
        st.markdown("## Analysis Charts")
        
        # Check if heel data exists for tab count
        has_heel = 'heel' in df.columns
        
        if has_heel:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["VMG Over Time", "SOG vs VMG", "Track Map", "Distance to Waypoint", "VMG vs Heel"])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["VMG Over Time", "SOG vs VMG", "Track Map", "Distance to Waypoint"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            colors = ['green' if v > 0 else 'red' for v in df['VMG']]
            ax1.scatter(df.index, df['VMG'], c=colors, alpha=0.6, s=20)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax1.set_xlabel('Data Point')
            ax1.set_ylabel('VMG (knots)')
            ax1.set_title('VMG to Waypoint Over Time')
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(df.index, df['SOG'], label='SOG', color='blue', alpha=0.7, linewidth=2)
            ax2.plot(df.index, df['VMG'], label='VMG', color='green', alpha=0.7, linewidth=2)
            ax2.set_xlabel('Data Point')
            ax2.set_ylabel('Speed (knots)')
            ax2.set_title('Speed Over Ground vs VMG')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)
        
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            scatter = ax3.scatter(df['latitude'], df['longitude'], 
                                c=df['VMG'], cmap='RdYlGn', 
                                s=30, alpha=0.8)
            ax3.plot(waypoint_lat, waypoint_lon, 'r*', markersize=20, 
                    label='Waypoint', markeredgecolor='black', markeredgewidth=1)
            ax3.set_ylim(df['longitude'].max(), df['longitude'].min())
            ax3.set_xlabel('Latitude')
            ax3.set_ylabel('Longitude')
            ax3.set_title('Track (colored by VMG)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axis('equal')
            plt.colorbar(scatter, ax=ax3, label='VMG (knots)')
            st.pyplot(fig3)
        
        with tab4:
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.plot(df.index, df['distance_to_waypoint'], color='purple', linewidth=2)
            ax4.set_xlabel('Data Point')
            ax4.set_ylabel('Distance (NM)')
            ax4.set_title('Distance to Waypoint Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.fill_between(df.index, df['distance_to_waypoint'], alpha=0.3, color='purple')
            st.pyplot(fig4)
            plt.close(fig4)
        
        if has_heel:
            with tab5:
                fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 10))
                
                scatter = ax5a.scatter(df['heel'], df['VMG'], 
                                      c=df['SOG'], cmap='viridis', 
                                      s=30, alpha=0.6)
                ax5a.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax5a.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                ax5a.set_xlabel('Heel Angle (degrees)')
                ax5a.set_ylabel('VMG (knots)')
                ax5a.set_title('VMG vs Heel Angle (colored by SOG)')
                ax5a.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax5a, label='SOG (knots)')
                
                df['heel_bucket'] = pd.cut(df['heel'], bins=20)
                heel_analysis = df.groupby('heel_bucket', observed=True).agg({
                    'VMG': 'mean',
                    'SOG': 'mean',
                    'heel': 'mean'
                }).dropna()
                
                ax5b.plot(heel_analysis['heel'], heel_analysis['VMG'], 
                         marker='o', linewidth=2, markersize=6, color='green', label='Avg VMG')
                ax5b.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax5b.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                ax5b.set_xlabel('Heel Angle (degrees)')
                ax5b.set_ylabel('Average VMG (knots)')
                ax5b.set_title('Average VMG by Heel Angle')
                ax5b.grid(True, alpha=0.3)
                ax5b.legend()
                
                st.pyplot(fig5)
                plt.close(fig5)
                
                st.markdown("### Heel Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Heel", f"{df['heel'].mean():.1f} deg")
                
                with col2:
                    st.metric("Max Heel", f"{df['heel'].abs().max():.1f} deg")
                
                with col3:
                    if len(heel_analysis) > 0:
                        optimal_heel = heel_analysis.loc[heel_analysis['VMG'].idxmax(), 'heel']
                        st.metric("Optimal Heel for VMG", f"{optimal_heel:.1f} deg")
                
                port_heel = df[df['heel'] < -2]
                stbd_heel = df[df['heel'] > 2]
                
                if len(port_heel) > 0 and len(stbd_heel) > 0:
                    st.markdown("### Port vs Starboard")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Port Tack** (heel < -2 deg)")
                        st.write(f"Average VMG: {port_heel['VMG'].mean():.2f} kts")
                        st.write(f"Average Heel: {port_heel['heel'].mean():.1f} deg")
                        st.write(f"Data points: {len(port_heel)}")
                    
                    with col2:
                        st.write("**Starboard Tack** (heel > 2 deg)")
                        st.write(f"Average VMG: {stbd_heel['VMG'].mean():.2f} kts")
                        st.write(f"Average Heel: {stbd_heel['heel'].mean():.1f} deg")
                        st.write(f"Data points: {len(stbd_heel)}")
        
        st.markdown("---")
        
        # Download results
        st.markdown("## Download Results")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Analysis as CSV",
            data=csv_data,
            file_name="vmg_analysis_results.csv",
            mime="text/csv"
        )
        
        with st.expander("View Raw Data"):
            st.dataframe(df)

# Footer
st.markdown("---")
st.markdown("Built for the Vakaros Atlas 2")
