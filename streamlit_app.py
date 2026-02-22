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
# GEOGRAPHIC CALCULATIONS (Same as vmg_analyzer.py)
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
    
    # Calculate bearing to waypoint from each position
    df['bearing_to_waypoint'] = df.apply(
        lambda row: calculate_bearing(
            row['latitude'], 
            row['longitude'], 
            waypoint_lat, 
            waypoint_lon
        ), 
        axis=1
    )
    
    # Calculate distance to waypoint
    df['distance_to_waypoint'] = df.apply(
        lambda row: calculate_distance(
            row['latitude'], 
            row['longitude'], 
            waypoint_lat, 
            waypoint_lon
        ), 
        axis=1
    )
    
    # Calculate angle difference
    df['angle_to_waypoint'] = df.apply(
        lambda row: normalize_angle(row['bearing_to_waypoint'] - row['COG']), 
        axis=1
    )
    
    # Calculate VMG
    df['VMG'] = df['SOG'] * np.cos(np.radians(df['angle_to_waypoint']))
    
    return df

def load_and_clean_data(uploaded_file):
    """Load CSV and standardize column names"""
    df = pd.read_csv(uploaded_file)
    
   # Column name mappings
    column_mappings = {
        # Latitude
        'lat': 'latitude', 
        'Lat': 'latitude', 
        'Latitude': 'latitude', 
        'LAT': 'latitude',
        
        # Longitude
        'lon': 'longitude', 
        'Lon': 'longitude', 
        'Longitude': 'longitude', 
        'LON': 'longitude', 
        'lng': 'longitude',
        
        # Speed Over Ground (SOG)
        'sog': 'SOG', 
        'SOG': 'SOG',
        'sog_kts': 'SOG',       # <-- ADDED THIS
        'speed': 'SOG', 
        'Speed': 'SOG', 
        'speed_knots': 'SOG',
        
        # Course Over Ground (COG)
        'cog': 'COG', 
        'COG': 'COG',
        'course': 'COG', 
        'Course': 'COG', 
        'heading': 'COG',
        
        # Timestamp
        'time': 'timestamp', 
        'Time': 'timestamp', 
        'timestamp': 'timestamp', 
        'Timestamp': 'timestamp',
    }
    
    df.rename(columns=column_mappings, inplace=True)
    
    # Check for required columns
    required_columns = ['latitude', 'longitude', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return None, missing_columns, df.columns.tolist()
    
    # Remove rows with missing data
    df = df.dropna(subset=['latitude', 'longitude', 'SOG', 'COG'])
    
    return df, None, None

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page config
st.set_page_config(
    page_title="Vakaros VMG Analyzer",
    page_icon="⛵",
    layout="wide"
)

# Header
st.title("⛵ Vakaros Atlas 2 VMG Analyzer")
st.markdown("**Analyze Velocity Made Good (VMG) from your Atlas 2 GPS sailing data**")
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Atlas 2 CSV export",
        type=['csv'],
        help="Export your session from the Vakaros Connect app"
    )
    
    st.markdown("---")
    st.header("📍 Waypoint")
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
    st.markdown("**💡 Tip:** Get coordinates from Google Maps by right-clicking a location")

# Main content
if uploaded_file is None:
    # Instructions when no file uploaded
    st.info("👈 Upload your Atlas 2 CSV file to get started")
    
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Export data** from Vakaros Connect app (Sessions → Export → CSV)
    2. **Upload the CSV** using the sidebar
    3. **Enter waypoint coordinates** (your destination or mark)
    4. **View instant analysis** with charts and statistics
    5. **Download results** as a new CSV file
    """)
    
    st.markdown("### 📊 What is VMG?")
    st.markdown("""
    **Velocity Made Good (VMG)** measures how fast you're moving toward a waypoint.
    
    - **Positive VMG** = Moving toward the waypoint ✅
    - **Negative VMG** = Moving away from the waypoint ❌
    - **VMG = SOG × cos(θ)** where θ = angle between your course and bearing to waypoint
    """)
    
    st.markdown("### 📄 CSV Format")
    st.markdown("""
    Your CSV should include these columns (case-insensitive):
    - `latitude` or `lat`
    - `longitude` or `lon`
    - `SOG` or `speed` (Speed Over Ground in knots)
    - `COG` or `course` (Course Over Ground in degrees)
    - `timestamp` (optional, for time-based analysis)
    """)
    
else:
    # Process the uploaded file
    with st.spinner("Loading data..."):
        df, missing_cols_cols, available_cols = load_and_clean_data(uploaded_file)
    
    if df is None:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.write("**Available columns in your CSV:**")
        st.write(available_cols)
        st.info("Please ensure your CSV has columns for latitude, longitude, SOG, and COG")
    else:
        st.success(f"✅ Loaded {len(df)} data points")
        
        # Calculate VMG
        with st.spinner("Calculating VMG..."):
            df = calculate_vmg(df, waypoint_lat, waypoint_lon)
        
        # Statistics
        st.markdown("## 📊 Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average VMG",
                f"{df['VMG'].mean():.2f} kts",
                help="Average velocity toward waypoint"
            )
        
        with col2:
            st.metric(
                "Max VMG",
                f"{df['VMG'].max():.2f} kts",
                help="Best velocity toward waypoint"
            )
        
        with col3:
            st.metric(
                "Average SOG",
                f"{df['SOG'].mean():.2f} kts",
                help="Average speed over ground"
            )
        
        with col4:
            distance_gained = df['distance_to_waypoint'].iloc[0] - df['distance_to_waypoint'].iloc[-1]
            st.metric(
                "Distance Gained",
                f"{distance_gained:.2f} NM",
                help="Net progress toward waypoint"
            )
        
        # Additional stats
        col5, col6, col7, col8 = st.columns(4)
        
        positive_vmg = df[df['VMG'] > 0]
        negative_vmg = df[df['VMG'] < 0]
        
        with col5:
            st.metric(
                "Starting Distance",
                f"{df['distance_to_waypoint'].iloc[0]:.2f} NM"
            )
        
        with col6:
            st.metric(
                "Ending Distance",
                f"{df['distance_to_waypoint'].iloc[-1]:.2f} NM"
            )
        
        with col7:
            pct_positive = len(positive_vmg) / len(df) * 100
            st.metric(
                "Positive VMG",
                f"{pct_positive:.1f}%",
                help="Percentage of time moving toward waypoint"
            )
        
        with col8:
            if len(positive_vmg) > 0:
                st.metric(
                    "Avg Positive VMG",
                    f"{positive_vmg['VMG'].mean():.2f} kts"
                )
        
        st.markdown("---")
        
        # Charts
        st.markdown("## 📈 Analysis Charts")
        
        # Create tabs for different views
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
        
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            scatter = ax3.scatter(df['longitude'], df['latitude'], 
                                c=df['VMG'], cmap='RdYlGn', 
                                s=30, alpha=0.8)
            ax3.plot(waypoint_lon, waypoint_lat, 'r*', markersize=20, 
                    label='Waypoint', markeredgecolor='black', markeredgewidth=1)
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
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
        
        st.markdown("---")
        
        # Download results
        st.markdown("## Download Results")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Analysis as CSV",
            data=csv,
            file_name="vmg_analysis_results.csv",
            mime="text/csv",
            help="Download the complete dataset with VMG calculations"
        )
