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
# WAYPOINT AUTO-DETECTION
# ============================================================================

def detect_windward_mark(df, start_time_str, variance_threshold=100):
    """
    Auto-detect windward mark (Waypoint 1) based on course change from initial heading
    """
    
    if 'timestamp' not in df.columns or 'COG' not in df.columns:
        return None, {"error": "Missing required columns for auto-detection"}
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        if start_time_str and start_time_str != "No filter - use all data":
            start_time = pd.to_datetime(start_time_str)
        else:
            start_time = df['timestamp'].min()
        
        # Get first 5 minutes of data after start
        five_min_after = start_time + pd.Timedelta(minutes=5)
        initial_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= five_min_after)]
        
        if len(initial_data) < 10:
            return None, {"error": "Not enough data in first 5 minutes"}
        
        # Calculate average initial course (typically close-hauled)
        avg_initial_course = initial_data['COG'].mean()
        
        # Look for significant course change after the initial period
        later_data = df[df['timestamp'] > five_min_after]
        
        if len(later_data) == 0:
            return None, {"error": "No data after initial 5 minutes"}
        
        later_data = later_data.copy()
        later_data['course_diff'] = later_data['COG'].apply(
            lambda x: min(abs(x - avg_initial_course), 
                         360 - abs(x - avg_initial_course))
        )
        
        # Find first point where course changes by threshold amount
        mark_candidates = later_data[later_data['course_diff'] >= variance_threshold]
        
        if len(mark_candidates) == 0:
            return None, {"error": f"No course change >= {variance_threshold}° detected"}
        
        mark_index = mark_candidates.index[0]
        
        # Use a few points before the course change as the mark location
        if mark_index > 5:
            mark_index -= 5
        
        mark_lat = df.loc[mark_index, 'latitude']
        mark_lon = df.loc[mark_index, 'longitude']
        mark_time = df.loc[mark_index, 'timestamp']
        course_change = mark_candidates.iloc[0]['course_diff']
        
        detection_info = {
            "detected": True,
            "mark_lat": mark_lat,
            "mark_lon": mark_lon,
            "mark_time": mark_time,
            "avg_initial_course": avg_initial_course,
            "course_change": course_change,
            "detection_index": mark_index
        }
        
        return (mark_lat, mark_lon), detection_info
        
    except Exception as e:
        return None, {"error": f"Detection failed: {str(e)}"}

def detect_leeward_mark(df, waypoint1_info, variance_threshold=100):
    """
    Auto-detect leeward mark (Waypoint 2) based on course change after Waypoint 1
    """
    
    if not waypoint1_info.get('detected'):
        return None, {"error": "Waypoint 1 must be detected first"}
    
    if 'timestamp' not in df.columns or 'COG' not in df.columns:
        return None, {"error": "Missing required columns for auto-detection"}
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Start from waypoint 1 detection point
        wp1_index = waypoint1_info['detection_index']
        wp1_time = waypoint1_info['mark_time']
        
        # Get data from waypoint 1 onwards
        df_after_wp1 = df[df.index >= wp1_index].copy()
        
        if len(df_after_wp1) < 20:
            return None, {"error": "Not enough data after Waypoint 1"}
        
        # Get first 5 minutes after Waypoint 1 (downwind leg)
        five_min_after = wp1_time + pd.Timedelta(minutes=5)
        downwind_data = df_after_wp1[(df_after_wp1['timestamp'] >= wp1_time) & 
                                     (df_after_wp1['timestamp'] <= five_min_after)]
        
        if len(downwind_data) < 10:
            return None, {"error": "Not enough data in first 5 minutes after Waypoint 1"}
        
        # Calculate average downwind course
        avg_downwind_course = downwind_data['COG'].mean()
        
        # Look for significant course change after the downwind leg
        later_data = df_after_wp1[df_after_wp1['timestamp'] > five_min_after]
        
        if len(later_data) == 0:
            return None, {"error": "No data after downwind leg"}
        
        later_data = later_data.copy()
        later_data['course_diff'] = later_data['COG'].apply(
            lambda x: min(abs(x - avg_downwind_course), 
                         360 - abs(x - avg_downwind_course))
        )
        
        # Find first point where course changes by threshold amount
        mark_candidates = later_data[later_data['course_diff'] >= variance_threshold]
        
        if len(mark_candidates) == 0:
            return None, {"error": f"No course change >= {variance_threshold}° detected after Waypoint 1"}
        
        mark_index = mark_candidates.index[0]
        
        # Use a few points before the course change as the mark location
        if mark_index > 5:
            mark_index -= 5
        
        mark_lat = df.loc[mark_index, 'latitude']
        mark_lon = df.loc[mark_index, 'longitude']
        mark_time = df.loc[mark_index, 'timestamp']
        course_change = mark_candidates.iloc[0]['course_diff']
        
        detection_info = {
            "detected": True,
            "mark_lat": mark_lat,
            "mark_lon": mark_lon,
            "mark_time": mark_time,
            "avg_downwind_course": avg_downwind_course,
            "course_change": course_change,
            "detection_index": mark_index
        }
        
        return (mark_lat, mark_lon), detection_info
        
    except Exception as e:
        return None, {"error": f"Waypoint 2 detection failed: {str(e)}"}

# ============================================================================
# DATA LOADING AND PROCESSING
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
    if not start_time_str or start_time_str == "No filter - use all data":
        return df, None
    
    if 'timestamp' not in df.columns:
        return df, "No timestamp column found - cannot filter by time"
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        start_time = pd.to_datetime(start_time_str)
        
        original_count = len(df)
        df = df[df['timestamp'] >= start_time]
        
        if len(df) == 0:
            return None, "No data points after the selected start time"
        
        filtered_count = original_count - len(df)
        return df, f"Filtered out {filtered_count} pre-race data points"
    
    except Exception as e:
        return None, f"Error filtering by time: {str(e)}"

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Vakaros VMG Analyzer",
    page_icon="⛵",
    layout="wide"
)

st.title("⛵ Vakaros Atlas 2 VMG Analyzer")
st.markdown("**Analyze Velocity Made Good (VMG) from your Atlas 2 GPS sailing data**")
st.markdown("---")

# Initialize session state
if 'waypoint1_lat' not in st.session_state:
    st.session_state.waypoint1_lat = 42.3601
if 'waypoint1_lon' not in st.session_state:
    st.session_state.waypoint1_lon = -71.0589
if 'waypoint2_lat' not in st.session_state:
    st.session_state.waypoint2_lat = None
if 'waypoint2_lon' not in st.session_state:
    st.session_state.waypoint2_lon = None
if 'wp1_info' not in st.session_state:
    st.session_state.wp1_info = {}
if 'wp2_info' not in st.session_state:
    st.session_state.wp2_info = {}

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Atlas 2 CSV export",
        type=['csv'],
        help="Export your session from the Vakaros Connect app"
    )
    
    st.markdown("---")
    st.header("📍 Waypoints")
    
    waypoint_method = st.radio(
        "Waypoint Selection Method",
        ["Manual Entry", "Auto-Detect Marks"],
        help="Auto-detect finds marks based on course changes"
    )
    
    if waypoint_method == "Manual Entry":
        st.subheader("Waypoint 1 (Windward Mark)")
        waypoint1_lat = st.number_input(
            "WP1 Latitude",
            value=st.session_state.waypoint1_lat,
            format="%.6f",
            help="Decimal degrees (North is positive)"
        )
        
        waypoint1_lon = st.number_input(
            "WP1 Longitude",
            value=st.session_state.waypoint1_lon,
            format="%.6f",
            help="Decimal degrees (West is negative)"
        )
        
        st.subheader("Waypoint 2 (Leeward Mark)")
        use_wp2 = st.checkbox("Add Waypoint 2", value=False)
        
        if use_wp2:
            waypoint2_lat = st.number_input(
                "WP2 Latitude",
                value=42.3500 if st.session_state.waypoint2_lat is None else st.session_state.waypoint2_lat,
                format="%.6f"
            )
            
            waypoint2_lon = st.number_input(
                "WP2 Longitude",
                value=-71.0600 if st.session_state.waypoint2_lon is None else st.session_state.waypoint2_lon,
                format="%.6f"
            )
        else:
            waypoint2_lat = None
            waypoint2_lon = None
    else:
        st.info("🎯 Waypoints will be auto-detected based on course changes")
        
        with st.expander("Auto-Detection Settings"):
            course_change_threshold = st.slider(
                "Course Change Threshold (degrees)",
                min_value=60,
                max_value=140,
                value=100,
                step=5,
                help="Degrees of course change to detect mark rounding"
            )
            
            st.markdown("""
            **Detection Process:**
            1. **Waypoint 1:** Detected when course changes by threshold after first 5 min upwind
            2. **Waypoint 2:** Detected when course changes by threshold after first 5 min downwind from WP1
            """)
        
        waypoint1_lat = st.session_state.waypoint1_lat
        waypoint1_lon = st.session_state.waypoint1_lon
        waypoint2_lat = st.session_state.waypoint2_lat
        waypoint2_lon = st.session_state.waypoint2_lon
    
    # Waypoint selection for analysis
    st.markdown("---")
    st.header("📊 Analysis Target")
    
    if waypoint_method == "Auto-Detect Marks" or (waypoint_method == "Manual Entry" and use_wp2):
        analyze_waypoint = st.radio(
            "Analyze VMG to:",
            ["Waypoint 1 (Windward)", "Waypoint 2 (Leeward)"],
            help="Select which waypoint to analyze VMG toward"
        )
    else:
        analyze_waypoint = "Waypoint 1 (Windward)"
    
    st.markdown("---")
    st.header("🏁 Race Start Time")
    
    race1_start = None
    
    if uploaded_file is not None:
        temp_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        
        timestamp_cols = [col for col in temp_df.columns if 'time' in col.lower()]
        
        if timestamp_cols:
            time_col = timestamp_cols[0]
            temp_df[time_col] = pd.to_datetime(temp_df[time_col])
            
            min_time = temp_df[time_col].min()
            max_time = temp_df[time_col].max()
            
            time_range = pd.date_range(
                start=min_time.floor('1min'),
                end=max_time,
                freq='30S'
            )
            
            time_options = ["No filter - use all data"] + [t.strftime('%Y-%m-%d %H:%M:%S') for t in time_range]
            
            selected_time = st.selectbox(
                "Select Race 1 Start Time",
                options=time_options,
                index=0,
                help="Choose the approximate start time of Race 1"
            )
            
            use_manual = st.checkbox("Enter custom time instead", value=False)
            
            if use_manual:
                race1_start = st.text_input(
                    "Custom Start Time",
                    placeholder="YYYY-MM-DD HH:MM:SS",
                    help=f"Data ranges from {min_time.strftime('%Y-%m-%d %H:%M:%S')} to {max_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                if selected_time != "No filter - use all data":
                    race1_start = selected_time
            
            st.info(f"📅 Data: {min_time.strftime('%H:%M:%S')} to {max_time.strftime('%H:%M:%S')}")
        else:
            st.warning("No timestamp column found in CSV")
            race1_start = None
    else:
        st.info("Upload CSV to select race start time")

# Main content
if uploaded_file is None:
    st.info("👈 Upload your Atlas 2 CSV file to get started")
    
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Export data** from Vakaros Connect app
    2. **Upload the CSV** using the sidebar
    3. **Choose waypoint method:**
       - Manual: Enter coordinates directly
       - Auto-detect: Automatically finds both marks
    4. **Select Race 1 start time**
    5. **View analysis** for upwind and downwind legs
    """)
    
    st.markdown("### 🎯 Auto-Detection Features")
    st.markdown("""
    **Waypoint 1 (Windward Mark):**
    - Detected when boat bears away after upwind leg
    - Based on course change from average heading in first 5 min
    
    **Waypoint 2 (Leeward Mark):**
    - Detected when boat rounds up after downwind leg
    - Based on course change from average heading after WP1
    """)

else:
    with st.spinner("Loading data..."):
        df, missing_cols, available_cols = load_and_clean_data(uploaded_file)
    
    if df is None:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.write("**Available columns in your CSV:**")
        st.write(available_cols)
    else:
        st.success(f"✅ Loaded {len(df)} data points")
        
        # Auto-detect waypoints if selected
        if waypoint_method == "Auto-Detect Marks":
            if race1_start and race1_start != "No filter - use all data":
                
                # Detect Waypoint 1
                with st.spinner("🔍 Detecting Waypoint 1 (Windward Mark)..."):
                    detected_wp1, wp1_info = detect_windward_mark(
                        df, race1_start, course_change_threshold
                    )
                    
                    if detected_wp1:
                        waypoint1_lat, waypoint1_lon = detected_wp1
                        st.session_state.waypoint1_lat = waypoint1_lat
                        st.session_state.waypoint1_lon = waypoint1_lon
                        st.session_state.wp1_info = wp1_info
                        
                        st.success(f"🎯 Waypoint 1 detected at: {waypoint1_lat:.6f}, {waypoint1_lon:.6f}")
                        
                        # Detect Waypoint 2
                        with st.spinner("🔍 Detecting Waypoint 2 (Leeward Mark)..."):
                            detected_wp2, wp2_info = detect_leeward_mark(
                                df, wp1_info, course_change_threshold
                            )
                            
                            if detected_wp2:
                                waypoint2_lat, waypoint2_lon = detected_wp2
                                st.session_state.waypoint2_lat = waypoint2_lat
                                st.session_state.waypoint2_lon = waypoint2_lon
                                st.session_state.wp2_info = wp2_info
                                
                                st.success(f"🎯 Waypoint 2 detected at: {waypoint2_lat:.6f}, {waypoint2_lon:.6f}")
                                
                                # Show detection summary
                                with st.expander("Detection Details"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Waypoint 1 (Windward):**")
                                        st.write(f"- Initial avg course: {wp1_info['avg_initial_course']:.1f}°")
                                        st.write(f"- Course change: {wp1_info['course_change']:.1f}°")
                                        st.write(f"- Time: {wp1_info['mark_time']}")
                                    with col2:
                                        st.write("**Waypoint 2 (Leeward):**")
                                        st.write(f"- Downwind avg course: {wp2_info['avg_downwind_course']:.1f}°")
                                        st.write(f"- Course change: {wp2_info['course_change']:.1f}°")
                                        st.write(f"- Time: {wp2_info['mark_time']}")
                            else:
                                st.warning(f"⚠️ Could not detect Waypoint 2: {wp2_info.get('error')}")
                    else:
                        st.warning(f"⚠️ Could not detect Waypoint 1: {wp1_info.get('error')}")
            else:
                st.warning("⚠️ Please select a race start time for auto-detection")
        
        # Apply race start time filter
        if race1_start and race1_start != "No filter - use all data":
            df, filter_message = filter_from_start(df, race1_start)
            
            if df is None:
                st.error(f"❌ {filter_message}")
                st.stop()
            else:
                st.success(f"🏁 {filter_message}")
        
        # Select which waypoint to analyze
        if analyze_waypoint == "Waypoint 2 (Leeward)" and waypoint2_lat is not None:
            active_waypoint_lat = waypoint2_lat
            active_waypoint_lon = waypoint2_lon
            active_waypoint_name = "Waypoint 2 (Leeward)"
        else:
            active_waypoint_lat = waypoint1_lat
            active_waypoint_lon = waypoint1_lon
            active_waypoint_name = "Waypoint 1 (Windward)"
        
        st.info(f"📍 Analyzing VMG to {active_waypoint_name}: {active_waypoint_lat:.6f}, {active_waypoint_lon:.6f}")
        
        # Show time range
        if 'timestamp' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 Start: {df['timestamp'].min()}")
            with col2:
                st.info(f"📅 End: {df['timestamp'].max()}")
        
        # Calculate VMG
        with st.spinner("Calculating VMG..."):
            df = calculate_vmg(df, active_waypoint_lat, active_waypoint_lon)
        
        # Statistics
        st.markdown("## 📊 Statistics")
        
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
        
        # Charts
        st.markdown("---")
        st.markdown("## 📈 Analysis Charts")
        
        has_heel = 'heel' in df.columns
        
        if has_heel:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["VMG Over Time", "SOG vs VMG", "Track Map", "Distance to Waypoint", "VMG vs Heel"])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["VMG Over Time", "SOG vs VMG", "Track Map", "Distance to Waypoint"])
        
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            scatter = ax3.scatter(df['latitude'], df['longitude'], 
                                c=df['VMG'], cmap='RdYlGn', 
                                s=30, alpha=0.8)
            
            # Plot both waypoints if available
            ax3.plot(waypoint1_lat, waypoint1_lon, 'r*', markersize=20, 
                    label='WP1 (Windward)', markeredgecolor='black', markeredgewidth=1)
            
            if waypoint2_lat is not None:
                ax3.plot(waypoint2_lat, waypoint2_lon, 'b*', markersize=20, 
                        label='WP2 (Leeward)', markeredgecolor='black', markeredgewidth=1)
            
            ax3.set_xlabel('Latitude')
            ax3.set_ylabel('Longitude')
            ax3.set_title(f'Track (colored by VMG to {active_waypoint_name})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axis('equal')
            plt.colorbar(scatter, ax=ax3, label='VMG (knots)')
            st.pyplot(fig3)
            plt.close(fig3)
        
        # Rest of the tabs remain the same...
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            colors = ['green' if v > 0 else 'red' for v in df['VMG']]
            ax1.scatter(df.index, df['VMG'], c=colors, alpha=0.6, s=20)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax1.set_xlabel('Data Point')
            ax1.set_ylabel('VMG (knots)')
            ax1.set_title(f'VMG to {active_waypoint_name} Over Time')
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
        
        with tab4:
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.plot(df.index, df['distance_to_waypoint'], color='purple', linewidth=2)
            ax4.set_xlabel('Data Point')
            ax4.set_ylabel('Distance (NM)')
            ax4.set_title(f'Distance to {active_waypoint_name} Over Time')
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
                ax5a.set_title(f'VMG to {active_waypoint_name} vs Heel Angle')
                ax5a.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax5a, label='SOG (knots)')
                
                df['heel_bucket'] = pd.cut(df['heel'], bins=20)
                heel_analysis = df.groupby('heel_bucket', observed=True).agg({
                    'VMG': 'mean',
                    'SOG': 'mean',
                    'heel': 'mean'
                }).dropna()
                
                if len(heel_analysis) > 0:
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
        
        st.markdown("---")
        
        # Download results
        st.markdown("## 📥 Download Results")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Analysis as CSV",
            data=csv_data,
            file_name="vmg_analysis_results.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built for the Vakaros Atlas 2 | Analysis of sailing performance data</p>
</div>
""", unsafe_allow_html=True)
