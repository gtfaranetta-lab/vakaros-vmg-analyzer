"""
Vakaros Atlas 2 VMG Analyzer
Calculates and visualizes Velocity Made Good (VMG) to a waypoint
"""
CSV_FILE = 'tim_1st_leg.csv'  # Path to your CSV file
WAYPOINT_LAT = 28.384775200000000          # Your waypoint latitude
WAYPOINT_LON = -80.6494951999999         # Your waypoint longitude


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

# ============================================================================
# GEOGRAPHIC CALCULATIONS
# ============================================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing from point 1 to point 2
    Returns bearing in degrees (0-360)
    """
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
    """
    Calculate distance between two points using Haversine formula
    Returns distance in nautical miles
    """
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

# ============================================================================
# VMG CALCULATION
# ============================================================================

def calculate_vmg(df, waypoint_lat, waypoint_lon):
    """
    Calculate VMG to waypoint for each data point
    
    Parameters:
    - df: DataFrame with columns 'latitude', 'longitude', 'SOG', 'COG'
    - waypoint_lat: waypoint latitude
    - waypoint_lon: waypoint longitude
    
    Returns:
    - DataFrame with added VMG columns
    """
    
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
    
    # Calculate angle difference between COG and bearing to waypoint
    df['angle_to_waypoint'] = df.apply(
        lambda row: normalize_angle(row['bearing_to_waypoint'] - row['COG']), 
        axis=1
    )
    
    # Calculate VMG (SOG * cos(angle))
    df['VMG'] = df['SOG'] * np.cos(np.radians(df['angle_to_waypoint']))
    
    return df

# ============================================================================
# DATA LOADING
# ============================================================================

def load_atlas2_data(filepath):
    """
    Load Vakaros Atlas 2 CSV export
    Handles common column name variations
    """
    df = pd.read_csv(filepath)
    
    # Print available columns for debugging
    print("Available columns in CSV:")
    print(df.columns.tolist())
    print()
    
    # Common column name mappings (adjust based on your export format)
    column_mappings = {
        # Latitude variations
        'lat': 'latitude',
        'Lat': 'latitude',
        'Latitude': 'latitude',
        'LAT': 'latitude',
        
        # Longitude variations
        'lon': 'longitude',
        'Lon': 'longitude',
        'Longitude': 'longitude',
        'LON': 'longitude',
        'lng': 'longitude',
        
        # Speed variations
        'sog': 'SOG',
        'speed': 'SOG',
        'Speed': 'SOG',
        'SOG': 'SOG',
        'speed_knots': 'SOG',
        
        # Course variations
        'cog': 'COG',
        'course': 'COG',
        'Course': 'COG',
        'COG': 'COG',
        'heading': 'COG',
        
        # Time variations
        'time': 'timestamp',
        'Time': 'timestamp',
        'timestamp': 'timestamp',
        'Timestamp': 'timestamp',
        'datetime': 'timestamp',
    }
    
    # Rename columns
    df.rename(columns=column_mappings, inplace=True)
    
    # Verify required columns exist
    required_columns = ['latitude', 'longitude', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print("Please check your CSV format or update column_mappings in the script")
        return None
    
    # Convert timestamp if it exists
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            print("Warning: Could not parse timestamp column")
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['latitude', 'longitude', 'SOG', 'COG'])
    
    print(f"Loaded {len(df)} data points")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "")
    print()
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_vmg_analysis(df, waypoint_lat, waypoint_lon):
    """
    Create comprehensive VMG visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Vakaros Atlas 2 - VMG Analysis', fontsize=16, fontweight='bold')
    
    # Use index if no timestamp
    x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
    x_label = 'Time' if 'timestamp' in df.columns else 'Data Point'
    
    # 1. VMG over time
    ax1 = axes[0, 0]
    colors = ['green' if v > 0 else 'red' for v in df['VMG']]
    ax1.scatter(x_data, df['VMG'], c=colors, alpha=0.6, s=10)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('VMG (knots)')
    ax1.set_title('VMG to Waypoint Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. SOG and VMG comparison
    ax2 = axes[0, 1]
    ax2.plot(x_data, df['SOG'], label='SOG', color='blue', alpha=0.7)
    ax2.plot(x_data, df['VMG'], label='VMG', color='green', alpha=0.7)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Speed (knots)')
    ax2.set_title('SOG vs VMG')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Track plot with VMG color coding
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['longitude'], df['latitude'], 
                         c=df['VMG'], cmap='RdYlGn', 
                         s=20, alpha=0.7)
    ax3.plot(waypoint_lon, waypoint_lat, 'r*', markersize=20, label='Waypoint')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Track (colored by VMG)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    plt.colorbar(scatter, ax=ax3, label='VMG (knots)')
    
    # 4. Distance to waypoint over time
    ax4 = axes[1, 1]
    ax4.plot(x_data, df['distance_to_waypoint'], color='purple')
    ax4.set_xlabel(x_label)
    ax4.set_ylabel('Distance (NM)')
    ax4.set_title('Distance to Waypoint')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistics(df):
    """
    Print VMG statistics
    """
    print("=" * 60)
    print("VMG STATISTICS")
    print("=" * 60)
    print(f"Average VMG:          {df['VMG'].mean():.2f} knots")
    print(f"Maximum VMG:          {df['VMG'].max():.2f} knots")
    print(f"Minimum VMG:          {df['VMG'].min():.2f} knots")
    print(f"Average SOG:          {df['SOG'].mean():.2f} knots")
    print(f"Maximum SOG:          {df['SOG'].max():.2f} knots")
    print()
    
    positive_vmg = df[df['VMG'] > 0]
    negative_vmg = df[df['VMG'] < 0]
    
    print(f"Time with positive VMG: {len(positive_vmg)} points ({len(positive_vmg)/len(df)*100:.1f}%)")
    print(f"Time with negative VMG: {len(negative_vmg)} points ({len(negative_vmg)/len(df)*100:.1f}%)")
    print()
    
    if len(positive_vmg) > 0:
        print(f"Average positive VMG:  {positive_vmg['VMG'].mean():.2f} knots")
    if len(negative_vmg) > 0:
        print(f"Average negative VMG:  {negative_vmg['VMG'].mean():.2f} knots")
    print()
    
    print(f"Starting distance:     {df['distance_to_waypoint'].iloc[0]:.2f} NM")
    print(f"Ending distance:       {df['distance_to_waypoint'].iloc[-1]:.2f} NM")
    print(f"Distance gained:       {df['distance_to_waypoint'].iloc[0] - df['distance_to_waypoint'].iloc[-1]:.2f} NM")
    print("=" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("VAKAROS ATLAS 2 - VMG ANALYZER")
    print("=" * 60)
    print()
    
    # ========== CONFIGURATION ==========
    # UPDATE THESE VALUES:
    
    CSV_FILE = 'atlas2_export.csv'  # Path to your Atlas 2 CSV export
    WAYPOINT_LAT = 42.3601  # Waypoint latitude (decimal degrees)
    WAYPOINT_LON = -71.0589  # Waypoint longitude (decimal degrees)
    
    # ===================================
    
    print(f"CSV File: {CSV_FILE}")
    print(f"Waypoint: {WAYPOINT_LAT}, {WAYPOINT_LON}")
    print()
    
    # Load data
    df = load_atlas2_data(CSV_FILE)
    
    if df is None:
        return
    
    # Calculate VMG
    print("Calculating VMG...")
    df = calculate_vmg(df, WAYPOINT_LAT, WAYPOINT_LON)
    print("VMG calculation complete!")
    print()
    
    # Print statistics
    print_statistics(df)
    print()
    
    # Optional: Save results to CSV
    output_file = 'vmg_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print()
    
    # Plot results
    print("Generating plots...")
    plot_vmg_analysis(df, WAYPOINT_LAT, WAYPOINT_LON)

if __name__ == "__main__":
    main()
