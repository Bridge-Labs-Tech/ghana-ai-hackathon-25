#!/usr/bin/env python3
"""
Ghana Transport Dataset Explorer
Comprehensive analysis of Accra's GTFS transport data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GhanaTransportExplorer:
    def __init__(self, dataset_path='dataset/'):
        """Initialize the transport data explorer"""
        self.dataset_path = dataset_path
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load all GTFS datasets"""
        print("Loading GTFS datasets...")
        
        try:
            # Load core datasets
            self.data['agency'] = pd.read_csv(f'{self.dataset_path}agency.txt')
            self.data['routes'] = pd.read_csv(f'{self.dataset_path}routes.txt')
            self.data['stops'] = pd.read_csv(f'{self.dataset_path}stops.txt')
            self.data['trips'] = pd.read_csv(f'{self.dataset_path}trips.txt')
            self.data['stop_times'] = pd.read_csv(f'{self.dataset_path}stop_times.txt')
            self.data['calendar'] = pd.read_csv(f'{self.dataset_path}calendar.txt')
            self.data['fare_attributes'] = pd.read_csv(f'{self.dataset_path}fare_attributes.txt')
            self.data['fare_rules'] = pd.read_csv(f'{self.dataset_path}fare_rules.txt')
            
            print("✅ All datasets loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading dataset: {e}")
            return
            
    def basic_statistics(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*60)
        print("📊 GHANA TRANSPORT DATASET STATISTICS")
        print("="*60)
        
        stats = {
            'Agencies': len(self.data['agency']),
            'Routes': len(self.data['routes']),
            'Stops': len(self.data['stops']),
            'Trips': len(self.data['trips']),
            'Stop Times': len(self.data['stop_times']),
            'Fare Rules': len(self.data['fare_rules'])
        }
        
        for key, value in stats.items():
            print(f"{key:15}: {value:,}")
            
        print("\n📍 Geographic Coverage:")
        lat_range = (self.data['stops']['stop_lat'].min(), self.data['stops']['stop_lat'].max())
        lon_range = (self.data['stops']['stop_lon'].min(), self.data['stops']['stop_lon'].max())
        print(f"Latitude Range:  {lat_range[0]:.4f} to {lat_range[1]:.4f}")
        print(f"Longitude Range: {lon_range[0]:.4f} to {lon_range[1]:.4f}")
        
    def agency_analysis(self):
        """Analyze transport agencies"""
        print("\n" + "="*60)
        print("🏢 TRANSPORT AGENCIES ANALYSIS")
        print("="*60)
        
        agencies = self.data['agency']
        
        # Agency count by type
        print(f"Total Agencies: {len(agencies)}")
        
        # Show some agency names
        print("\nSample Agencies:")
        for i, (_, agency) in enumerate(agencies.head(10).iterrows()):
            name = agency['agency_name'] if pd.notna(agency['agency_name']) else "Unnamed"
            print(f"  {i+1:2d}. {name}")
            
        # Agency distribution
        agency_routes = self.data['routes'].groupby('agency_id').size()
        print(f"\nRoutes per Agency:")
        print(f"  Average: {agency_routes.mean():.1f}")
        print(f"  Median:  {agency_routes.median():.1f}")
        print(f"  Max:     {agency_routes.max()}")
        
    def route_analysis(self):
        """Analyze bus routes"""
        print("\n" + "="*60)
        print("🚌 ROUTE ANALYSIS")
        print("="*60)
        
        routes = self.data['routes']
        
        print(f"Total Routes: {len(routes)}")
        print(f"Route Types: {routes['route_type'].unique()}")
        
        # Route naming patterns
        route_names = routes['route_short_name'].dropna()
        print(f"\nRoute Naming Patterns:")
        print(f"  Alphanumeric routes: {len(route_names[route_names.str.match(r'^[A-Z0-9]+$', na=False)])}")
        print(f"  Letter routes: {len(route_names[route_names.str.match(r'^[A-Z]+$', na=False)])}")
        
        # Show sample routes
        print("\nSample Routes:")
        for i, (_, route) in enumerate(routes.head(10).iterrows()):
            name = route['route_short_name']
            long_name = route['route_long_name'] if pd.notna(route['route_long_name']) else "No description"
            print(f"  {name:6} - {long_name}")
            
    def stop_analysis(self):
        """Analyze bus stops"""
        print("\n" + "="*60)
        print("🛑 STOP ANALYSIS")
        print("="*60)
        
        stops = self.data['stops']
        
        print(f"Total Stops: {len(stops):,}")
        
        # Geographic distribution
        print(f"\nGeographic Distribution:")
        print(f"  Northernmost: {stops['stop_lat'].max():.4f}°N")
        print(f"  Southernmost: {stops['stop_lat'].min():.4f}°S")
        print(f"  Easternmost:  {stops['stop_lon'].max():.4f}°E")
        print(f"  Westernmost:  {stops['stop_lon'].min():.4f}°W")
        
        # Stop density analysis
        lat_bins = pd.cut(stops['stop_lat'], bins=10)
        lon_bins = pd.cut(stops['stop_lon'], bins=10)
        
        print(f"\nStop Density:")
        print(f"  Most dense latitude range: {lat_bins.value_counts().index[0]}")
        print(f"  Most dense longitude range: {lon_bins.value_counts().index[0]}")
        
        # Terminal vs regular stops
        terminal_stops = stops[stops['stop_name'].str.contains('Terminal', case=False, na=False)]
        print(f"\nTerminal Stops: {len(terminal_stops)}")
        print(f"Regular Stops: {len(stops) - len(terminal_stops)}")
        
    def schedule_analysis(self):
        """Analyze schedules and timing"""
        print("\n" + "="*60)
        print("⏰ SCHEDULE ANALYSIS")
        print("="*60)
        
        stop_times = self.data['stop_times']
        
        print(f"Total Stop Times: {len(stop_times):,}")
        
        # Convert times to datetime for analysis
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S')
        stop_times['departure_time'] = pd.to_datetime(stop_times['departure_time'], format='%H:%M:%S')
        
        # Service hours
        earliest = stop_times['arrival_time'].dt.time.min()
        latest = stop_times['arrival_time'].dt.time.max()
        print(f"\nService Hours:")
        print(f"  Earliest service: {earliest}")
        print(f"  Latest service: {latest}")
        
        # Peak hours analysis
        hour_distribution = stop_times['arrival_time'].dt.hour.value_counts().sort_index()
        peak_hour = hour_distribution.idxmax()
        print(f"\nPeak Hour: {peak_hour}:00 ({hour_distribution[peak_hour]} arrivals)")
        
        # Trip duration analysis
        trip_durations = stop_times.groupby('trip_id').agg({
            'arrival_time': ['min', 'max']
        })
        trip_durations.columns = ['start', 'end']
        trip_durations['duration'] = (trip_durations['end'] - trip_durations['start']).dt.total_seconds() / 60
        
        print(f"\nTrip Duration Statistics (minutes):")
        print(f"  Average: {trip_durations['duration'].mean():.1f}")
        print(f"  Median:  {trip_durations['duration'].median():.1f}")
        print(f"  Max:     {trip_durations['duration'].max():.1f}")
        
    def create_interactive_map(self):
        """Create an interactive map of all stops"""
        print("\n" + "="*60)
        print("🗺️  CREATING INTERACTIVE MAP")
        print("="*60)
        
        stops = self.data['stops']
        
        # Calculate center point
        center_lat = stops['stop_lat'].mean()
        center_lon = stops['stop_lon'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add stops to map
        for _, stop in stops.iterrows():
            popup_text = f"<b>{stop['stop_name']}</b><br>ID: {stop['stop_id']}"
            folium.Marker(
                [stop['stop_lat'], stop['stop_lon']],
                popup=popup_text,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
        # Save map
        map_file = 'output/accra_transport_map.html'
        m.save(map_file)
        print(f"✅ Interactive map saved as: {map_file}")
        print(f"   Open this file in your browser to explore the transport network")
        
    def identify_optimization_opportunities(self):
        """Identify potential optimization opportunities"""
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION OPPORTUNITIES")
        print("="*60)
        
        # 1. Stop density analysis
        stops = self.data['stops']
        print("1. STOP DENSITY ANALYSIS")
        print("   - High density areas may need route consolidation")
        print("   - Low density areas may need new routes")
        
        # 2. Route coverage analysis
        routes = self.data['routes']
        print("\n2. ROUTE COVERAGE ANALYSIS")
        print(f"   - {len(routes)} routes serving the city")
        print("   - Opportunity: Optimize route overlap")
        
        # 3. Schedule optimization
        stop_times = self.data['stop_times']
        print("\n3. SCHEDULE OPTIMIZATION")
        print("   - Analyze headways between consecutive trips")
        print("   - Identify gaps in service")
        
        # 4. Geographic gaps
        print("\n4. GEOGRAPHIC GAPS")
        print("   - Areas far from existing stops")
        print("   - Opportunity: Add new stops or routes")
        
        # 5. Agency coordination
        agencies = self.data['agency']
        print(f"\n5. AGENCY COORDINATION")
        print(f"   - {len(agencies)} agencies operating independently")
        print("   - Opportunity: Coordinate schedules and routes")
        
    def generate_insights(self):
        """Generate actionable insights"""
        print("\n" + "="*60)
        print("💡 ACTIONABLE INSIGHTS")
        print("="*60)
        
        insights = [
            "1. **Route Consolidation**: Multiple agencies serve similar routes - opportunity for consolidation",
            "2. **Stop Optimization**: High stop density in central areas suggests potential for stop consolidation",
            "3. **Schedule Coordination**: Different agencies have overlapping schedules - opportunity for coordination",
            "4. **Coverage Gaps**: Some areas have limited service - opportunity for new routes",
            "5. **Peak Hour Optimization**: Service frequency could be optimized during peak hours",
            "6. **Terminal Efficiency**: Terminal stops could be optimized for better passenger flow",
            "7. **Fare Integration**: Multiple fare systems could be integrated for better user experience",
            "8. **Real-time Updates**: Current static schedules could be enhanced with real-time updates"
        ]
        
        for insight in insights:
            print(f"   {insight}")
            
    def run_full_analysis(self):
        """Run complete dataset analysis"""
        print("🚀 GHANA TRANSPORT DATASET EXPLORER")
        print("="*60)
        
        self.basic_statistics()
        self.agency_analysis()
        self.route_analysis()
        self.stop_analysis()
        self.schedule_analysis()
        self.identify_optimization_opportunities()
        self.generate_insights()
        self.create_interactive_map()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open output/accra_transport_map.html to explore the network")
        print("2. Run route_optimizer.py for AI-powered optimization")
        print("3. Implement demand prediction with demand_predictor.py")
        print("4. Build custom visualizations with visualization.py")

if __name__ == "__main__":
    # Initialize and run the explorer
    explorer = GhanaTransportExplorer()
    explorer.run_full_analysis() 