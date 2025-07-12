#!/usr/bin/env python3
"""
Ghana Transport Visualization Module
Interactive maps and charts for transport data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class GhanaTransportVisualizer:
    """Main class for creating transport visualizations"""
    
    def __init__(self, dataset_path='dataset/'):
        self.dataset_path = dataset_path
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load transport datasets"""
        print("Loading data for visualization...")
        
        try:
            self.data['routes'] = pd.read_csv(f'{self.dataset_path}routes.txt')
            self.data['stops'] = pd.read_csv(f'{self.dataset_path}stops.txt')
            self.data['stop_times'] = pd.read_csv(f'{self.dataset_path}stop_times.txt')
            self.data['trips'] = pd.read_csv(f'{self.dataset_path}trips.txt')
            self.data['agency'] = pd.read_csv(f'{self.dataset_path}agency.txt')
            
            print("✅ Data loaded for visualization!")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return
            
    def create_network_overview_map(self):
        """Create an overview map of the entire transport network"""
        print("\n🗺️ Creating network overview map...")
        
        stops = self.data['stops']
        routes = self.data['routes']
        
        # Calculate center point
        center_lat = stops['stop_lat'].mean()
        center_lon = stops['stop_lon'].mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add stops with different colors based on type
        for _, stop in stops.iterrows():
            # Determine color based on stop name
            if 'Terminal' in stop['stop_name']:
                color = 'red'
                icon = 'info-sign'
            elif 'Station' in stop['stop_name']:
                color = 'blue'
                icon = 'info-sign'
            else:
                color = 'green'
                icon = 'map-marker'
                
            popup_text = f"""
            <b>{stop['stop_name']}</b><br>
            ID: {stop['stop_id']}<br>
            Lat: {stop['stop_lat']:.4f}<br>
            Lon: {stop['stop_lon']:.4f}
            """
            
            folium.Marker(
                [stop['stop_lat'], stop['stop_lon']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
            
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Stop Types</b></p>
        <p><i class="fa fa-map-marker fa-2x" style="color:red"></i> Terminals</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:blue"></i> Stations</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:green"></i> Regular Stops</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        map_file = 'network_overview_map.html'
        m.save(map_file)
        print(f"✅ Network overview map saved as: {map_file}")
        
    def create_agency_analysis_charts(self):
        """Create charts analyzing agency distribution"""
        print("\n📊 Creating agency analysis charts...")
        
        routes = self.data['routes']
        agency = self.data['agency']
        trips = self.data['trips']
        stop_times = self.data['stop_times']
        stops = self.data['stops']
        
        # Merge routes with agency info
        routes_with_agency = routes.merge(agency, on='agency_id', how='left')
        
        # Get geographic coordinates by merging with stops data
        route_trips = routes_with_agency.merge(trips[['route_id', 'trip_id']], on='route_id')
        route_stops = route_trips.merge(stop_times[['trip_id', 'stop_id']], on='trip_id')
        routes_with_coords = route_stops.merge(stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id')
        
        # Agency route count
        agency_routes = routes_with_agency.groupby('agency_name').size().sort_values(ascending=False)
        
        plt.figure(figsize=(15, 10))
        
        # Top agencies by route count
        plt.subplot(2, 2, 1)
        top_agencies = agency_routes.head(15)
        plt.barh(range(len(top_agencies)), top_agencies.values)
        plt.yticks(range(len(top_agencies)), top_agencies.index)
        plt.xlabel('Number of Routes')
        plt.title('Top 15 Agencies by Route Count')
        plt.gca().invert_yaxis()
        
        # Route distribution
        plt.subplot(2, 2, 2)
        route_counts = agency_routes.value_counts()
        plt.pie(route_counts.values, labels=route_counts.index, autopct='%1.1f%%')
        plt.title('Route Distribution by Agency Size')
        
        # Agency geographic distribution
        plt.subplot(2, 2, 3)
        agency_coords = routes_with_coords.groupby('agency_name').agg({
            'stop_lat': 'mean',
            'stop_lon': 'mean'
        }).reset_index()
        
        # Get route counts for sizing
        agency_route_counts = routes_with_agency.groupby('agency_name').size()
        
        # Create scatter plot with proper sizing
        for _, row in agency_coords.iterrows():
            agency_name = row['agency_name']
            route_count = agency_route_counts.get(agency_name, 1)
            plt.scatter(row['stop_lon'], row['stop_lat'], 
                       s=route_count * 20, alpha=0.6, label=agency_name)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Agency Geographic Distribution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Route type analysis
        plt.subplot(2, 2, 4)
        route_types = routes['route_type'].value_counts()
        plt.bar(route_types.index, route_types.values)
        plt.xlabel('Route Type')
        plt.ylabel('Count')
        plt.title('Route Type Distribution')
        
        plt.tight_layout()
        plt.savefig('output/agency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Agency analysis charts saved as: output/agency_analysis.png")
        
    def create_schedule_heatmap(self):
        """Create a heatmap of service frequency by hour and stop"""
        print("\n⏰ Creating schedule heatmap...")
        
        stop_times = self.data['stop_times'].copy()
        
        # Convert times to datetime
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S')
        stop_times['hour'] = stop_times['arrival_time'].dt.hour
        
        # Create heatmap data
        heatmap_data = stop_times.groupby(['stop_id', 'hour']).size().unstack(fill_value=0)
        
        # Get top stops by total frequency
        top_stops = stop_times['stop_id'].value_counts().head(20).index
        heatmap_data = heatmap_data.loc[top_stops]
        
        # Get stop names for labels
        stops = self.data['stops']
        stop_names = stops[stops['stop_id'].isin(top_stops)].set_index('stop_id')['stop_name']
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Service Frequency'},
                   xticklabels=True,
                   yticklabels=False)
        
        plt.title('Service Frequency Heatmap (Top 20 Stops)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Stop ID')
        
        # Add stop names as annotations
        for i, stop_id in enumerate(heatmap_data.index):
            if stop_id in stop_names.index:
                stop_name = stop_names[stop_id]
                plt.text(-0.5, i + 0.5, f"{stop_id}: {stop_name[:20]}...", 
                        fontsize=8, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('output/schedule_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Schedule heatmap saved as: output/schedule_heatmap.png")
        
    def create_geographic_analysis(self):
        """Create geographic analysis visualizations"""
        print("\n🌍 Creating geographic analysis...")
        
        stops = self.data['stops']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Stop density map
        axes[0, 0].scatter(stops['stop_lon'], stops['stop_lat'], 
                          alpha=0.6, s=20, c='blue')
        axes[0, 0].set_title('Stop Distribution')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        
        # 2. Stop density histogram
        axes[0, 1].hist2d(stops['stop_lon'], stops['stop_lat'], 
                          bins=50, cmap='viridis')
        axes[0, 1].set_title('Stop Density Heatmap')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        
        # 3. Latitude distribution
        axes[1, 0].hist(stops['stop_lat'], bins=30, alpha=0.7, color='green')
        axes[1, 0].set_title('Latitude Distribution')
        axes[1, 0].set_xlabel('Latitude')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Longitude distribution
        axes[1, 1].hist(stops['stop_lon'], bins=30, alpha=0.7, color='red')
        axes[1, 1].set_title('Longitude Distribution')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('output/geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Geographic analysis saved as: output/geographic_analysis.png")
        
    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard"""
        print("\n📈 Creating interactive dashboard...")
        
        stops = self.data['stops']
        routes = self.data['routes']
        stop_times = self.data['stop_times']
        agency = self.data['agency']
        
        # Convert times for analysis
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S')
        stop_times['hour'] = stop_times['arrival_time'].dt.hour
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stop Distribution', 'Service Frequency by Hour', 
                          'Route Count by Agency', 'Stop Types'),
            specs=[[{"type": "scattergeo"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Geographic stop distribution
        fig.add_trace(
            go.Scattergeo(
                lon=stops['stop_lon'],
                lat=stops['stop_lat'],
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.7),
                text=stops['stop_name'],
                name='Stops'
            ),
            row=1, col=1
        )
        
        # 2. Service frequency by hour
        hourly_freq = stop_times['hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_freq.index, y=hourly_freq.values, name='Service Frequency'),
            row=1, col=2
        )
        
        # 3. Route count by agency (top 10) - merge with agency names
        routes_with_agency = routes.merge(agency[['agency_id', 'agency_name']], on='agency_id', how='left')
        agency_routes = routes_with_agency.groupby('agency_name').size().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=agency_routes.index, y=agency_routes.values, name='Routes per Agency'),
            row=2, col=1
        )
        
        # 4. Stop types
        stop_types = []
        for name in stops['stop_name']:
            if 'Terminal' in name:
                stop_types.append('Terminal')
            elif 'Station' in name:
                stop_types.append('Station')
            else:
                stop_types.append('Regular')
                
        stop_type_counts = pd.Series(stop_types).value_counts()
        fig.add_trace(
            go.Pie(labels=stop_type_counts.index, values=stop_type_counts.values, name='Stop Types'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Ghana Transport Network Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update geographic subplot
        fig.update_geos(
            row=1, col=1,
            center=dict(lat=stops['stop_lat'].mean(), lon=stops['stop_lon'].mean()),
            projection_scale=2
        )
        
        # Save dashboard
        fig.write_html('interactive_dashboard.html')
        print("✅ Interactive dashboard saved as: interactive_dashboard.html")
        
    def create_route_analysis_charts(self):
        """Create charts analyzing route characteristics"""
        print("\n🚌 Creating route analysis charts...")
        
        routes = self.data['routes']
        trips = self.data['trips']
        stop_times = self.data['stop_times']
        
        # Merge data
        route_trips = routes.merge(trips[['route_id', 'trip_id']], on='route_id')
        route_stops = route_trips.merge(stop_times[['trip_id', 'stop_id']], on='trip_id')
        
        # Calculate route statistics
        route_stats = route_stops.groupby('route_id').agg({
            'trip_id': 'nunique',
            'stop_id': 'nunique'
        }).rename(columns={'trip_id': 'n_trips', 'stop_id': 'n_stops'})
        
        plt.figure(figsize=(15, 10))
        
        # 1. Trip count distribution
        plt.subplot(2, 3, 1)
        plt.hist(route_stats['n_trips'], bins=20, alpha=0.7)
        plt.xlabel('Number of Trips')
        plt.ylabel('Number of Routes')
        plt.title('Trip Count Distribution')
        
        # 2. Stop count distribution
        plt.subplot(2, 3, 2)
        plt.hist(route_stats['n_stops'], bins=20, alpha=0.7, color='orange')
        plt.xlabel('Number of Stops')
        plt.ylabel('Number of Routes')
        plt.title('Stop Count Distribution')
        
        # 3. Trips vs Stops scatter
        plt.subplot(2, 3, 3)
        plt.scatter(route_stats['n_stops'], route_stats['n_trips'], alpha=0.6)
        plt.xlabel('Number of Stops')
        plt.ylabel('Number of Trips')
        plt.title('Trips vs Stops')
        
        # 4. Route type distribution
        plt.subplot(2, 3, 4)
        route_types = routes['route_type'].value_counts()
        plt.pie(route_types.values, labels=route_types.index, autopct='%1.1f%%')
        plt.title('Route Type Distribution')
        
        # 5. Agency route count
        plt.subplot(2, 3, 5)
        agency_routes = routes.groupby('agency_id').size().sort_values(ascending=False).head(15)
        plt.barh(range(len(agency_routes)), agency_routes.values)
        plt.yticks(range(len(agency_routes)), agency_routes.index)
        plt.xlabel('Number of Routes')
        plt.title('Routes per Agency (Top 15)')
        plt.gca().invert_yaxis()
        
        # 6. Route efficiency (trips per stop)
        plt.subplot(2, 3, 6)
        route_stats['efficiency'] = route_stats['n_trips'] / route_stats['n_stops']
        plt.hist(route_stats['efficiency'], bins=20, alpha=0.7, color='green')
        plt.xlabel('Trips per Stop')
        plt.ylabel('Number of Routes')
        plt.title('Route Efficiency Distribution')
        
        plt.tight_layout()
        plt.savefig('output/route_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Route analysis charts saved as: output/route_analysis.png")
        
    def run_full_visualization(self):
        """Run complete visualization pipeline"""
        print("🚀 GHANA TRANSPORT VISUALIZATION")
        print("="*60)
        
        # Create all visualizations
        self.create_network_overview_map()
        self.create_agency_analysis_charts()
        self.create_schedule_heatmap()
        self.create_geographic_analysis()
        self.create_interactive_dashboard()
        self.create_route_analysis_charts()
        
        print("\n" + "="*60)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("- output/network_overview_map.html: Interactive network map")
        print("- output/agency_analysis.png: Agency distribution charts")
        print("- output/schedule_heatmap.png: Service frequency heatmap")
        print("- output/geographic_analysis.png: Geographic distribution")
        print("- output/interactive_dashboard.html: Interactive Plotly dashboard")
        print("- output/route_analysis.png: Route characteristics analysis")
        print("\nNext steps:")
        print("1. Open HTML files in browser for interactive exploration")
        print("2. Use insights for route optimization")
        print("3. Share visualizations with stakeholders")
        print("4. Integrate with real-time data")

if __name__ == "__main__":
    # Initialize and run the visualizer
    visualizer = GhanaTransportVisualizer()
    visualizer.run_full_visualization() 