#!/usr/bin/env python3
"""
Ghana Transport Route Optimizer
PyTorch-based AI system for optimizing public transport routes
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import folium
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RouteEfficiencyModel(nn.Module):
    """PyTorch model for predicting route efficiency"""
    
    def __init__(self, input_size, hidden_size=64):
        super(RouteEfficiencyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class StopClusteringModel(nn.Module):
    """PyTorch model for stop clustering optimization"""
    
    def __init__(self, n_stops, n_clusters):
        super(StopClusteringModel, self).__init__()
        self.n_stops = n_stops
        self.n_clusters = n_clusters
        # Learnable cluster assignments
        self.cluster_assignments = nn.Parameter(torch.randn(n_stops, n_clusters))
        
    def forward(self, stop_coordinates):
        # Soft assignment to clusters
        assignments = torch.softmax(self.cluster_assignments, dim=1)
        return assignments

class GhanaRouteOptimizer:
    """Main class for route optimization using PyTorch"""
    
    def __init__(self, dataset_path='dataset/'):
        self.dataset_path = dataset_path
        self.data = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_data()
        
    def load_data(self):
        """Load and preprocess transport data"""
        print("Loading transport data for optimization...")
        
        try:
            # Load core datasets
            self.data['routes'] = pd.read_csv(f'{self.dataset_path}routes.txt')
            self.data['stops'] = pd.read_csv(f'{self.dataset_path}stops.txt')
            self.data['stop_times'] = pd.read_csv(f'{self.dataset_path}stop_times.txt')
            self.data['trips'] = pd.read_csv(f'{self.dataset_path}trips.txt')
            self.data['agency'] = pd.read_csv(f'{self.dataset_path}agency.txt')
            
            print("✅ Data loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return
            
    def prepare_route_features(self):
        """Prepare features for route efficiency prediction"""
        print("\n🔧 Preparing route features...")
        
        routes = self.data['routes']
        stops = self.data['stops']
        stop_times = self.data['stop_times']
        
        # Merge data to get route information
        route_features = []
        
        for _, route in routes.iterrows():
            route_id = route['route_id']
            
            # Get trips for this route
            route_trips = self.data['trips'][self.data['trips']['route_id'] == route_id]
            
            if len(route_trips) == 0:
                continue
                
            # Get stop times for this route
            trip_ids = route_trips['trip_id'].tolist()
            route_stop_times = stop_times[stop_times['trip_id'].isin(trip_ids)]
            
            if len(route_stop_times) == 0:
                continue
                
            # Calculate route features
            features = {
                'route_id': route_id,
                'agency_id': route['agency_id'],
                'route_type': route['route_type'],
                'n_trips': len(route_trips),
                'n_stops': len(route_stop_times['stop_id'].unique()),
                'avg_stop_sequence': route_stop_times['stop_sequence'].mean(),
                'max_stop_sequence': route_stop_times['stop_sequence'].max()
            }
            
            # Calculate geographic features
            route_stops = route_stop_times['stop_id'].unique()
            route_stop_coords = stops[stops['stop_id'].isin(route_stops)]
            
            if len(route_stop_coords) > 0:
                features.update({
                    'lat_range': route_stop_coords['stop_lat'].max() - route_stop_coords['stop_lat'].min(),
                    'lon_range': route_stop_coords['stop_lon'].max() - route_stop_coords['stop_lon'].min(),
                    'avg_lat': route_stop_coords['stop_lat'].mean(),
                    'avg_lon': route_stop_coords['stop_lon'].mean()
                })
            else:
                features.update({
                    'lat_range': 0, 'lon_range': 0, 'avg_lat': 0, 'avg_lon': 0
                })
            
            route_features.append(features)
            
        self.route_features_df = pd.DataFrame(route_features)
        print(f"✅ Prepared features for {len(self.route_features_df)} routes")
        
    def train_efficiency_model(self):
        """Train PyTorch model for route efficiency prediction"""
        print("\n🤖 Training Route Efficiency Model...")
        
        if not hasattr(self, 'route_features_df'):
            self.prepare_route_features()
            
        # Prepare features for training
        feature_cols = ['agency_id', 'route_type', 'n_trips', 'n_stops', 
                       'avg_stop_sequence', 'max_stop_sequence', 
                       'lat_range', 'lon_range', 'avg_lat', 'avg_lon']
        
        X = self.route_features_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create synthetic efficiency scores (in real scenario, these would come from actual data)
        # Higher efficiency = lower score (better routes)
        efficiency_scores = (
            self.route_features_df['n_stops'] * 0.3 +  # More stops = less efficient
            self.route_features_df['lat_range'] * 0.2 +  # Longer routes = less efficient
            self.route_features_df['lon_range'] * 0.2 +  # Wider routes = less efficient
            np.random.normal(0, 0.1, len(self.route_features_df))  # Add some noise
        )
        
        # Normalize efficiency scores
        efficiency_scores = (efficiency_scores - efficiency_scores.mean()) / efficiency_scores.std()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(efficiency_scores).to(self.device)
        
        # Initialize model
        input_size = len(feature_cols)
        model = RouteEfficiencyModel(input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        n_epochs = 100
        train_losses = []
        
        print("Training progress:")
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
                
        self.efficiency_model = model
        self.feature_scaler = scaler
        self.feature_cols = feature_cols
        
        print("✅ Route efficiency model trained successfully!")
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('Route Efficiency Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('output/training_loss.png')
        plt.close()
        
    def optimize_stop_clustering(self, n_clusters=50):
        """Optimize stop clustering using PyTorch"""
        print(f"\n🎯 Optimizing Stop Clustering ({n_clusters} clusters)...")
        
        stops = self.data['stops']
        coordinates = stops[['stop_lat', 'stop_lon']].values
        
        # Normalize coordinates
        scaler = StandardScaler()
        coordinates_scaled = scaler.fit_transform(coordinates)
        
        # Convert to PyTorch tensor
        coords_tensor = torch.FloatTensor(coordinates_scaled).to(self.device)
        
        # Initialize clustering model
        clustering_model = StopClusteringModel(len(stops), n_clusters).to(self.device)
        optimizer = optim.Adam(clustering_model.parameters(), lr=0.01)
        
        # Training loop for clustering
        n_epochs = 200
        print("Training clustering model...")
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Get cluster assignments
            assignments = clustering_model(coords_tensor)
            
            # Calculate clustering loss (minimize within-cluster distance)
            loss = 0
            for i in range(n_clusters):
                cluster_mask = assignments[:, i] > 0.1  # Soft threshold
                if cluster_mask.sum() > 0:
                    cluster_coords = coords_tensor[cluster_mask]
                    cluster_center = cluster_coords.mean(dim=0)
                    cluster_distances = torch.norm(cluster_coords - cluster_center, dim=1)
                    loss += (cluster_distances * assignments[cluster_mask, i]).sum()
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
                
        # Get final cluster assignments
        with torch.no_grad():
            final_assignments = clustering_model(coords_tensor)
            cluster_ids = torch.argmax(final_assignments, dim=1).cpu().numpy()
            
        # Add cluster information to stops
        stops_with_clusters = stops.copy()
        stops_with_clusters['cluster_id'] = cluster_ids
        
        self.stops_with_clusters = stops_with_clusters
        self.clustering_model = clustering_model
        
        print("✅ Stop clustering optimization complete!")
        
        # Analyze clustering results
        cluster_sizes = stops_with_clusters['cluster_id'].value_counts()
        print(f"\nCluster Analysis:")
        print(f"  Average cluster size: {cluster_sizes.mean():.1f}")
        print(f"  Largest cluster: {cluster_sizes.max()}")
        print(f"  Smallest cluster: {cluster_sizes.min()}")
        
    def create_optimization_visualization(self):
        """Create visualization of optimization results"""
        print("\n🗺️ Creating optimization visualization...")
        
        if not hasattr(self, 'stops_with_clusters'):
            print("❌ Run optimize_stop_clustering() first")
            return
            
        stops = self.stops_with_clusters
        
        # Create map
        center_lat = stops['stop_lat'].mean()
        center_lon = stops['stop_lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Color palette for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
                 'gray', 'black', 'lightgray']
        
        # Add stops colored by cluster
        for _, stop in stops.iterrows():
            cluster_id = stop['cluster_id']
            color = colors[cluster_id % len(colors)]
            
            popup_text = f"<b>{stop['stop_name']}</b><br>Cluster: {cluster_id}"
            folium.Marker(
                [stop['stop_lat'], stop['stop_lon']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
            
        # Save map
        map_file = 'output/optimized_stops_map.html'
        m.save(map_file)
        print(f"✅ Optimization visualization saved as: {map_file}")
        
    def suggest_route_improvements(self):
        """Suggest route improvements based on analysis"""
        print("\n💡 ROUTE IMPROVEMENT SUGGESTIONS")
        print("="*50)
        
        if not hasattr(self, 'route_features_df'):
            self.prepare_route_features()
            
        # Analyze route characteristics
        routes = self.route_features_df
        
        print("1. ROUTE CONSOLIDATION OPPORTUNITIES:")
        # Find routes with similar characteristics
        similar_routes = routes[routes['n_stops'] < routes['n_stops'].quantile(0.25)]
        print(f"   - {len(similar_routes)} routes could be consolidated (low stop count)")
        
        print("\n2. EFFICIENCY IMPROVEMENTS:")
        # Routes with high geographic spread
        wide_routes = routes[routes['lat_range'] > routes['lat_range'].quantile(0.75)]
        print(f"   - {len(wide_routes)} routes have excessive geographic spread")
        
        print("\n3. STOP OPTIMIZATION:")
        if hasattr(self, 'stops_with_clusters'):
            cluster_analysis = self.stops_with_clusters['cluster_id'].value_counts()
            small_clusters = cluster_analysis[cluster_analysis < 5]
            print(f"   - {len(small_clusters)} clusters have very few stops (potential for consolidation)")
            
        print("\n4. AGENCY COORDINATION:")
        agency_routes = routes.groupby('agency_id').size()
        small_agencies = agency_routes[agency_routes < 5]
        print(f"   - {len(small_agencies)} agencies operate very few routes (coordination opportunity)")
        
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print("\n📊 OPTIMIZATION REPORT")
        print("="*50)
        
        report = {
            "Total Routes": len(self.data['routes']),
            "Total Stops": len(self.data['stops']),
            "Total Agencies": len(self.data['agency']),
            "Model Trained": hasattr(self, 'efficiency_model'),
            "Clustering Applied": hasattr(self, 'stops_with_clusters')
        }
        
        for key, value in report.items():
            print(f"{key:20}: {value}")
            
        if hasattr(self, 'route_features_df'):
            print(f"\nRoute Statistics:")
            print(f"  Average stops per route: {self.route_features_df['n_stops'].mean():.1f}")
            print(f"  Average trips per route: {self.route_features_df['n_trips'].mean():.1f}")
            
        print(f"\nOptimization Recommendations:")
        print("1. Implement route efficiency scoring")
        print("2. Consolidate stops in dense clusters")
        print("3. Coordinate schedules across agencies")
        print("4. Optimize route coverage gaps")
        print("5. Implement real-time optimization")
        
    def run_full_optimization(self):
        """Run complete optimization pipeline"""
        print("🚀 GHANA TRANSPORT ROUTE OPTIMIZER")
        print("="*60)
        
        # Prepare data
        self.prepare_route_features()
        
        # Train efficiency model
        self.train_efficiency_model()
        
        # Optimize stop clustering
        self.optimize_stop_clustering(n_clusters=50)
        
        # Create visualizations
        self.create_optimization_visualization()
        
        # Generate insights
        self.suggest_route_improvements()
        self.generate_optimization_report()
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("- output/training_loss.png: Model training progress")
        print("- output/optimized_stops_map.html: Interactive cluster map")
        print("\nNext steps:")
        print("1. Review optimization suggestions")
        print("2. Implement demand prediction")
        print("3. Build real-time optimization system")
        print("4. Deploy for transport authorities")

if __name__ == "__main__":
    # Initialize and run the optimizer
    optimizer = GhanaRouteOptimizer()
    optimizer.run_full_optimization() 