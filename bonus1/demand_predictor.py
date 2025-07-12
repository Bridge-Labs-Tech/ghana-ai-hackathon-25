#!/usr/bin/env python3
"""
Ghana Transport Demand Predictor
PyTorch-based AI system for predicting passenger demand
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DemandPredictionModel(nn.Module):
    """PyTorch model for predicting passenger demand"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(DemandPredictionModel, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            
        layers.append(nn.Linear(hidden_size, hidden_size // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size // 2, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series demand prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class GhanaDemandPredictor:
    """Main class for demand prediction using PyTorch"""
    
    def __init__(self, dataset_path='dataset/'):
        self.dataset_path = dataset_path
        self.data = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_data()
        
    def load_data(self):
        """Load and preprocess transport data"""
        print("Loading transport data for demand prediction...")
        
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
            
    def prepare_demand_features(self):
        """Prepare features for demand prediction"""
        print("\n🔧 Preparing demand prediction features...")
        
        stop_times = self.data['stop_times'].copy()
        stops = self.data['stops']
        trips = self.data['trips']
        routes = self.data['routes']
        
        # Convert times to datetime
        stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], format='%H:%M:%S')
        stop_times['departure_time'] = pd.to_datetime(stop_times['departure_time'], format='%H:%M:%S')
        
        # Extract time features
        stop_times['hour'] = stop_times['arrival_time'].dt.hour
        stop_times['minute'] = stop_times['arrival_time'].dt.minute
        stop_times['day_of_week'] = 0  # Assume all data is from same day type
        stop_times['is_peak'] = ((stop_times['hour'] >= 7) & (stop_times['hour'] <= 9)) | \
                               ((stop_times['hour'] >= 17) & (stop_times['hour'] <= 19))
        
        # Merge with route and stop information
        stop_times = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
        stop_times = stop_times.merge(routes[['route_id', 'agency_id']], on='route_id')
        stop_times = stop_times.merge(stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id')
        
        # Calculate demand proxies (in real scenario, this would be actual passenger counts)
        # For now, we'll use frequency of service as a proxy for demand
        demand_features = []
        
        for stop_id in stop_times['stop_id'].unique():
            stop_data = stop_times[stop_times['stop_id'] == stop_id]
            
            # Aggregate by hour
            hourly_demand = stop_data.groupby('hour').size().reset_index(name='demand')
            
            for _, row in hourly_demand.iterrows():
                hour = row['hour']
                demand = row['demand']
                
                # Get stop characteristics
                stop_info = stop_data.iloc[0]
                
                features = {
                    'stop_id': stop_id,
                    'hour': hour,
                    'minute': 0,  # Aggregate to hour level
                    'day_of_week': 0,
                    'is_peak': (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19),
                    'stop_lat': stop_info['stop_lat'],
                    'stop_lon': stop_info['stop_lon'],
                    'agency_id': stop_info['agency_id'],
                    'demand': demand
                }
                
                demand_features.append(features)
                
        self.demand_df = pd.DataFrame(demand_features)
        print(f"✅ Prepared demand features for {len(self.demand_df)} stop-hour combinations")
        
    def train_demand_model(self):
        """Train PyTorch model for demand prediction"""
        print("\n🤖 Training Demand Prediction Model...")
        
        if not hasattr(self, 'demand_df'):
            self.prepare_demand_features()
            
        # Prepare features for training
        feature_cols = ['hour', 'minute', 'day_of_week', 'is_peak', 
                       'stop_lat', 'stop_lon', 'agency_id']
        
        X = self.demand_df[feature_cols].values
        y = self.demand_df['demand'].values
        
        # Encode categorical variables
        le_agency = LabelEncoder()
        X[:, 6] = le_agency.fit_transform(X[:, 6])  # agency_id
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        input_size = len(feature_cols)
        model = DemandPredictionModel(input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        n_epochs = 200
        train_losses = []
        test_losses = []
        
        print("Training progress:")
        for epoch in range(n_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
                
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
                
        self.demand_model = model
        self.demand_scaler = scaler
        self.demand_feature_cols = feature_cols
        self.le_agency = le_agency
        
        print("✅ Demand prediction model trained successfully!")
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Demand Prediction Model Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot predictions vs actual
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).cpu().numpy()
            test_pred = model(X_test_tensor).cpu().numpy()
            
        plt.subplot(1, 2, 2)
        plt.scatter(y_train, train_pred, alpha=0.5, label='Training')
        plt.scatter(y_test, test_pred, alpha=0.5, label='Test')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        plt.xlabel('Actual Demand')
        plt.ylabel('Predicted Demand')
        plt.title('Demand Prediction Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/demand_prediction_results.png')
        plt.close()
        
    def predict_demand_for_stop(self, stop_id, hour, is_peak=False):
        """Predict demand for a specific stop and hour"""
        if not hasattr(self, 'demand_model'):
            print("❌ Train the model first using train_demand_model()")
            return None
            
        # Get stop information
        stops = self.data['stops']
        stop_info = stops[stops['stop_id'] == stop_id]
        
        if len(stop_info) == 0:
            print(f"❌ Stop {stop_id} not found")
            return None
            
        # Prepare features
        features = np.array([
            hour,  # hour
            0,     # minute
            0,     # day_of_week
            1 if is_peak else 0,  # is_peak
            stop_info.iloc[0]['stop_lat'],  # stop_lat
            stop_info.iloc[0]['stop_lon'],  # stop_lon
            0      # agency_id (default)
        ])
        
        # Scale features
        features_scaled = self.demand_scaler.transform(features.reshape(1, -1))
        
        # Predict
        model = self.demand_model
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction = model(features_tensor).cpu().numpy()[0]
            
        return prediction
        
    def analyze_demand_patterns(self):
        """Analyze demand patterns across the network"""
        print("\n📊 ANALYZING DEMAND PATTERNS")
        print("="*50)
        
        if not hasattr(self, 'demand_df'):
            self.prepare_demand_features()
            
        demand_df = self.demand_df
        
        # Hourly demand patterns
        hourly_demand = demand_df.groupby('hour')['demand'].mean()
        peak_hour = hourly_demand.idxmax()
        print(f"Peak Hour: {peak_hour}:00 (avg demand: {hourly_demand[peak_hour]:.1f})")
        
        # Peak vs off-peak analysis
        peak_demand = demand_df[demand_df['is_peak']]['demand'].mean()
        off_peak_demand = demand_df[~demand_df['is_peak']]['demand'].mean()
        print(f"Peak Demand: {peak_demand:.1f} vs Off-Peak: {off_peak_demand:.1f}")
        
        # Geographic demand distribution
        high_demand_stops = demand_df.groupby('stop_id')['demand'].mean().nlargest(10)
        print(f"\nTop 10 High-Demand Stops:")
        for stop_id, demand in high_demand_stops.items():
            stop_name = self.data['stops'][self.data['stops']['stop_id'] == stop_id]['stop_name'].iloc[0]
            print(f"  {stop_name}: {demand:.1f}")
            
        # Agency demand analysis
        agency_demand = demand_df.groupby('agency_id')['demand'].mean().sort_values(ascending=False)
        print(f"\nAgency Demand Analysis:")
        for agency_id, demand in agency_demand.head(5).items():
            agency_name = self.data['agency'][self.data['agency']['agency_id'] == agency_id]['agency_name'].iloc[0]
            print(f"  {agency_name}: {demand:.1f}")
            
    def create_demand_visualization(self):
        """Create visualization of demand patterns"""
        print("\n📈 Creating demand visualization...")
        
        if not hasattr(self, 'demand_df'):
            self.prepare_demand_features()
            
        demand_df = self.demand_df
        
        # Create demand heatmap
        plt.figure(figsize=(15, 10))
        
        # Hourly demand heatmap
        plt.subplot(2, 2, 1)
        hourly_demand = demand_df.groupby('hour')['demand'].mean()
        plt.bar(hourly_demand.index, hourly_demand.values)
        plt.title('Average Demand by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Demand')
        
        # Peak vs off-peak comparison
        plt.subplot(2, 2, 2)
        peak_data = [demand_df[demand_df['is_peak']]['demand'].mean(),
                    demand_df[~demand_df['is_peak']]['demand'].mean()]
        plt.bar(['Peak Hours', 'Off-Peak Hours'], peak_data)
        plt.title('Peak vs Off-Peak Demand')
        plt.ylabel('Average Demand')
        
        # Demand distribution
        plt.subplot(2, 2, 3)
        plt.hist(demand_df['demand'], bins=30, alpha=0.7)
        plt.title('Demand Distribution')
        plt.xlabel('Demand')
        plt.ylabel('Frequency')
        
        # Geographic demand (scatter plot)
        plt.subplot(2, 2, 4)
        avg_demand_by_stop = demand_df.groupby(['stop_lat', 'stop_lon'])['demand'].mean().reset_index()
        plt.scatter(avg_demand_by_stop['stop_lon'], avg_demand_by_stop['stop_lat'], 
                   c=avg_demand_by_stop['demand'], s=50, alpha=0.6, cmap='viridis')
        plt.colorbar(label='Average Demand')
        plt.title('Geographic Demand Distribution')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.tight_layout()
        plt.savefig('output/demand_analysis.png')
        plt.close()
        
        print("✅ Demand visualization saved as: output/demand_analysis.png")
        
    def generate_demand_insights(self):
        """Generate insights from demand analysis"""
        print("\n💡 DEMAND PREDICTION INSIGHTS")
        print("="*50)
        
        insights = [
            "1. **Peak Hour Optimization**: Service frequency should be increased during peak hours",
            "2. **Geographic Hotspots**: High-demand areas need more frequent service",
            "3. **Agency Coordination**: Some agencies serve high-demand routes - opportunity for coordination",
            "4. **Predictive Scheduling**: AI can predict demand to optimize bus allocation",
            "5. **Dynamic Pricing**: Demand-based pricing could optimize passenger distribution",
            "6. **Route Optimization**: High-demand stops should be prioritized in route planning",
            "7. **Capacity Planning**: Vehicle capacity should match predicted demand",
            "8. **Real-time Adjustments**: Live demand prediction can enable dynamic route adjustments"
        ]
        
        for insight in insights:
            print(f"   {insight}")
            
    def run_full_demand_analysis(self):
        """Run complete demand prediction pipeline"""
        print("🚀 GHANA TRANSPORT DEMAND PREDICTOR")
        print("="*60)
        
        # Prepare data
        self.prepare_demand_features()
        
        # Train model
        self.train_demand_model()
        
        # Analyze patterns
        self.analyze_demand_patterns()
        
        # Create visualizations
        self.create_demand_visualization()
        
        # Generate insights
        self.generate_demand_insights()
        
        print("\n" + "="*60)
        print("✅ DEMAND ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("- output/demand_prediction_results.png: Model performance")
        print("- output/demand_analysis.png: Demand patterns visualization")
        print("\nNext steps:")
        print("1. Use predict_demand_for_stop() for specific predictions")
        print("2. Integrate with route optimization")
        print("3. Build real-time demand monitoring")
        print("4. Deploy for dynamic scheduling")

if __name__ == "__main__":
    # Initialize and run the demand predictor
    predictor = GhanaDemandPredictor()
    predictor.run_full_demand_analysis() 