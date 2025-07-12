#!/usr/bin/env python3
"""
Ghana Transport Workshop - Quick Start
Simple demonstration of the workshop capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset_explorer import GhanaTransportExplorer
from route_optimizer import GhanaRouteOptimizer
from demand_predictor import GhanaDemandPredictor
from visualization import GhanaTransportVisualizer

def quick_demo():
    """Run a quick demonstration of the workshop features"""
    print("🚀 GHANA TRANSPORT WORKSHOP - QUICK START")
    print("="*60)
    
    # 1. Basic dataset exploration
    print("\n1. 📊 DATASET EXPLORATION")
    print("-" * 30)
    explorer = GhanaTransportExplorer()
    explorer.basic_statistics()
    
    # 2. Route optimization demo
    print("\n2. 🤖 ROUTE OPTIMIZATION DEMO")
    print("-" * 30)
    optimizer = GhanaRouteOptimizer()
    optimizer.prepare_route_features()
    print("   - Route features prepared")
    print("   - Ready for AI optimization")
    
    # 3. Demand prediction demo
    print("\n3. 📈 DEMAND PREDICTION DEMO")
    print("-" * 30)
    predictor = GhanaDemandPredictor()
    predictor.prepare_demand_features()
    print("   - Demand features prepared")
    print("   - Ready for demand modeling")
    
    # 4. Visualization demo
    print("\n4. 🗺️ VISUALIZATION DEMO")
    print("-" * 30)
    visualizer = GhanaTransportVisualizer()
    visualizer.create_network_overview_map()
    print("   - Interactive map created")
    
    # 5. Key insights
    print("\n5. 💡 KEY INSIGHTS")
    print("-" * 30)
    insights = [
        "• 93 transport agencies operate in Accra",
        "• 653 routes serve 2,566 stops",
        "• 4,797 stop times show detailed schedules",
        "• All routes are bus services (type 3)",
        "• Geographic coverage spans Accra metropolitan area"
    ]
    
    for insight in insights:
        print(f"   {insight}")
        
    # 6. Next steps
    print("\n6. 🎯 NEXT STEPS")
    print("-" * 30)
    next_steps = [
        "Run 'python dataset_explorer.py' for full analysis",
        "Run 'python route_optimizer.py' for AI optimization",
        "Run 'python demand_predictor.py' for demand prediction",
        "Run 'python visualization.py' for interactive charts",
        "Open generated HTML files in browser"
    ]
    
    for step in next_steps:
        print(f"   {step}")
        
    print("\n" + "="*60)
    print("✅ QUICK START COMPLETE!")
    print("="*60)
    print("\nReady to explore the Ghana transport network!")
    print("Check the README.md for detailed instructions.")

def show_dataset_preview():
    """Show a preview of the dataset structure"""
    print("\n📋 DATASET PREVIEW")
    print("="*40)
    
    try:
        # Load a sample of each dataset
        agency = pd.read_csv('dataset/agency.txt')
        routes = pd.read_csv('dataset/routes.txt')
        stops = pd.read_csv('dataset/stops.txt')
        
        print(f"\nAgency Data ({len(agency)} agencies):")
        print(agency.head(3).to_string(index=False))
        
        print(f"\nRoutes Data ({len(routes)} routes):")
        print(routes.head(3).to_string(index=False))
        
        print(f"\nStops Data ({len(stops)} stops):")
        print(stops.head(3).to_string(index=False))
        
    except FileNotFoundError:
        print("❌ Dataset files not found. Make sure you're in the correct directory.")
        print("   Expected files: dataset/agency.txt, dataset/routes.txt, etc.")

if __name__ == "__main__":
    # Show dataset preview
    show_dataset_preview()
    
    # Run quick demo
    quick_demo() 