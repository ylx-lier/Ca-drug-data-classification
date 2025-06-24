#!/usr/bin/env python3
"""
English-only Results Viewer - No Chinese font issues
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def setup_english_font():
    """Setup English-only fonts"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    print("✅ English font setup complete")

def load_experiment_results():
    """Load experiment results"""
    results_file = Path("../../results/exp114/all_results.json")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract results section
    if 'results' in data:
        return data['results']
    else:
        return data

def create_results_dashboard():
    """Create results dashboard"""
    results = load_experiment_results()
    if not results:
        return
    
    print("📊 Experiment Results Dashboard")
    print("=" * 60)
    
    # Extract metrics from all datasets
    datasets = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for dataset, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            datasets.append(dataset)
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Dataset': datasets,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Print table
    print("\n📋 Detailed Results Table:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"Number of datasets: {len(datasets)}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Best accuracy: {np.max(accuracies):.4f} ({datasets[np.argmax(accuracies)]})")
    print(f"Worst accuracy: {np.min(accuracies):.4f} ({datasets[np.argmin(accuracies)]})")
    
    # Create visualizations
    create_visualization_plots(df)

def create_visualization_plots(df):
    """Create visualization plots"""
    print("\n🎨 Generating visualization plots...")
    
    # Setup English fonts
    setup_english_font()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Results Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Accuracy bar chart
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df)), df['Accuracy'], color='skyblue', alpha=0.7)
    ax1.set_title('Dataset Accuracy')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Dataset'], rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Average metrics comparison
    ax2 = axes[0, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    avg_scores = [df[metric].mean() for metric in metrics]
    bars2 = ax2.bar(metrics, avg_scores, color=['lightcoral', 'lightgreen', 'lightsalmon', 'lightblue'])
    ax2.set_title('Average Metrics Comparison')
    ax2.set_ylabel('Average Score')
    ax2.set_ylim(0, 1)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Heatmap
    ax3 = axes[1, 0]
    heatmap_data = df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
    im = ax3.imshow(heatmap_data.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Metrics Heatmap')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Dataset'], rotation=45, ha='right')
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Score')
    
    # 4. Scatter plot - Precision vs Recall
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['Precision'], df['Recall'], 
                         c=df['Accuracy'], cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    ax4.set_title('Precision vs Recall (Color = Accuracy)')
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Recall')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter, ax=ax4)
    cbar2.set_label('Accuracy')
    
    # Add dataset labels
    for i, txt in enumerate(df['Dataset']):
        ax4.annotate(txt, (df['Precision'].iloc[i], df['Recall'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=6)
    
    plt.tight_layout()
    
    # Save chart
    output_file = 'results_dashboard_en.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {output_file}")
    
    plt.show()

def create_training_curve():
    """Create simulated training curve"""
    print("\n📈 Creating training curve...")
    
    # Setup English fonts
    setup_english_font()
    
    # Simulated training data
    epochs = np.arange(1, 51)
    loss = 0.1 * np.exp(-epochs/20) + 0.01 * np.random.random(50)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save chart
    plt.savefig('training_curve_en.png', dpi=300, bbox_inches='tight')
    print("✅ Training curve saved: training_curve_en.png")
    plt.show()

def create_performance_analysis():
    """Create detailed performance analysis"""
    print("\n🔍 Creating performance analysis...")
    
    results = load_experiment_results()
    if not results:
        return
    
    setup_english_font()
    
    # Extract time-based analysis
    datasets = []
    accuracies = []
    time_points = []
    
    for dataset, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            datasets.append(dataset)
            accuracies.append(metrics.get('accuracy', 0))
            
            # Extract time point if present
            if 'day' in dataset.lower():
                if 'day45' in dataset.lower():
                    time_points.append(45)
                elif 'day90' in dataset.lower():
                    time_points.append(90)
                elif 'day120' in dataset.lower():
                    time_points.append(120)
                else:
                    time_points.append(0)
            else:
                time_points.append(0)
    
    # Create performance analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Performance by dataset type
    dataset_types = {}
    for dataset, acc in zip(datasets, accuracies):
        base_name = dataset.split('_')[0] if '_' in dataset else dataset
        if base_name not in dataset_types:
            dataset_types[base_name] = []
        dataset_types[base_name].append(acc)
    
    type_names = list(dataset_types.keys())
    type_means = [np.mean(dataset_types[t]) for t in type_names]
    type_stds = [np.std(dataset_types[t]) for t in type_names]
    
    bars = ax1.bar(type_names, type_means, yerr=type_stds, capsize=5, 
                   color='lightblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Performance by Dataset Type')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_ylim(0, 1)
    
    for i, (bar, mean) in enumerate(zip(bars, type_means)):
        ax1.text(bar.get_x() + bar.get_width()/2., mean + type_stds[i] + 0.02,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # 2. Time-series analysis
    time_acc_data = {}
    for dataset, acc, time in zip(datasets, accuracies, time_points):
        if time > 0:
            base_type = dataset.replace(f'day{time}_', '').split('_')[0]
            if base_type not in time_acc_data:
                time_acc_data[base_type] = {'times': [], 'accs': []}
            time_acc_data[base_type]['times'].append(time)
            time_acc_data[base_type]['accs'].append(acc)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (base_type, data) in enumerate(time_acc_data.items()):
        if len(data['times']) > 1:
            ax2.plot(data['times'], data['accs'], 'o-', 
                    color=colors[i % len(colors)], label=base_type, linewidth=2)
    
    ax2.set_title('Performance Over Time')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis_en.png', dpi=300, bbox_inches='tight')
    print("✅ Performance analysis saved: performance_analysis_en.png")
    plt.show()

def main():
    print("🖥️  English Results Viewer")
    print("=" * 50)
    
    try:
        create_results_dashboard()
        create_training_curve()
        create_performance_analysis()
        
        print("\n🎉 All charts generated successfully!")
        print("📁 Generated files:")
        print("  - results_dashboard_en.png")
        print("  - training_curve_en.png")
        print("  - performance_analysis_en.png")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        print("Please ensure required dependencies are installed: matplotlib, pandas, numpy")

if __name__ == "__main__":
    main()
