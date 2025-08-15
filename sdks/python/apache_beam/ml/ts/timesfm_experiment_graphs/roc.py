import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc as sklearn_auc
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
BASE_PATH_DATASETS = Path("/Users/ashokrd/Downloads/TSB-UAD-Public-v2/")
BASE_PATH_JSONL = Path("/Users/ashokrd/Downloads/all_jsonl_files/")
EVALUATION_SPLIT = 0.8  # Use the final 20% of data for evaluation

def calculate_performance_metrics(dataset_path, jsonl_path):
    """
    Calculates ROC AUC and PR AUC scores on the final 20% of the data.
    Includes a fix for handling 'infinity' scores.
    """
    try:
        df = pd.read_csv(dataset_path)
        if 'Label' not in df.columns: return None
        labels = df['Label'].to_numpy()

        all_anomalies = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_anomalies.extend(data.get('anomalies', []))

        # Sanitize the outlier scores to handle infinity values
        finite_scores = [a['outlier_score'] for a in all_anomalies if np.isfinite(a['outlier_score'])]
        max_finite_score = max(finite_scores) if finite_scores else 1e12

        prediction_scores = np.zeros_like(labels, dtype=float)
        for anomaly in all_anomalies:
            score = anomaly['outlier_score']
            if not np.isfinite(score):
                score = max_finite_score * 1.1 
            
            idx = int(anomaly['timestamp'])
            if idx < len(prediction_scores):
                prediction_scores[idx] = score
        
        # Slicing logic for the final 20%
        split_point = int(len(labels) * EVALUATION_SPLIT)
        y_true_eval = labels[split_point:]
        y_score_eval = prediction_scores[split_point:]

        if len(y_true_eval) == 0: return None
        if len(np.unique(y_true_eval)) < 2: return None

        roc_auc = roc_auc_score(y_true_eval, y_score_eval)
        fpr, tpr, _ = roc_curve(y_true_eval, y_score_eval)

        precision, recall, _ = precision_recall_curve(y_true_eval, y_score_eval)
        pr_auc = sklearn_auc(recall, precision)
        
        return {
            "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr,
            "pr_auc": pr_auc, "precision": precision, "recall": recall
        }
    except Exception as e:
        print(f"  [Error] Failed processing {dataset_path.name}: {e}")
        return None

def print_overall_table(title, results_list, metric='roc_auc'):
    """Prints the overall top/bottom performers."""
    metric_name = "ROC AUC" if metric == 'roc_auc' else "PR AUC"
    print(f"\n--- {title} ({metric_name}) ---")
    print(f"{'Rank':<5} {metric_name+' Score':<15} {'Model':<12} {'Dataset':<50}")
    print(f"{'-'*5} {'-'*15} {'-'*12} {'-'*50}")
    for result in results_list:
        rank = result.get('rank', '#')
        score = f"{result[metric]:.4f}"
        model = result['model_type'].capitalize()
        dataset = result['dataset']
        print(f"{rank:<5} {score:<15} {model:<12} {dataset:<50}")

def print_improvement_table(title, improvement_list, metric='roc_auc'):
    """Prints the top fine-tuning improvements."""
    metric_name = "ROC AUC" if metric == 'roc_auc' else "PR AUC"
    improvement_key = 'roc_improvement' if metric == 'roc_auc' else 'pr_improvement'
    finetuned_key = 'finetuned_roc_auc' if metric == 'roc_auc' else 'finetuned_pr_auc'
    original_key = 'original_roc_auc' if metric == 'roc_auc' else 'original_pr_auc'
    
    print(f"\n--- {title} ({metric_name}) ---")
    print(f"{'Rank':<5} {'Improvement (Δ)':<20} {f'Finetuned {metric_name}':<20} {f'Original {metric_name}':<20} {'Dataset':<50}")
    print(f"{'-'*5} {'-'*20} {'-'*20} {'-'*20} {'-'*50}")
    for i, result in enumerate(improvement_list, 1):
        improvement = f"+{result[improvement_key]:.4f}"
        f_score = f"{result[finetuned_key]:.4f}"
        o_score = f"{result[original_key]:.4f}"
        dataset = result['dataset']
        print(f"{i:<5} {improvement:<20} {f_score:<20} {o_score:<20} {dataset:<50}")

def print_bottom_improvement_table(title, improvement_list, metric='roc_auc'):
    """Prints the datasets where fine-tuning was least effective."""
    metric_name = "ROC AUC" if metric == 'roc_auc' else "PR AUC"
    improvement_key = 'roc_improvement' if metric == 'roc_auc' else 'pr_improvement'
    finetuned_key = 'finetuned_roc_auc' if metric == 'roc_auc' else 'finetuned_pr_auc'
    original_key = 'original_roc_auc' if metric == 'roc_auc' else 'original_pr_auc'

    print(f"\n--- {title} ({metric_name}) ---")
    print(f"{'Rank':<5} {'Improvement (Δ)':<20} {f'Finetuned {metric_name}':<20} {f'Original {metric_name}':<20} {'Dataset':<50}")
    print(f"{'-'*5} {'-'*20} {'-'*20} {'-'*20} {'-'*50}")
    for i, result in enumerate(reversed(improvement_list), 1):
        improvement = f"{result[improvement_key]:+.4f}"
        f_score = f"{result[finetuned_key]:.4f}"
        o_score = f"{result[original_key]:.4f}"
        dataset = result['dataset']
        print(f"{i:<5} {improvement:<20} {f_score:<20} {o_score:<20} {dataset:<50}")

def plot_aggregate_boxplot_and_histogram(all_results, metric='roc_auc', num_bins=20):
    """
    Creates a figure with a box plot and a histogram comparing the distribution 
    of original vs finetuned scores for a given metric.
    
    Args:
        all_results (list): List of result dictionaries.
        metric (str): The metric to plot ('roc_auc' or 'pr_auc').
        num_bins (int): The number of bins to use for the histogram.
    """
    metric_name = "ROC AUC" if metric == 'roc_auc' else "PR AUC"
    original_scores = [r[metric] for r in all_results if r['model_type'] == 'original']
    finetuned_scores = [r[metric] for r in all_results if r['model_type'] == 'finetuned']

    if not original_scores or not finetuned_scores:
        print("\nCould not generate aggregate plots: missing data for one or both model types.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Box Plot (Left Subplot) ---
    boxplot_parts = axes[0].boxplot([original_scores, finetuned_scores],
                                    labels=['Original Models', 'Finetuned Models'],
                                    patch_artist=True,
                                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                                    medianprops=dict(color='red', linewidth=2.5))
    axes[0].set_title(f'Distribution of {metric_name} Scores by Model Type', fontsize=16, fontweight='bold')
    axes[0].set_ylabel(f'{metric_name} Score', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='y', labelsize=12)
    # Make x-tick labels bold
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontweight='bold', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Histogram (Right Subplot) ---
    all_scores = original_scores + finetuned_scores
    min_range = np.floor(min(all_scores) * 10) / 10
    max_range = np.ceil(max(all_scores) * 10) / 10
    bins = np.linspace(min_range, max_range, num_bins + 1)

    axes[1].hist(original_scores, bins=bins, alpha=0.7, label='Original', color='blue', edgecolor='black')
    axes[1].hist(finetuned_scores, bins=bins, alpha=0.7, label='Finetuned', color='red', edgecolor='black')
    
    axes[1].set_xticks(bins)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # Make x-tick labels bold
    for label in axes[1].get_xticklabels():
        label.set_fontweight('bold')

    axes[1].set_title(f'Frequency of {metric_name} Scores', fontsize=16, fontweight='bold')
    axes[1].set_xlabel(f'{metric_name} Score', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xlim(min_range, max_range)

    fig.suptitle(f'Comparison of Model Performance ({metric_name}): Original vs. Finetuned', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    filename = f"aggregate_{metric}_performance.png"
    plt.savefig(filename, dpi=300)
    print(f"\nGenerated aggregate performance plot (boxplot and histogram) as '{filename}'")

def main():
    """Main function to run all analyses and generate plots."""
    print("🚀 Starting analysis...")
    with open('new_datasets.txt', 'r') as f:
        all_dataset_paths = [Path(line.strip()) for line in f if line.strip()]
    print(f"Found {len(all_dataset_paths)} potential datasets to analyze.\n")

    all_results, improvement_results = [], []

    for dataset_path in all_dataset_paths:
        base_name = dataset_path.stem
        # print(f"Processing Dataset: {base_name}")
        original_jsonl_path = BASE_PATH_JSONL / f"plot_data_{base_name}_original.jsonl"
        finetuned_jsonl_path = BASE_PATH_JSONL / f"plot_data_{base_name}_finetuned.jsonl"
        
        original_result, finetuned_result = None, None

        if original_jsonl_path.exists():
            original_result = calculate_performance_metrics(dataset_path, original_jsonl_path)
            if original_result:
                all_results.append({
                    "roc_auc": original_result["roc_auc"], 
                    "pr_auc": original_result["pr_auc"],
                    "dataset": base_name, 
                    "model_type": "original"
                })

        if finetuned_jsonl_path.exists():
            finetuned_result = calculate_performance_metrics(dataset_path, finetuned_jsonl_path)
            if finetuned_result:
                all_results.append({
                    "roc_auc": finetuned_result["roc_auc"],
                    "pr_auc": finetuned_result["pr_auc"],
                    "dataset": base_name, 
                    "model_type": "finetuned"
                })
        
        if original_result and finetuned_result:
            roc_improvement = finetuned_result['roc_auc'] - original_result['roc_auc']
            pr_improvement = finetuned_result['pr_auc'] - original_result['pr_auc']
            improvement_results.append({
                'dataset': base_name, 
                'roc_improvement': roc_improvement, 
                'pr_improvement': pr_improvement,
                'original_roc_auc': original_result['roc_auc'], 
                'finetuned_roc_auc': finetuned_result['roc_auc'],
                'original_pr_auc': original_result['pr_auc'],
                'finetuned_pr_auc': finetuned_result['pr_auc']
            })

    print("\n" + "="*80)
    print("🏆 Analysis Complete: Final Report 🏆".center(80))
    print("="*80)

    if not all_results:
        print("No valid results were generated. Please check file paths and data content.")
        return

    # --- Generate Tables for ROC AUC ---
    print(len(all_results), "valid results found.")
    sorted_roc_results = sorted(all_results, key=lambda x: x['roc_auc'], reverse=True)
    for i, result in enumerate(sorted_roc_results, 1):
        result['rank'] = i
    print_overall_table("Top 10 Overall Performers", sorted_roc_results[:10], metric='roc_auc')
    
    if improvement_results:
        sorted_roc_improvements = sorted(improvement_results, key=lambda x: x['roc_improvement'], reverse=True)
        print_improvement_table("Top 10 Improvements from Fine-Tuning", sorted_roc_improvements[:10], metric='roc_auc')
        print_bottom_improvement_table("Bottom 10 Improvements from Fine-Tuning", sorted_roc_improvements[-10:], metric='roc_auc')

    # --- Generate Tables for PR AUC ---
    sorted_pr_results = sorted(all_results, key=lambda x: x['pr_auc'], reverse=True)
    for i, result in enumerate(sorted_pr_results, 1):
        result['rank'] = i
    print_overall_table("Top 10 Overall Performers", sorted_pr_results[:10], metric='pr_auc')

    if improvement_results:
        sorted_pr_improvements = sorted(improvement_results, key=lambda x: x['pr_improvement'], reverse=True)
        print_improvement_table("Top 10 Improvements from Fine-Tuning", sorted_pr_improvements[:10], metric='pr_auc')
        print_bottom_improvement_table("Bottom 10 Improvements from Fine-Tuning", sorted_pr_improvements[-10:], metric='pr_auc')
    
    # --- Generate Plots ---
    # Plot 1: ROC and PR Curves for the single best model (by ROC AUC)
    best_overall_result = sorted_roc_results[0]
    print(f"\nGenerating ROC and PR plots for the top overall performer: {best_overall_result['dataset']}...")
    best_dataset_path = next(BASE_PATH_DATASETS.rglob(f"**/{best_overall_result['dataset']}.csv"))
    best_jsonl_path = BASE_PATH_JSONL / f"plot_data_{best_overall_result['dataset']}_{best_overall_result['model_type']}.jsonl"
    performance_data_for_plot = calculate_performance_metrics(best_dataset_path, best_jsonl_path)
    
    if performance_data_for_plot:
        # ROC Curve
        plt.figure(figsize=(8, 8))
        plt.plot(performance_data_for_plot['fpr'], performance_data_for_plot['tpr'], color='darkorange', lw=2, label=f"ROC curve (AUC = {performance_data_for_plot['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f"ROC Curve for Best Model: {best_overall_result['dataset']} ({best_overall_result['model_type'].capitalize()})", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f"best_model_roc_curve.png", dpi=300)
        plt.close()
        print(f"  Saved best model ROC curve as 'best_model_roc_curve.png'")

        # PR Curve
        plt.figure(figsize=(8, 8))
        plt.plot(performance_data_for_plot['recall'], performance_data_for_plot['precision'], color='blue', lw=2, label=f"PR curve (AUC = {performance_data_for_plot['pr_auc']:.4f})")
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f"Precision-Recall Curve for Best Model: {best_overall_result['dataset']} ({best_overall_result['model_type'].capitalize()})", fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(f"best_model_pr_curve.png", dpi=300)
        plt.close()
        print(f"  Saved best model PR curve as 'best_model_pr_curve.png'")

    # Plot 2: Aggregate Box Plot and Histogram of all results
    print("\nGenerating aggregate performance plots...")
    plot_aggregate_boxplot_and_histogram(all_results, metric='roc_auc')
    plot_aggregate_boxplot_and_histogram(all_results, metric='pr_auc')

    # Plot 3: Epoch Comparison Bar Chart
    print("\nGenerating epoch comparison plots...")
    plot_epoch_comparison()

    print("\nGenerating horizon comparison plots...")
    plot_horizon_comparison()


def plot_epoch_comparison():
    """
    Calculates and plots bar charts comparing ROC AUC and PR AUC scores for different epochs.
    """
    dataset_path = Path("/Users/ashokrd/Downloads/TSB-UAD-Public-v2/WSD/WSD_94.csv")
    epoch_files = {
        "Original" : "/Users/ashokrd/Downloads/all_jsonl_files/plot_data_WSD_94_original.jsonl",
        "Epoch 1": "/Users/ashokrd/Downloads/all_jsonl_files/plot_data_WSD_94_finetuned.jsonl",
        "Epoch 2": "WSD_94_hyperparameter/plot_data_epoch_2_finetuned.jsonl",
        "Epoch 4": "WSD_94_hyperparameter/plot_data_epoch_4_finetuned.jsonl",
        "Epoch 8": "WSD_94_hyperparameter/plot_data_epoch_8_finetuned.jsonl",
    }
    epoch_noise_files = {
        "Original" : "/Users/ashokrd/Downloads/all_jsonl_files/plot_data_WSD_94_original.jsonl",
        "Epoch 1": "/Users/ashokrd/Developer/timesfm_experiment_graphs/finetune_with_noise/plot_data_epoch_1_finetuned.jsonl",
        "Epoch 2": "/Users/ashokrd/Developer/timesfm_experiment_graphs/finetune_with_noise/plot_data_epoch_2_finetuned.jsonl",
        "Epoch 4": "/Users/ashokrd/Developer/timesfm_experiment_graphs/finetune_with_noise/plot_data_epoch_4_finetuned.jsonl",
        "Epoch 8": "/Users/ashokrd/Developer/timesfm_experiment_graphs/finetune_with_noise/plot_data_epoch_8_finetuned.jsonl",
    }

    epoch_metrics = {}
    for epoch_label, file_path in epoch_files.items():
        jsonl_path = Path(file_path)
        if jsonl_path.exists():
            result = calculate_performance_metrics(dataset_path, jsonl_path)
            if result:
                epoch_metrics[epoch_label] = {'roc_auc': result['roc_auc'], 'pr_auc': result['pr_auc']}
        else:
            print(f"  [Warning] File not found: {file_path}")

    epoch_noise_metrics = {}
    for epoch_label, file_path in epoch_noise_files.items():
        jsonl_path = Path(file_path)
        if jsonl_path.exists():
            result = calculate_performance_metrics(dataset_path, jsonl_path)
            if result:
                epoch_noise_metrics[epoch_label] = {'roc_auc': result['roc_auc'], 'pr_auc': result['pr_auc']}
        else:
            print(f"  [Warning] File not found: {file_path}")

    if not epoch_metrics:
        print("Could not generate epoch comparison plot: no valid AUC scores found.")
        return

    labels = list(epoch_metrics.keys())
    roc_scores = [epoch_metrics[label]['roc_auc'] for label in labels]
    pr_scores = [epoch_metrics[label]['pr_auc'] for label in labels]

    noise_labels = list(epoch_noise_metrics.keys())
    noise_roc_scores = [epoch_noise_metrics[label]['roc_auc'] for label in noise_labels]

    # Plot for ROC AUC (Bar Chart)
    plt.figure(figsize=(10, 6))
    bars_roc = plt.bar(labels, roc_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    plt.title('ROC AUC Score vs. Training Epoch', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_roc:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("epoch_comparison_roc.png", dpi=300)
    plt.close()
    print("Generated epoch comparison plot for ROC AUC as 'epoch_comparison_roc.png'")

    # Plot for ROC AUC (Line Chart Overlay)
    plt.figure(figsize=(10, 6))
    plt.plot(labels, roc_scores, marker='o', linestyle='-', label='Original Epochs')
    plt.plot(noise_labels, noise_roc_scores, marker='x', linestyle='--', label='Noise Epochs')
    plt.ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    plt.title('ROC AUC Score vs. Training Epoch (with Noise)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    for i, label in enumerate(labels):
        plt.text(label, roc_scores[i] + 0.02, f'{roc_scores[i]:.4f}', ha='center', va='bottom')
    for i, label in enumerate(noise_labels):
        plt.text(label, noise_roc_scores[i] - 0.04, f'{noise_roc_scores[i]:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("epoch_comparison_roc_noise.png", dpi=300)
    plt.close()
    print("Generated epoch comparison plot for ROC AUC with noise as 'epoch_comparison_roc_noise.png'")

    # Plot for PR AUC
    plt.figure(figsize=(10, 6))
    bars_pr = plt.bar(labels, pr_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('PR AUC Score', fontsize=12, fontweight='bold')
    plt.title('PR AUC Score vs. Training Epoch', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_pr:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("epoch_comparison_pr.png", dpi=300)
    plt.close()
    print("Generated epoch comparison plot for PR AUC as 'epoch_comparison_pr.png'")



def plot_horizon_comparison():
    """
    Calculates and plots bar charts comparing ROC AUC and PR AUC scores for different horizons.
    """
    dataset_path = Path("/Users/ashokrd/Downloads/TSB-UAD-Public-v2/WSD/WSD_94.csv")
    horizon_files = {
        "16" : "WSD_94_hyperparameter/plot_data_horizon_16_finetuned.jsonl",
        "32": "WSD_94_hyperparameter/plot_data_horizon_32_finetuned.jsonl",
        "64": "WSD_94_hyperparameter/plot_data_horizon_64_finetuned.jsonl",
        "96": "WSD_94_hyperparameter/plot_data_horizon_96_finetuned.jsonl",
        "128": "/Users/ashokrd/Downloads/all_jsonl_files/plot_data_WSD_94_finetuned.jsonl",
    }

    horizon_metrics = {}
    for horizon_label, file_path in horizon_files.items():
        jsonl_path = Path(file_path)
        if jsonl_path.exists():
            result = calculate_performance_metrics(dataset_path, jsonl_path)
            if result:
                horizon_metrics[horizon_label] = {'roc_auc': result['roc_auc'], 'pr_auc': result['pr_auc']}
        else:
            print(f"  [Warning] File not found: {file_path}")

    if not horizon_metrics:
        print("Could not generate horizon comparison plot: no valid AUC scores found.")
        return

    labels = list(horizon_metrics.keys())
    roc_scores = [horizon_metrics[label]['roc_auc'] for label in labels]
    pr_scores = [horizon_metrics[label]['pr_auc'] for label in labels]

    # Plot for ROC AUC
    plt.figure(figsize=(10, 6))
    bars_roc = plt.bar(labels, roc_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    plt.title('ROC AUC Score vs. Training Horizon', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_roc:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("horizon_comparison_roc.png", dpi=300)
    plt.close()
    print("Generated horizon comparison plot for ROC AUC as 'horizon_comparison_roc.png'")

    # Plot for PR AUC
    plt.figure(figsize=(10, 6))
    bars_pr = plt.bar(labels, pr_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('PR AUC Score', fontsize=12, fontweight='bold')
    plt.title('PR AUC Score vs. Training Horizon', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_pr:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("horizon_comparison_pr.png", dpi=300)
    plt.close()
    print("Generated horizon comparison plot for PR AUC as 'horizon_comparison_pr.png'")


if __name__ == "__main__":
    main()
