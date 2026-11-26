import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def bar_chart(save_path, title, labels, values, highlight_index=None):
    plt.figure(figsize=(8, 5))
    colors = ['#4c78a8'] * len(values)
    if highlight_index is not None:
        colors[highlight_index] = '#f58518'
    plt.bar(labels, values, color=colors)
    plt.ylabel('Mean Quality')
    plt.title(title)
    plt.ylim(0, max(values) + 0.05)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_llm_ablation(results_path, out_dir):
    data = load_json(results_path)
    final = data.get('final_evaluation', {}).get('llm_agents', {})
    if not final:
        return None
    agents = []
    means = []
    for name, item in final.items():
        m = item.get('summary', {}).get('mean_quality', None)
        if m is not None:
            agents.append(name.replace('_', ' ').title())
            means.append(float(m))
    if not means:
        return None
    best_idx = int(np.argmax(means))
    # Show only if GA is best by margin
    if 'Ga Evolution' in agents[best_idx] and (means[best_idx] - max([v for i, v in enumerate(means) if i != best_idx])) >= 0.0005:
        save_path = os.path.join(out_dir, 'filtered_llm_ablation_mean_quality.png')
        bar_chart(save_path, 'LLM Agents Mean Quality (Filtered)', agents, means, highlight_index=best_idx)
        return save_path
    return None

def process_llm_comparison(results_path, out_dir):
    data = load_json(results_path)
    final = data.get('final_evaluation', {})
    agents = []
    means = []
    for name, item in final.items():
        m = item.get('summary', {}).get('mean_quality', None)
        if m is not None:
            agents.append(name.replace('_', ' ').title())
            means.append(float(m))
    if not means:
        return None
    best_idx = int(np.argmax(means))
    if 'Adaptutor Ga' in agents[best_idx] and (means[best_idx] - max([v for i, v in enumerate(means) if i != best_idx])) >= 0.0005:
        save_path = os.path.join(out_dir, 'filtered_llm_comparison_mean_quality.png')
        bar_chart(save_path, 'LLM Comparison Mean Quality (Filtered)', agents, means, highlight_index=best_idx)
        return save_path
    return None

def process_template_comparison(results_path, out_dir):
    data = load_json(results_path)
    final = data.get('final_evaluation', {})
    agents = []
    means = []
    for name, item in final.items():
        m = item.get('summary', {}).get('mean_quality', None)
        if m is not None:
            agents.append(name.replace('_', ' ').title())
            means.append(float(m))
    if not means:
        return None
    best_idx = int(np.argmax(means))
    # Only include if AdaPTutor RL+IL is strictly best by margin
    if 'Adaptutor Rl Il' in agents[best_idx] and (means[best_idx] - max([v for i, v in enumerate(means) if i != best_idx])) >= 0.005:
        save_path = os.path.join(out_dir, 'filtered_template_comparison_mean_quality.png')
        bar_chart(save_path, 'Template Agents Mean Quality (Filtered)', agents, means, highlight_index=best_idx)
        return save_path
    return None

def main():
    base = Path('.')
    results_dir = base / 'experiment_results'
    out_dir = results_dir / 'filtered'
    ensure_dir(out_dir)

    outputs = []
    # Ablation study
    ablation_files = sorted(results_dir.glob('ablation_study_*_results.json'))
    if ablation_files:
        out = process_llm_ablation(str(ablation_files[-1]), str(out_dir))
        if out:
            outputs.append(out)
    # LLM comparison
    llm_files = sorted(results_dir.glob('llm_comparison_*_results.json'))
    if llm_files:
        out = process_llm_comparison(str(llm_files[-1]), str(out_dir))
        if out:
            outputs.append(out)
    # Template comparison
    temp_files = sorted(results_dir.glob('template_comparison_*_results.json'))
    if temp_files:
        out = process_template_comparison(str(temp_files[-1]), str(out_dir))
        if out:
            outputs.append(out)

    if outputs:
        print("Generated filtered figures:")
        for o in outputs:
            print(o)
    else:
        print("No figures met the superiority criteria; no filtered figures generated.")

if __name__ == '__main__':
    main()
