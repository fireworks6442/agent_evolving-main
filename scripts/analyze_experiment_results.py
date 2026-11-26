"""
Experiment Results Analysis and Visualization Script
Analyzes and visualizes results from all three AdaPTutor experiments:
1. Template Comparison Experiment
2. LLM Comparison Experiment  
3. Ablation Study Experiment

Generates tables and figures matching the paper's format.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import glob

class ExperimentAnalyzer:
    def __init__(self, results_dir='experiment_results'):
        """Initialize the experiment analyzer"""
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_experiment_results(self):
        """Load results from all experiments"""
        results = {}
        
        # Find all result files
        result_files = glob.glob(os.path.join(self.results_dir, '*_results.json'))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    experiment_type = self._identify_experiment_type(data)
                    if experiment_type:
                        results[experiment_type] = data
                        print(f"Loaded {experiment_type} results from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def _identify_experiment_type(self, data):
        """Identify experiment type from data structure"""
        if 'experiment_id' in data:
            exp_id = data['experiment_id']
            if 'template_comparison' in exp_id:
                return 'template_comparison'
            elif 'llm_comparison' in exp_id:
                return 'llm_comparison'
            elif 'ablation_study' in exp_id:
                return 'ablation_study'
        return None
    
    def analyze_template_comparison(self, data