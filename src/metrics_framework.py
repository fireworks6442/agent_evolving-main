"""
Comprehensive Metrics Framework
Tracks and measures agent evolution and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MetricsFramework:
    def __init__(self):
        """Initialize metrics framework with enhanced tracking"""
        self.metrics_history = []
        self.evolution_metrics = []
        
        # Performance baselines for tracking improvement
        self.performance_baselines = {
            'initial_quality_score': None,
            'initial_classification_accuracy': None,
            'initial_reward': None,
            'target_quality_score': 0.70,  # Target 70% quality
            'target_classification_accuracy': 0.80,  # Target 80% accuracy
            'baseline_established': False
        }
        
        # Improvement tracking
        self.improvement_history = {
            'quality_improvements': [],
            'accuracy_improvements': [],
            'reward_improvements': [],
            'consecutive_improvements': 0,
            'best_quality_score': 0.0,
            'best_accuracy': 0.0,
            'best_reward': 0.0,
            'improvement_streaks': []
        }
        
        # Performance memory for trend analysis
        self.performance_memory = {
            'recent_quality_scores': [],
            'recent_accuracies': [],
            'recent_rewards': [],
            'window_size': 10
        }
        self.evolution_tracking = {
            'classification_performance': [],
            'response_quality': [],
            'learning_effectiveness': [],
            'system_health': [],
            'user_satisfaction': []
        }
        
    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate classification performance metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive and negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                metrics['auc_score'] = auc
            except:
                metrics['auc_score'] = None
        
        return metrics
    
    def calculate_response_quality_metrics(self, response_data_list):
        """Calculate response quality metrics"""
        if not response_data_list:
            return {}
        
        quality_scores = [r['quality_score'] for r in response_data_list]
        response_times = [r['response_time'] for r in response_data_list]
        
        # Quality metrics
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        # Response time metrics
        avg_response_time = np.mean(response_times)
        response_time_std = np.std(response_times)
        
        # Urgency-specific metrics
        high_urgency_responses = [r for r in response_data_list if r['urgency_level'] == 1]
        low_urgency_responses = [r for r in response_data_list if r['urgency_level'] == 0]
        
        high_urgency_quality = np.mean([r['quality_score'] for r in high_urgency_responses]) if high_urgency_responses else 0
        low_urgency_quality = np.mean([r['quality_score'] for r in low_urgency_responses]) if low_urgency_responses else 0
        
        high_urgency_time = np.mean([r['response_time'] for r in high_urgency_responses]) if high_urgency_responses else 0
        low_urgency_time = np.mean([r['response_time'] for r in low_urgency_responses]) if low_urgency_responses else 0
        
        metrics = {
            'avg_quality_score': avg_quality,
            'quality_std': quality_std,
            'min_quality_score': min_quality,
            'max_quality_score': max_quality,
            'avg_response_time': avg_response_time,
            'response_time_std': response_time_std,
            'high_urgency_quality': high_urgency_quality,
            'low_urgency_quality': low_urgency_quality,
            'high_urgency_response_time': high_urgency_time,
            'low_urgency_response_time': low_urgency_time,
            'total_responses': len(response_data_list),
            'high_urgency_count': len(high_urgency_responses),
            'low_urgency_count': len(low_urgency_responses)
        }
        
        return metrics
    
    def calculate_evolution_metrics(self, historical_data, window_size=10):
        """Calculate evolution and learning effectiveness metrics"""
        if len(historical_data) < 2:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(historical_data)
        
        # Calculate trends
        metrics = {}
        
        # Performance improvement over time
        if 'accuracy' in df.columns:
            recent_accuracy = df['accuracy'].tail(window_size).mean()
            early_accuracy = df['accuracy'].head(window_size).mean()
            accuracy_improvement = recent_accuracy - early_accuracy
            metrics['accuracy_improvement'] = accuracy_improvement
            metrics['accuracy_trend'] = self._calculate_trend(df['accuracy'].values)
        
        if 'avg_quality_score' in df.columns:
            recent_quality = df['avg_quality_score'].tail(window_size).mean()
            early_quality = df['avg_quality_score'].head(window_size).mean()
            quality_improvement = recent_quality - early_quality
            metrics['quality_improvement'] = quality_improvement
            metrics['quality_trend'] = self._calculate_trend(df['avg_quality_score'].values)
        
        # Learning rate (rate of improvement)
        if len(df) >= 5:
            # Calculate learning rate as slope of performance over time
            x = np.arange(len(df))
            if 'accuracy' in df.columns:
                accuracy_slope = np.polyfit(x, df['accuracy'].values, 1)[0]
                metrics['accuracy_learning_rate'] = accuracy_slope
            
            if 'avg_quality_score' in df.columns:
                quality_slope = np.polyfit(x, df['avg_quality_score'].values, 1)[0]
                metrics['quality_learning_rate'] = quality_slope
        
        # Stability metrics (variance in recent performance)
        if len(df) >= window_size:
            recent_data = df.tail(window_size)
            if 'accuracy' in recent_data.columns:
                metrics['accuracy_stability'] = 1 / (1 + recent_data['accuracy'].std())
            if 'avg_quality_score' in recent_data.columns:
                metrics['quality_stability'] = 1 / (1 + recent_data['avg_quality_score'].std())
        
        # Adaptation speed (how quickly performance improves after changes)
        metrics['adaptation_speed'] = self._calculate_adaptation_speed(df)
        
        return metrics
    
    def _calculate_trend(self, values):
        """Calculate trend direction: 1 for improving, -1 for declining, 0 for stable"""
        if len(values) < 3:
            return 0
        
        # Use linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 1  # Improving
        elif slope < -0.001:
            return -1  # Declining
        else:
            return 0  # Stable
    
    def _calculate_adaptation_speed(self, df):
        """Calculate how quickly the agent adapts to changes"""
        if len(df) < 5:
            return 0
        
        # Look for performance changes after significant drops
        adaptation_speeds = []
        
        if 'accuracy' in df.columns:
            accuracy_values = df['accuracy'].values
            for i in range(1, len(accuracy_values) - 2):
                # Check for significant drop
                if accuracy_values[i] < accuracy_values[i-1] - 0.05:
                    # Measure recovery time
                    recovery_time = 0
                    for j in range(i+1, min(i+5, len(accuracy_values))):
                        if accuracy_values[j] >= accuracy_values[i-1]:
                            recovery_time = j - i
                            break
                    
                    if recovery_time > 0:
                        adaptation_speeds.append(1 / recovery_time)
        
        return np.mean(adaptation_speeds) if adaptation_speeds else 0
    
    def calculate_system_health_metrics(self, response_data_list, error_log=None):
        """Calculate system health and efficiency metrics"""
        if not response_data_list:
            return {}
        
        # Processing throughput
        total_responses = len(response_data_list)
        total_time = sum(r['response_time'] for r in response_data_list)
        throughput = total_responses / max(total_time, 0.001)  # responses per second
        
        # Error rate
        error_count = len(error_log) if error_log else 0
        error_rate = error_count / max(total_responses, 1)
        
        # Resource utilization (simplified)
        avg_response_time = np.mean([r['response_time'] for r in response_data_list])
        max_response_time = np.max([r['response_time'] for r in response_data_list])
        
        # Efficiency score (quality per unit time)
        avg_quality = np.mean([r['quality_score'] for r in response_data_list])
        efficiency_score = avg_quality / max(avg_response_time, 0.001)
        
        metrics = {
            'throughput': throughput,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'efficiency_score': efficiency_score,
            'total_processed': total_responses,
            'total_errors': error_count
        }
        
        return metrics
    
    def calculate_user_satisfaction_metrics(self, feedback_data=None, response_data_list=None):
        """Calculate user satisfaction metrics (simulated for now)"""
        metrics = {}
        
        if response_data_list:
            # Simulated satisfaction based on response quality and urgency handling
            quality_scores = [r['quality_score'] for r in response_data_list]
            
            # High urgency responses should be faster and higher quality
            high_urgency_responses = [r for r in response_data_list if r['urgency_level'] == 1]
            
            if high_urgency_responses:
                high_urgency_quality = np.mean([r['quality_score'] for r in high_urgency_responses])
                high_urgency_speed = np.mean([r['response_time'] for r in high_urgency_responses])
                
                # Satisfaction score based on quality and speed for urgent requests
                urgency_satisfaction = high_urgency_quality * (1 / max(high_urgency_speed, 0.1))
                metrics['urgency_handling_satisfaction'] = min(urgency_satisfaction, 1.0)
            
            # Overall satisfaction based on quality consistency
            quality_consistency = 1 - np.std(quality_scores)
            metrics['quality_consistency_satisfaction'] = max(quality_consistency, 0)
            
            # Response appropriateness (simulated)
            metrics['response_appropriateness'] = np.mean(quality_scores)
        
        # If actual feedback data is provided
        if feedback_data:
            metrics.update(feedback_data)
        
        return metrics
    
    def record_metrics_snapshot(self, classifier_metrics=None, response_metrics=None, 
                              system_metrics=None, satisfaction_metrics=None, timestamp=None):
        """Record a complete metrics snapshot"""
        if timestamp is None:
            timestamp = datetime.now()
        
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'classification_metrics': classifier_metrics or {},
            'response_quality_metrics': response_metrics or {},
            'system_health_metrics': system_metrics or {},
            'user_satisfaction_metrics': satisfaction_metrics or {},
        }
        
        # Calculate evolution metrics if we have historical data
        if len(self.metrics_history) > 0:
            evolution_metrics = self.calculate_evolution_metrics(self.metrics_history)
            snapshot['evolution_metrics'] = evolution_metrics
        
        self.metrics_history.append(snapshot)
        
        # Update evolution tracking
        self._update_evolution_tracking(snapshot)
        
        return snapshot
    
    def _update_evolution_tracking(self, snapshot):
        """Update evolution tracking with new snapshot"""
        # Extract key metrics for tracking
        if 'classification_metrics' in snapshot and snapshot['classification_metrics']:
            self.evolution_tracking['classification_performance'].append({
                'timestamp': snapshot['timestamp'],
                'accuracy': snapshot['classification_metrics'].get('accuracy', 0),
                'f1_score': snapshot['classification_metrics'].get('f1_score', 0)
            })
        
        if 'response_quality_metrics' in snapshot and snapshot['response_quality_metrics']:
            self.evolution_tracking['response_quality'].append({
                'timestamp': snapshot['timestamp'],
                'avg_quality': snapshot['response_quality_metrics'].get('avg_quality_score', 0),
                'avg_response_time': snapshot['response_quality_metrics'].get('avg_response_time', 0)
            })
        
        if 'evolution_metrics' in snapshot and snapshot['evolution_metrics']:
            self.evolution_tracking['learning_effectiveness'].append({
                'timestamp': snapshot['timestamp'],
                'accuracy_improvement': snapshot['evolution_metrics'].get('accuracy_improvement', 0),
                'quality_improvement': snapshot['evolution_metrics'].get('quality_improvement', 0)
            })
        
        if 'system_health_metrics' in snapshot and snapshot['system_health_metrics']:
            self.evolution_tracking['system_health'].append({
                'timestamp': snapshot['timestamp'],
                'throughput': snapshot['system_health_metrics'].get('throughput', 0),
                'error_rate': snapshot['system_health_metrics'].get('error_rate', 0)
            })
        
        if 'user_satisfaction_metrics' in snapshot and snapshot['user_satisfaction_metrics']:
            self.evolution_tracking['user_satisfaction'].append({
                'timestamp': snapshot['timestamp'],
                'overall_satisfaction': np.mean(list(snapshot['user_satisfaction_metrics'].values()))
            })
    
    def generate_comprehensive_report(self, save_plots=True):
        """Generate a comprehensive metrics report"""
        if not self.metrics_history:
            return "No metrics data available"
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_snapshots': len(self.metrics_history),
            'time_period': {
                'start': self.metrics_history[0]['timestamp'],
                'end': self.metrics_history[-1]['timestamp']
            }
        }
        
        # Latest metrics summary
        latest_snapshot = self.metrics_history[-1]
        report['latest_metrics'] = latest_snapshot
        
        # Evolution summary
        if len(self.metrics_history) > 1:
            first_snapshot = self.metrics_history[0]
            
            # Performance changes
            performance_changes = {}
            
            if (latest_snapshot.get('classification_metrics') and 
                first_snapshot.get('classification_metrics')):
                
                latest_acc = latest_snapshot['classification_metrics'].get('accuracy', 0)
                first_acc = first_snapshot['classification_metrics'].get('accuracy', 0)
                performance_changes['accuracy_change'] = latest_acc - first_acc
                
                latest_f1 = latest_snapshot['classification_metrics'].get('f1_score', 0)
                first_f1 = first_snapshot['classification_metrics'].get('f1_score', 0)
                performance_changes['f1_change'] = latest_f1 - first_f1
            
            if (latest_snapshot.get('response_quality_metrics') and 
                first_snapshot.get('response_quality_metrics')):
                
                latest_quality = latest_snapshot['response_quality_metrics'].get('avg_quality_score', 0)
                first_quality = first_snapshot['response_quality_metrics'].get('avg_quality_score', 0)
                performance_changes['quality_change'] = latest_quality - first_quality
            
            report['performance_evolution'] = performance_changes
        
        # Generate visualizations
        if save_plots:
            self._generate_evolution_plots()
        
        return report
    
    def _generate_evolution_plots(self):
        """Generate evolution visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Agent Evolution Metrics Dashboard', fontsize=16)
        
        # Classification Performance Evolution
        if self.evolution_tracking['classification_performance']:
            df_class = pd.DataFrame(self.evolution_tracking['classification_performance'])
            df_class['timestamp'] = pd.to_datetime(df_class['timestamp'])
            
            axes[0, 0].plot(df_class['timestamp'], df_class['accuracy'], 'b-', label='Accuracy')
            axes[0, 0].plot(df_class['timestamp'], df_class['f1_score'], 'r-', label='F1 Score')
            axes[0, 0].set_title('Classification Performance Over Time')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Response Quality Evolution
        if self.evolution_tracking['response_quality']:
            df_quality = pd.DataFrame(self.evolution_tracking['response_quality'])
            df_quality['timestamp'] = pd.to_datetime(df_quality['timestamp'])
            
            axes[0, 1].plot(df_quality['timestamp'], df_quality['avg_quality'], 'g-', label='Quality Score')
            axes[0, 1].set_title('Response Quality Over Time')
            axes[0, 1].set_ylabel('Quality Score')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # System Health Evolution
        if self.evolution_tracking['system_health']:
            df_health = pd.DataFrame(self.evolution_tracking['system_health'])
            df_health['timestamp'] = pd.to_datetime(df_health['timestamp'])
            
            ax1 = axes[0, 2]
            ax2 = ax1.twinx()
            
            ax1.plot(df_health['timestamp'], df_health['throughput'], 'b-', label='Throughput')
            ax2.plot(df_health['timestamp'], df_health['error_rate'], 'r-', label='Error Rate')
            
            ax1.set_title('System Health Over Time')
            ax1.set_ylabel('Throughput (resp/sec)', color='b')
            ax2.set_ylabel('Error Rate', color='r')
            ax1.tick_params(axis='x', rotation=45)
        
        # Learning Effectiveness
        if self.evolution_tracking['learning_effectiveness']:
            df_learning = pd.DataFrame(self.evolution_tracking['learning_effectiveness'])
            df_learning['timestamp'] = pd.to_datetime(df_learning['timestamp'])
            
            axes[1, 0].plot(df_learning['timestamp'], df_learning['accuracy_improvement'], 'purple', label='Accuracy Improvement')
            axes[1, 0].plot(df_learning['timestamp'], df_learning['quality_improvement'], 'orange', label='Quality Improvement')
            axes[1, 0].set_title('Learning Effectiveness Over Time')
            axes[1, 0].set_ylabel('Improvement Score')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # User Satisfaction
        if self.evolution_tracking['user_satisfaction']:
            df_satisfaction = pd.DataFrame(self.evolution_tracking['user_satisfaction'])
            df_satisfaction['timestamp'] = pd.to_datetime(df_satisfaction['timestamp'])
            
            axes[1, 1].plot(df_satisfaction['timestamp'], df_satisfaction['overall_satisfaction'], 'brown', label='Overall Satisfaction')
            axes[1, 1].set_title('User Satisfaction Over Time')
            axes[1, 1].set_ylabel('Satisfaction Score')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Metrics Summary Heatmap
        if len(self.metrics_history) > 1:
            # Create a summary heatmap of key metrics
            summary_data = []
            for snapshot in self.metrics_history[-10:]:  # Last 10 snapshots
                row = {}
                if snapshot.get('classification_metrics'):
                    row['Accuracy'] = snapshot['classification_metrics'].get('accuracy', 0)
                    row['F1 Score'] = snapshot['classification_metrics'].get('f1_score', 0)
                if snapshot.get('response_quality_metrics'):
                    row['Quality'] = snapshot['response_quality_metrics'].get('avg_quality_score', 0)
                    row['Response Time'] = 1 / max(snapshot['response_quality_metrics'].get('avg_response_time', 1), 0.1)
                if snapshot.get('system_health_metrics'):
                    row['Throughput'] = snapshot['system_health_metrics'].get('throughput', 0)
                    row['Error Rate'] = 1 - snapshot['system_health_metrics'].get('error_rate', 0)
                
                summary_data.append(row)
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                sns.heatmap(df_summary.T, annot=True, cmap='RdYlGn', ax=axes[1, 2])
                axes[1, 2].set_title('Recent Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig('agent_evolution_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Evolution dashboard saved as 'agent_evolution_dashboard.png'")
    
    def save_metrics_data(self, filepath=None):
        """Save all metrics data to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"metrics_data_{timestamp}.json"
        
        data = {
            'metrics_history': self.metrics_history,
            'evolution_tracking': self.evolution_tracking,
            'performance_baselines': self.performance_baselines
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Metrics data saved to {filepath}")
        return filepath
    
    def load_metrics_data(self, filepath):
        """Load metrics data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.metrics_history = data.get('metrics_history', [])
            self.evolution_tracking = data.get('evolution_tracking', {})
            self.performance_baselines = data.get('performance_baselines', {})
            
            print(f"Metrics data loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading metrics data: {e}")
            return False

    def establish_baseline(self, quality_score, classification_accuracy, reward):
        """Establish performance baselines for improvement tracking"""
        if not self.performance_baselines['baseline_established']:
            self.performance_baselines['initial_quality_score'] = quality_score
            self.performance_baselines['initial_classification_accuracy'] = classification_accuracy
            self.performance_baselines['initial_reward'] = reward
            self.performance_baselines['baseline_established'] = True
            
            # Initialize best scores
            self.improvement_history['best_quality_score'] = quality_score
            self.improvement_history['best_accuracy'] = classification_accuracy
            self.improvement_history['best_reward'] = reward
            
            print(f"Performance baselines established:")
            print(f"  Initial Quality Score: {quality_score:.4f}")
            print(f"  Initial Classification Accuracy: {classification_accuracy:.4f}")
            print(f"  Initial Reward: {reward:.4f}")
    
    def update_performance_memory(self, quality_score, accuracy, reward):
        """Update performance memory with recent scores"""
        window_size = self.performance_memory['window_size']
        
        # Add new scores
        self.performance_memory['recent_quality_scores'].append(quality_score)
        self.performance_memory['recent_accuracies'].append(accuracy)
        self.performance_memory['recent_rewards'].append(reward)
        
        # Maintain window size
        if len(self.performance_memory['recent_quality_scores']) > window_size:
            self.performance_memory['recent_quality_scores'].pop(0)
            self.performance_memory['recent_accuracies'].pop(0)
            self.performance_memory['recent_rewards'].pop(0)
    
    def calculate_improvement_bonus(self, current_quality, current_accuracy, current_reward):
        """Calculate reward bonus based on sustained improvement trends"""
        if not self.performance_baselines['baseline_established']:
            return 0.0
        
        bonus = 0.0
        
        # Quality improvement bonus
        quality_improvement = current_quality - self.performance_baselines['initial_quality_score']
        if quality_improvement > 0:
            # Progressive bonus for quality improvement
            quality_bonus = min(quality_improvement * 2.0, 0.5)  # Max 0.5 bonus
            bonus += quality_bonus
            
            # Extra bonus for reaching target
            if current_quality >= self.performance_baselines['target_quality_score']:
                bonus += 0.3
        
        # Accuracy improvement bonus
        accuracy_improvement = current_accuracy - self.performance_baselines['initial_classification_accuracy']
        if accuracy_improvement > 0:
            # Progressive bonus for accuracy improvement
            accuracy_bonus = min(accuracy_improvement * 1.5, 0.4)  # Max 0.4 bonus
            bonus += accuracy_bonus
            
            # Extra bonus for reaching target
            if current_accuracy >= self.performance_baselines['target_classification_accuracy']:
                bonus += 0.2
        
        # Consistency bonus for sustained improvement
        if len(self.performance_memory['recent_quality_scores']) >= 5:
            recent_quality_avg = np.mean(self.performance_memory['recent_quality_scores'][-5:])
            if recent_quality_avg > self.performance_baselines['initial_quality_score']:
                bonus += 0.1  # Consistency bonus
        
        # Best performance bonus
        if current_quality > self.improvement_history['best_quality_score']:
            self.improvement_history['best_quality_score'] = current_quality
            bonus += 0.15  # New best quality bonus
            
        if current_accuracy > self.improvement_history['best_accuracy']:
            self.improvement_history['best_accuracy'] = current_accuracy
            bonus += 0.15  # New best accuracy bonus
            
        if current_reward > self.improvement_history['best_reward']:
            self.improvement_history['best_reward'] = current_reward
            bonus += 0.1  # New best reward bonus
        
        # Consecutive improvement streak bonus
        self._update_improvement_streak(current_quality, current_accuracy)
        if self.improvement_history['consecutive_improvements'] >= 3:
            streak_bonus = min(self.improvement_history['consecutive_improvements'] * 0.05, 0.3)
            bonus += streak_bonus
        
        return min(bonus, 1.0)  # Cap total bonus at 1.0
    
    def _update_improvement_streak(self, current_quality, current_accuracy):
        """Update consecutive improvement streak tracking"""
        if len(self.performance_memory['recent_quality_scores']) < 2:
            return
        
        prev_quality = self.performance_memory['recent_quality_scores'][-2]
        prev_accuracy = self.performance_memory['recent_accuracies'][-2]
        
        # Check if both quality and accuracy improved
        if current_quality > prev_quality and current_accuracy > prev_accuracy:
            self.improvement_history['consecutive_improvements'] += 1
        else:
            # Reset streak if no improvement
            if self.improvement_history['consecutive_improvements'] > 0:
                self.improvement_history['improvement_streaks'].append(
                    self.improvement_history['consecutive_improvements']
                )
            self.improvement_history['consecutive_improvements'] = 0
    
    def get_improvement_progress(self):
        """Get current improvement progress towards targets"""
        if not self.performance_baselines['baseline_established']:
            return {}
        
        progress = {}
        
        # Quality progress
        if self.improvement_history['best_quality_score'] > 0:
            quality_progress = (
                (self.improvement_history['best_quality_score'] - self.performance_baselines['initial_quality_score']) /
                (self.performance_baselines['target_quality_score'] - self.performance_baselines['initial_quality_score'])
            )
            progress['quality_progress'] = min(max(quality_progress, 0), 1.0)
        
        # Accuracy progress
        if self.improvement_history['best_accuracy'] > 0:
            accuracy_progress = (
                (self.improvement_history['best_accuracy'] - self.performance_baselines['initial_classification_accuracy']) /
                (self.performance_baselines['target_classification_accuracy'] - self.performance_baselines['initial_classification_accuracy'])
            )
            progress['accuracy_progress'] = min(max(accuracy_progress, 0), 1.0)
        
        # Overall progress
        if 'quality_progress' in progress and 'accuracy_progress' in progress:
            progress['overall_progress'] = (progress['quality_progress'] + progress['accuracy_progress']) / 2
        
        # Improvement statistics
        progress['consecutive_improvements'] = self.improvement_history['consecutive_improvements']
        progress['best_streak'] = max(self.improvement_history['improvement_streaks']) if self.improvement_history['improvement_streaks'] else 0
        progress['total_improvements'] = len(self.improvement_history['quality_improvements'])
        
        return progress

def test_metrics_framework():
    """Test the metrics framework with sample data"""
    print("Testing Metrics Framework...")
    
    # Initialize framework
    framework = MetricsFramework()
    
    # Simulate some classification results
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9, 0.1, 0.6, 0.7, 0.8, 0.2])
    
    # Calculate classification metrics
    class_metrics = framework.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    print("Classification Metrics:")
    for key, value in class_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Simulate response data
    response_data = []
    for i in range(20):
        response_data.append({
            'quality_score': np.random.uniform(0.5, 1.0),
            'response_time': np.random.uniform(0.1, 2.0),
            'urgency_level': np.random.choice([0, 1])
        })
    
    # Calculate response metrics
    response_metrics = framework.calculate_response_quality_metrics(response_data)
    print("\nResponse Quality Metrics:")
    for key, value in response_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Calculate system health metrics
    system_metrics = framework.calculate_system_health_metrics(response_data)
    print("\nSystem Health Metrics:")
    for key, value in system_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Record metrics snapshots
    for i in range(5):
        # Simulate improving performance over time
        improved_class_metrics = {k: v + i * 0.01 for k, v in class_metrics.items() if isinstance(v, float)}
        improved_response_metrics = {k: v + i * 0.005 for k, v in response_metrics.items() if isinstance(v, float)}
        
        framework.record_metrics_snapshot(
            classifier_metrics=improved_class_metrics,
            response_metrics=improved_response_metrics,
            system_metrics=system_metrics
        )
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report()
    print("\nComprehensive Report Generated")
    
    return framework

if __name__ == "__main__":
    framework = test_metrics_framework()