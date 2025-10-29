"""
Results Analysis Tools

Aggregates results across models, creates comparison tables, and generates visualizations.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from statistics import mean, stdev
import csv


class ResultsAnalyzer:
    """
    Analyzes AVER benchmark results across multiple models

    Creates comparison tables, calculates statistics, exports data.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize analyzer

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.all_results = []

    def load_all_results(self) -> int:
        """Load all result JSON files"""
        self.all_results = []

        for json_file in self.results_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.all_results.append(data)

        print(f"Loaded {len(self.all_results)} result files")
        return len(self.all_results)

    def create_model_comparison_table(self) -> Dict[str, Any]:
        """
        Create Table 1: Model Comparison

        Columns: Model, Avg Detection, Avg Diagnosis, Avg Recovery, Total Score, Tasks Completed
        """
        model_stats = {}

        for result_file in self.all_results:
            agent_id = result_file.get("agent_id", "unknown")

            # Extract model name from first task result
            model_name = "unknown"
            if result_file.get("results"):
                model_name = result_file["results"][0].get("model_name", "unknown")

            if model_name not in model_stats:
                model_stats[model_name] = {
                    "detection_scores": [],
                    "diagnosis_scores": [],
                    "recovery_scores": [],
                    "total_scores": [],
                    "tasks_completed": 0
                }

            # Aggregate scores
            for task_result in result_file.get("results", []):
                scores = task_result.get("scores", {})
                model_stats[model_name]["detection_scores"].append(scores.get("detection", 0))
                model_stats[model_name]["diagnosis_scores"].append(scores.get("diagnosis", 0))
                model_stats[model_name]["recovery_scores"].append(scores.get("recovery", 0))
                model_stats[model_name]["total_scores"].append(scores.get("total", 0))
                model_stats[model_name]["tasks_completed"] += 1

        # Calculate averages
        comparison_table = []
        for model, stats in model_stats.items():
            comparison_table.append({
                "model": model,
                "avg_detection": round(mean(stats["detection_scores"]), 3) if stats["detection_scores"] else 0,
                "avg_diagnosis": round(mean(stats["diagnosis_scores"]), 3) if stats["diagnosis_scores"] else 0,
                "avg_recovery": round(mean(stats["recovery_scores"]), 3) if stats["recovery_scores"] else 0,
                "avg_total": round(mean(stats["total_scores"]), 1) if stats["total_scores"] else 0,
                "tasks_completed": stats["tasks_completed"],
                "std_dev": round(stdev(stats["total_scores"]), 1) if len(stats["total_scores"]) > 1 else 0
            })

        # Sort by total score
        comparison_table.sort(key=lambda x: x["avg_total"], reverse=True)

        return {
            "title": "Model Comparison - Overall Performance",
            "data": comparison_table
        }

    def create_category_analysis_table(self) -> Dict[str, Any]:
        """
        Create Table 2: Performance by Category

        Shows: Model × Category performance matrix
        """
        # Structure: model -> category -> scores
        model_category_scores = defaultdict(lambda: defaultdict(list))

        for result_file in self.all_results:
            for task_result in result_file.get("results", []):
                model = task_result.get("model_name", "unknown")
                task_id = task_result.get("task_id", "")
                total_score = task_result.get("scores", {}).get("total", 0)

                # Extract category from task_id (e.g., "aver_hallucination_...")
                category = task_id.split('_')[1] if '_' in task_id else "unknown"

                model_category_scores[model][category].append(total_score)

        # Build table
        category_table = []
        for model, categories in model_category_scores.items():
            row = {"model": model}
            for category, scores in categories.items():
                row[category] = round(mean(scores), 1) if scores else 0.0
            category_table.append(row)

        return {
            "title": "Performance by Error Category",
            "data": category_table
        }

    def create_difficulty_analysis_table(self) -> Dict[str, Any]:
        """
        Create Table 3: Performance by Difficulty

        Shows: Model × Difficulty level matrix
        """
        model_difficulty_scores = defaultdict(lambda: defaultdict(list))

        for result_file in self.all_results:
            for task_result in result_file.get("results", []):
                model = task_result.get("model_name", "unknown")
                task_id = task_result.get("task_id", "")
                total_score = task_result.get("scores", {}).get("total", 0)

                # Extract difficulty from task_id (e.g., "..._{difficulty}_...")
                parts = task_id.split('_')
                # Difficulty is typically before the last number
                difficulty = "unknown"
                for i, part in enumerate(parts):
                    if part.isdigit() and i < len(parts) - 1:
                        difficulty = f"Level_{part}"
                        break

                model_difficulty_scores[model][difficulty].append(total_score)

        # Build table
        difficulty_table = []
        for model, difficulties in model_difficulty_scores.items():
            row = {"model": model}
            for difficulty, scores in sorted(difficulties.items()):
                row[difficulty] = round(mean(scores), 1) if scores else 0.0
            difficulty_table.append(row)

        return {
            "title": "Performance by Difficulty Level",
            "data": difficulty_table
        }

    def export_to_csv(self, table_data: Dict, output_file: str):
        """Export table to CSV for statistical analysis"""
        with open(output_file, 'w', newline='') as f:
            if not table_data["data"]:
                return

            writer = csv.DictWriter(f, fieldnames=table_data["data"][0].keys())
            writer.writeheader()
            writer.writerows(table_data["data"])

        print(f"Exported to: {output_file}")

    def print_markdown_table(self, table_data: Dict):
        """Print table in markdown format for paper"""
        print(f"\n## {table_data['title']}\n")

        if not table_data["data"]:
            print("No data")
            return

        # Get headers
        headers = list(table_data["data"][0].keys())

        # Print header
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---" for _ in headers]) + "|")

        # Print rows
        for row in table_data["data"]:
            values = [str(row.get(h, "")) for h in headers]
            print("| " + " | ".join(values) + " |")

        print()

    def generate_all_tables(self):
        """Generate all analysis tables"""
        print("="*80)
        print("AVER RESULTS ANALYSIS")
        print("="*80)

        # Table 1: Model Comparison
        table1 = self.create_model_comparison_table()
        self.print_markdown_table(table1)
        self.export_to_csv(table1, "paper/tables/table1_model_comparison.csv")

        # Table 2: By Category
        table2 = self.create_category_analysis_table()
        self.print_markdown_table(table2)
        self.export_to_csv(table2, "paper/tables/table2_by_category.csv")

        # Table 3: By Difficulty
        table3 = self.create_difficulty_analysis_table()
        self.print_markdown_table(table3)
        self.export_to_csv(table3, "paper/tables/table3_by_difficulty.csv")

        print("="*80)
        print("All tables generated!")
        print("CSV files saved to: paper/tables/")


def create_results_summary(results_file: str) -> Dict[str, Any]:
    """
    Create comprehensive summary from a single results file

    Perfect schema for research analysis.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    summary = {
        # Metadata
        "agent_id": data.get("agent_id"),
        "model_name": data["results"][0].get("model_name") if data.get("results") else "unknown",
        "timestamp": data.get("timestamp"),
        "num_tasks": data.get("num_tasks"),

        # Aggregate scores
        "aggregate_scores": data.get("aggregate_scores", {}),

        # By category breakdown
        "by_category": {},

        # By difficulty breakdown
        "by_difficulty": {},

        # Success rate analysis
        "success_rate_60": 0,  # Tasks scoring >= 60
        "success_rate_80": 0,  # Tasks scoring >= 80

        # Detection/Diagnosis/Recovery breakdown
        "detection_distribution": {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0},
        "diagnosis_distribution": {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0},
        "recovery_distribution": {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0},

        # All task results
        "tasks": data.get("results", [])
    }

    # Calculate success rates and distributions
    for task_result in data.get("results", []):
        total = task_result["scores"]["total"]
        detection = task_result["scores"]["detection"] * 100
        diagnosis = task_result["scores"]["diagnosis"] * 100
        recovery = task_result["scores"]["recovery"] * 100

        # Success rates
        if total >= 60:
            summary["success_rate_60"] += 1
        if total >= 80:
            summary["success_rate_80"] += 1

        # Distributions
        for score, dist_key in [(detection, "detection_distribution"),
                                 (diagnosis, "diagnosis_distribution"),
                                 (recovery, "recovery_distribution")]:
            if score < 25:
                summary[dist_key]["0-25%"] += 1
            elif score < 50:
                summary[dist_key]["25-50%"] += 1
            elif score < 75:
                summary[dist_key]["50-75%"] += 1
            else:
                summary[dist_key]["75-100%"] += 1

        # Category aggregation
        task_id = task_result.get("task_id", "")
        category = task_id.split('_')[1] if '_' in task_id else "unknown"
        if category not in summary["by_category"]:
            summary["by_category"][category] = []
        summary["by_category"][category].append(total)

        # Difficulty aggregation
        parts = task_id.split('_')
        for i, part in enumerate(parts):
            if part.isdigit() and i < len(parts) - 1:
                diff = f"difficulty_{part}"
                if diff not in summary["by_difficulty"]:
                    summary["by_difficulty"][diff] = []
                summary["by_difficulty"][diff].append(total)
                break

    # Calculate averages
    for category, scores in summary["by_category"].items():
        summary["by_category"][category] = round(mean(scores), 1)

    for difficulty, scores in summary["by_difficulty"].items():
        summary["by_difficulty"][difficulty] = round(mean(scores), 1)

    return summary


if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analyzer.load_all_results()
    analyzer.generate_all_tables()
