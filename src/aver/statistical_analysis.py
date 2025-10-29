"""
Statistical Analysis Tools

Implements significance testing and statistical analysis for research paper.
"""

from typing import List, Dict, Any, Tuple
from statistics import mean, stdev
import math


def calculate_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for scores

    Args:
        scores: List of scores
        confidence: Confidence level (default: 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    if len(scores) < 2:
        return (0.0, 0.0)

    n = len(scores)
    mean_score = mean(scores)
    std_error = stdev(scores) / math.sqrt(n)

    # t-value for 95% confidence (approximation)
    # For proper implementation, use scipy.stats.t.ppf
    t_value = 1.96 if n >= 30 else 2.0  # Conservative estimate

    margin = t_value * std_error
    lower = mean_score - margin
    upper = mean_score + margin

    return (lower, upper)


def t_test_independent(
    group1_scores: List[float],
    group2_scores: List[float]
) -> Dict[str, Any]:
    """
    Independent samples t-test

    Tests if two models have significantly different performance.

    Args:
        group1_scores: Scores from model 1
        group2_scores: Scores from model 2

    Returns:
        Dict with t-statistic, p-value, significant flag
    """
    n1 = len(group1_scores)
    n2 = len(group2_scores)

    if n1 < 2 or n2 < 2:
        return {'error': 'Insufficient data'}

    mean1 = mean(group1_scores)
    mean2 = mean(group2_scores)

    var1 = stdev(group1_scores) ** 2
    var2 = stdev(group2_scores) ** 2

    # Pooled standard error
    pooled_se = math.sqrt(var1/n1 + var2/n2)

    if pooled_se == 0:
        return {'error': 'No variance'}

    # t-statistic
    t_stat = (mean1 - mean2) / pooled_se

    # Degrees of freedom (Welch's approximation)
    df = ((var1/n1 + var2/n2) ** 2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # p-value (simplified - use scipy for exact)
    # For |t| > 2, p < 0.05 (rough approximation)
    significant = abs(t_stat) > 2.0

    return {
        't_statistic': round(t_stat, 3),
        'degrees_freedom': round(df, 1),
        'mean_difference': round(mean1 - mean2, 2),
        'significant_at_p05': significant,
        'effect_size': 'large' if abs(mean1 - mean2) > 10 else 'medium' if abs(mean1 - mean2) > 5 else 'small'
    }


def cohens_kappa(
    annotator1: List[float],
    annotator2: List[float]
) -> float:
    """
    Calculate Cohen's Kappa for inter-annotator agreement

    Args:
        annotator1: Scores from annotator 1
        annotator2: Scores from annotator 2

    Returns:
        Kappa value (-1 to 1)
    """
    if len(annotator1) != len(annotator2):
        raise ValueError("Annotators must have same number of ratings")

    n = len(annotator1)

    # Observed agreement
    agreements = sum(1 for i in range(n) if annotator1[i] == annotator2[i])
    p_observed = agreements / n

    # Expected agreement (by chance)
    # Count occurrences of each score
    scores = set(annotator1 + annotator2)
    p_expected = 0.0

    for score in scores:
        p1 = annotator1.count(score) / n
        p2 = annotator2.count(score) / n
        p_expected += p1 * p2

    # Kappa
    if p_expected == 1.0:
        return 1.0

    kappa = (p_observed - p_expected) / (1 - p_expected)

    return round(kappa, 3)


def calculate_effect_size(mean1: float, mean2: float, pooled_std: float) -> float:
    """
    Calculate Cohen's d effect size

    Args:
        mean1: Mean of group 1
        mean2: Mean of group 2
        pooled_std: Pooled standard deviation

    Returns:
        Effect size (Cohen's d)
    """
    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return round(d, 3)


def analyze_model_comparison(
    model_results: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    Complete statistical comparison of models

    Args:
        model_results: Dict mapping model_name -> list of scores

    Returns:
        Statistical analysis results
    """
    models = list(model_results.keys())
    analysis = {
        'models': models,
        'num_models': len(models),
        'pairwise_comparisons': []
    }

    # Pairwise comparisons
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]

            comparison = t_test_independent(
                model_results[model1],
                model_results[model2]
            )

            comparison['model1'] = model1
            comparison['model2'] = model2

            analysis['pairwise_comparisons'].append(comparison)

    # Overall statistics
    analysis['model_statistics'] = {}
    for model, scores in model_results.items():
        ci = calculate_confidence_interval(scores)
        analysis['model_statistics'][model] = {
            'mean': round(mean(scores), 2),
            'std_dev': round(stdev(scores), 2) if len(scores) > 1 else 0,
            'ci_95_lower': round(ci[0], 2),
            'ci_95_upper': round(ci[1], 2),
            'n': len(scores)
        }

    return analysis


def generate_significance_report(analysis: Dict[str, Any]) -> str:
    """Generate markdown report of statistical analysis"""

    report = []
    report.append("# Statistical Analysis Report")
    report.append("")
    report.append("## Model Performance Statistics")
    report.append("")

    # Model statistics table
    report.append("| Model | Mean | Std Dev | 95% CI |")
    report.append("|-------|------|---------|--------|")

    for model, stats in analysis['model_statistics'].items():
        ci = f"[{stats['ci_95_lower']}, {stats['ci_95_upper']}]"
        report.append(f"| {model} | {stats['mean']} | {stats['std_dev']} | {ci} |")

    report.append("")
    report.append("## Pairwise Comparisons (t-tests)")
    report.append("")

    for comp in analysis['pairwise_comparisons']:
        if 'error' in comp:
            continue

        sig = "✓ Significant (p<0.05)" if comp['significant_at_p05'] else "✗ Not significant"
        report.append(f"**{comp['model1']} vs {comp['model2']}**")
        report.append(f"- Mean difference: {comp['mean_difference']}")
        report.append(f"- t-statistic: {comp['t_statistic']}")
        report.append(f"- {sig}")
        report.append(f"- Effect size: {comp['effect_size']}")
        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    print("Statistical Analysis Tools")
    print()
    print("Functions:")
    print("  - calculate_confidence_interval()")
    print("  - t_test_independent()")
    print("  - cohens_kappa()")
    print("  - analyze_model_comparison()")
    print("  - generate_significance_report()")
