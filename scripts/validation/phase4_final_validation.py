#!/usr/bin/env python3
"""
Phase 4: Final Validation Summary

Purpose: Aggregate all phase results and generate final validation report.

Outputs:
- Comprehensive validation report
- Pass/Fail summary for all stages
- Recommendations for next steps
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

RESULTS_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results")
OUTPUT_DIR = RESULTS_ROOT / "phase4_final_validation"

PHASE_DIRS = {
    'phase1': RESULTS_ROOT / "phase1_tau_stability",
    'phase2': RESULTS_ROOT / "phase2_guided_search",
    'phase3': RESULTS_ROOT / "phase3_stage3_revalidation",
}


# ============================================================================
# Result Loading
# ============================================================================

def find_latest_run(phase_dir: Path) -> Optional[Path]:
    """Find the most recent run directory."""
    if not phase_dir.exists():
        return None
    runs = sorted(phase_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def load_phase1_summary(run_dir: Path) -> Dict:
    """Load Phase 1 summary."""
    summary = {
        'status': 'not_run',
        'stable_params_found': False,
        'best_stability_rate': 0.0,
        'best_params': None,
        'key_findings': [],
    }

    if not run_dir:
        return summary

    report_file = run_dir / "stability_report.json"
    if not report_file.exists():
        return summary

    with open(report_file) as f:
        report = json.load(f)

    # Extract key metrics
    param_analysis = report.get('parameter_analysis', {})
    if param_analysis:
        best_key = max(param_analysis.keys(),
                      key=lambda k: param_analysis[k]['stability_rate'])
        best = param_analysis[best_key]

        summary['status'] = 'completed'
        summary['best_stability_rate'] = best['stability_rate']
        summary['best_params'] = best_key
        summary['stable_params_found'] = best['stability_rate'] >= 0.4

        # Key findings
        if best['stability_rate'] >= 0.7:
            summary['key_findings'].append("Strong stable parameter combination found")
        elif best['stability_rate'] >= 0.4:
            summary['key_findings'].append("Marginal stable parameters found")
        else:
            summary['key_findings'].append("No stable parameters found")

    # Collapse analysis
    collapse = report.get('collapse_analysis', {})
    if collapse.get('collapse_rate', 0) > 0.3:
        summary['key_findings'].append(f"High collapse rate: {collapse['collapse_rate']*100:.1f}%")

    return summary


def load_phase2_summary(run_dir: Path) -> Dict:
    """Load Phase 2 summary."""
    summary = {
        'status': 'not_run',
        'guided_search_effective': False,
        'best_search_window': None,
        'false_peak_reduction': 0.0,
        'key_findings': [],
    }

    if not run_dir:
        return summary

    report_file = run_dir / "comparison_report.json"
    if not report_file.exists():
        return summary

    with open(report_file) as f:
        report = json.load(f)

    summary['status'] = 'completed'

    # Find best search window
    window_analysis = report.get('search_window_analysis', {})
    if window_analysis:
        best_key = min(window_analysis.keys(),
                      key=lambda k: window_analysis[k]['false_peak_rate_guided'])
        best = window_analysis[best_key]

        summary['best_search_window'] = best_key

        # Calculate improvement
        overall = report.get('overall_comparison', {})
        fp_global = overall.get('false_peak_rate_global', 0)
        fp_guided = best['false_peak_rate_guided']

        if fp_global > 0:
            summary['false_peak_reduction'] = (fp_global - fp_guided) / fp_global * 100
            summary['guided_search_effective'] = summary['false_peak_reduction'] > 10

        if summary['false_peak_reduction'] > 30:
            summary['key_findings'].append("Guided search significantly reduces false peaks")
        elif summary['false_peak_reduction'] > 10:
            summary['key_findings'].append("Guided search moderately effective")
        else:
            summary['key_findings'].append("Guided search shows limited improvement")

    # Stability improvement
    stability = report.get('stability_comparison', {})
    if stability.get('stability_improvement', 0) > 10:
        summary['key_findings'].append(f"tau stability improved by {stability['stability_improvement']:.1f}%")

    return summary


def load_phase3_summary(run_dir: Path) -> Dict:
    """Load Phase 3 summary."""
    summary = {
        'status': 'not_run',
        'stage3_improved': False,
        'old_pass_rate': 0.0,
        'new_pass_rate': 0.0,
        'improvement': 0.0,
        'key_findings': [],
        'failure_reasons': {},
    }

    if not run_dir:
        return summary

    report_file = run_dir / "revalidation_report.json"
    if not report_file.exists():
        return summary

    with open(report_file) as f:
        report = json.load(f)

    summary['status'] = 'completed'

    # Pass rates
    old_method = report.get('old_method', {})
    new_method = report.get('new_method', {})
    comparison = report.get('comparison', {})

    summary['old_pass_rate'] = old_method.get('pass_rate', 0) * 100
    summary['new_pass_rate'] = new_method.get('pass_rate', 0) * 100
    summary['improvement'] = comparison.get('pass_rate_improvement', 0)
    summary['stage3_improved'] = summary['improvement'] > 10

    if summary['improvement'] > 30:
        summary['key_findings'].append(f"Significant Stage 3 improvement: +{summary['improvement']:.1f}%")
    elif summary['improvement'] > 10:
        summary['key_findings'].append(f"Moderate Stage 3 improvement: +{summary['improvement']:.1f}%")
    elif summary['improvement'] > 0:
        summary['key_findings'].append(f"Marginal Stage 3 improvement: +{summary['improvement']:.1f}%")
    else:
        summary['key_findings'].append("No Stage 3 improvement observed")

    # Failure analysis
    failure_analysis = report.get('failure_analysis', {}).get('new_method', {})
    summary['failure_reasons'] = failure_analysis

    if failure_analysis:
        top_reason = max(failure_analysis.items(), key=lambda x: x[1])
        summary['key_findings'].append(f"Primary failure reason: {top_reason[0]} ({top_reason[1]} cases)")

    return summary


# ============================================================================
# Final Validation Report
# ============================================================================

def generate_final_report(output_dir: Path) -> Dict:
    """Generate comprehensive final validation report."""
    print("\n" + "="*70)
    print("  PHASE 4: Final Validation Summary")
    print("="*70)

    # Load all phase summaries
    phase1_run = find_latest_run(PHASE_DIRS['phase1'])
    phase2_run = find_latest_run(PHASE_DIRS['phase2'])
    phase3_run = find_latest_run(PHASE_DIRS['phase3'])

    phase1 = load_phase1_summary(phase1_run)
    phase2 = load_phase2_summary(phase2_run)
    phase3 = load_phase3_summary(phase3_run)

    report = {
        'timestamp': datetime.now().isoformat(),
        'phases': {
            'phase1_tau_stability': phase1,
            'phase2_guided_search': phase2,
            'phase3_stage3_revalidation': phase3,
        },
        'overall_assessment': {},
        'recommendations': [],
        'decision': '',
    }

    # Print phase summaries
    print("\n  Phase 1: tau Stability Diagnosis")
    print("  " + "-"*50)
    print(f"    Status: {phase1['status']}")
    if phase1['status'] == 'completed':
        print(f"    Best stability rate: {phase1['best_stability_rate']*100:.1f}%")
        print(f"    Best parameters: {phase1['best_params']}")
        for finding in phase1['key_findings']:
            print(f"    - {finding}")

    print("\n  Phase 2: Guided Peak Search")
    print("  " + "-"*50)
    print(f"    Status: {phase2['status']}")
    if phase2['status'] == 'completed':
        print(f"    Best search window: {phase2['best_search_window']}")
        print(f"    False peak reduction: {phase2['false_peak_reduction']:.1f}%")
        for finding in phase2['key_findings']:
            print(f"    - {finding}")

    print("\n  Phase 3: Stage 3 Re-validation")
    print("  " + "-"*50)
    print(f"    Status: {phase3['status']}")
    if phase3['status'] == 'completed':
        print(f"    Old pass rate: {phase3['old_pass_rate']:.1f}%")
        print(f"    New pass rate: {phase3['new_pass_rate']:.1f}%")
        print(f"    Improvement: {phase3['improvement']:+.1f}%")
        for finding in phase3['key_findings']:
            print(f"    - {finding}")

    # Overall assessment
    print("\n" + "="*70)
    print("  OVERALL ASSESSMENT")
    print("="*70)

    # Determine overall status
    completed_phases = sum([
        phase1['status'] == 'completed',
        phase2['status'] == 'completed',
        phase3['status'] == 'completed',
    ])

    report['overall_assessment'] = {
        'phases_completed': completed_phases,
        'total_phases': 3,
        'tau_stability_achieved': phase1['stable_params_found'],
        'guided_search_effective': phase2['guided_search_effective'],
        'stage3_improved': phase3['stage3_improved'],
    }

    # Generate recommendations
    if phase1['status'] != 'completed':
        report['recommendations'].append("Run Phase 1 to diagnose tau stability")
    elif not phase1['stable_params_found']:
        report['recommendations'].append("Consider alternative signals (chirp, pink noise) for evaluation")
        report['recommendations'].append("Review frequency band selection")

    if phase2['status'] != 'completed':
        report['recommendations'].append("Run Phase 2 to validate guided search")
    elif not phase2['guided_search_effective']:
        report['recommendations'].append("Problem may not be false peaks - investigate signal quality")

    if phase3['status'] != 'completed':
        report['recommendations'].append("Run Phase 3 to re-validate Stage 3")
    else:
        if phase3['new_pass_rate'] < 60:
            report['recommendations'].append("Stage 3 pass rate still low - investigate OMP alignment")
        if 'baseline_unreliable' in phase3['failure_reasons']:
            report['recommendations'].append("Focus on improving baseline reliability")
        if 'low_psr' in phase3['failure_reasons']:
            report['recommendations'].append("Signal quality issues - review preprocessing")

    # Final decision
    if completed_phases == 3:
        if phase3['new_pass_rate'] >= 80:
            report['decision'] = "VALIDATION_SUCCESS"
            decision_msg = "Validation successful! Stage 3 pass rate meets target."
        elif phase3['new_pass_rate'] >= 60:
            report['decision'] = "VALIDATION_PARTIAL"
            decision_msg = "Partial validation. Stage 3 improved but not fully reliable."
        else:
            report['decision'] = "VALIDATION_NEEDS_WORK"
            decision_msg = "More work needed. Review recommendations above."
    else:
        report['decision'] = "INCOMPLETE"
        decision_msg = f"Only {completed_phases}/3 phases completed. Run remaining phases."

    print(f"\n  Decision: {report['decision']}")
    print(f"  {decision_msg}")

    print("\n  Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"    {i}. {rec}")

    return report


def plot_validation_summary(report: Dict, output_dir: Path):
    """Generate summary visualization."""
    print("\n  Generating summary visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Phase completion status
    ax = axes[0, 0]
    phases = ['Phase 1\ntau Stability', 'Phase 2\nGuided Search', 'Phase 3\nStage 3']
    statuses = [
        report['phases']['phase1_tau_stability']['status'],
        report['phases']['phase2_guided_search']['status'],
        report['phases']['phase3_stage3_revalidation']['status'],
    ]
    colors = ['green' if s == 'completed' else 'red' for s in statuses]

    ax.barh(phases, [1 if s == 'completed' else 0 for s in statuses], color=colors, alpha=0.7)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel('Completed')
    ax.set_title('Phase Completion Status')
    for i, s in enumerate(statuses):
        ax.text(0.5, i, s, ha='center', va='center', fontweight='bold')

    # Plot 2: Key metrics
    ax = axes[0, 1]
    metrics = ['tau Stability\nRate', 'False Peak\nReduction', 'Stage 3\nPass Rate']
    values = [
        report['phases']['phase1_tau_stability']['best_stability_rate'] * 100,
        report['phases']['phase2_guided_search']['false_peak_reduction'],
        report['phases']['phase3_stage3_revalidation']['new_pass_rate'],
    ]

    bars = ax.bar(metrics, values, color=['blue', 'orange', 'green'], alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Key Metrics')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
               ha='center', fontweight='bold')

    # Plot 3: Stage 3 improvement
    ax = axes[1, 0]
    phase3 = report['phases']['phase3_stage3_revalidation']
    if phase3['status'] == 'completed':
        methods = ['Old Method', 'New Method']
        rates = [phase3['old_pass_rate'], phase3['new_pass_rate']]
        colors = ['red', 'green']

        bars = ax.bar(methods, rates, color=colors, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('Stage 3 Improvement')

        improvement = phase3['improvement']
        ax.annotate(f'+{improvement:.1f}%', xy=(0.5, max(rates)),
                   xytext=(0.5, max(rates) + 10),
                   ha='center', fontsize=12, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black'))
    else:
        ax.text(0.5, 0.5, 'Phase 3 not completed', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot 4: Failure analysis
    ax = axes[1, 1]
    failure_reasons = phase3.get('failure_reasons', {})
    if failure_reasons:
        reasons = list(failure_reasons.keys())
        counts = list(failure_reasons.values())

        ax.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        ax.set_title('Stage 3 Failure Reasons')
    else:
        ax.text(0.5, 0.5, 'No failure data', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'final_validation_summary.png', dpi=150)
    plt.close()

    print(f"    Saved to {output_dir / 'final_validation_summary.png'}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run Phase 4: Final Validation Summary."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")

    # Generate report
    report = generate_final_report(output_dir)

    # Save report
    report_file = output_dir / "final_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved to: {report_file}")

    # Generate visualization
    plot_validation_summary(report, output_dir)

    # Print final summary
    print("\n" + "="*70)
    print("  FINAL VALIDATION COMPLETE")
    print("="*70)
    print(f"\n  Decision: {report['decision']}")
    print(f"  Results: {output_dir}")

    # Return exit code based on decision
    if report['decision'] == 'VALIDATION_SUCCESS':
        return 0
    elif report['decision'] == 'VALIDATION_PARTIAL':
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())

