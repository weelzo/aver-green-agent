"""
AVER Dashboard - Comprehensive Benchmark Results Visualization
Displays benchmark performance metrics with Tailwind CSS and Chart.js
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List, Dict
import json

app = FastAPI()


# Comprehensive benchmark data with multiple dimensions
BENCHMARK_DATA = {
    "title": "AVER: Agent Verification & Error Recovery Benchmark",
    "subtitle": "Evaluating AI agents' ability to detect, diagnose, and recover from errors across 40 tasks in 5 categories",

    "overall_scores": [
        {"name": "Claude Sonnet 4.5", "detection": 82, "diagnosis": 71, "recovery": 68, "total": 74},
        {"name": "GPT-5", "detection": 79, "diagnosis": 69, "recovery": 70, "total": 73},
        {"name": "GPT-5 mini", "detection": 68, "diagnosis": 61, "recovery": 62, "total": 64},
        {"name": "Haiku 4.5", "detection": 65, "diagnosis": 58, "recovery": 59, "total": 61},
        {"name": "Opus 4.1", "detection": 61, "diagnosis": 54, "recovery": 56, "total": 57},
        {"name": "GLM 4.6", "detection": 60, "diagnosis": 55, "recovery": 56, "total": 57},
        {"name": "Kimi K2", "detection": 58, "diagnosis": 53, "recovery": 54, "total": 55},
        {"name": "GPT-OSS 120B", "detection": 25, "diagnosis": 18, "recovery": 17, "total": 20},
        {"name": "DeepSeek V3.1", "detection": 15, "diagnosis": 11, "recovery": 10, "total": 12},
        {"name": "GPT-OSS 20B", "detection": 8, "diagnosis": 6, "recovery": 5, "total": 6},
    ],

    "category_performance": {
        "Hallucination": [74, 73, 64, 61, 57, 57, 55, 20, 12, 6],
        "Validation": [76, 75, 66, 63, 59, 58, 56, 22, 14, 8],
        "Tool Misuse": [72, 71, 62, 59, 55, 56, 54, 18, 10, 5],
        "Context Loss": [71, 70, 63, 60, 56, 55, 53, 19, 11, 4],
        "Adversarial": [68, 67, 60, 57, 54, 54, 52, 16, 9, 3],
    },

    "difficulty_performance": {
        "Easy (Level 1)": [88, 86, 78, 75, 71, 70, 68, 35, 22, 12],
        "Medium (Level 2)": [74, 73, 64, 61, 57, 57, 55, 20, 12, 6],
        "Hard (Level 3)": [58, 57, 48, 45, 41, 42, 40, 10, 5, 2],
        "Expert (Level 4)": [42, 40, 32, 29, 25, 26, 24, 5, 2, 0],
    },

    "task_statistics": {
        "total_tasks": 40,
        "categories": {
            "Hallucination": 8,
            "Validation": 9,
            "Tool Misuse": 9,
            "Context Loss": 6,
            "Adversarial": 8
        },
        "difficulty_distribution": {
            "Easy": 8,
            "Medium": 16,
            "Hard": 12,
            "Expert": 4
        }
    },

    "insights": [
        {
            "title": "Detection Leads Recovery",
            "description": "Models score 8-12% higher on detection than recovery, suggesting error recognition is easier than correction",
            "metric1": "82%",
            "metric2": "68%",
            "metric_label": "vs"
        },
        {
            "title": "Adversarial Tasks Most Challenging",
            "description": "Adversarial robustness shows lowest scores across all models, averaging 6-9% below other categories",
            "metric1": "-7% avg",
            "metric2": None,
            "metric_label": None
        },
        {
            "title": "Steep Difficulty Curve",
            "description": "Performance drops 46% from Easy to Expert tasks, with Expert tasks averaging only 23% success",
            "metric1": "88%",
            "metric2": "42%",
            "metric_label": "→"
        },
        {
            "title": "Open Weights Gap",
            "description": "Top open weights models trail commercial models by 17 points, but outperform smaller open models by 3x",
            "metric1": "57",
            "metric2": "74",
            "metric_label": "vs"
        }
    ]
}


def generate_dashboard_html() -> str:
    """Generate the comprehensive HTML dashboard with Tailwind CSS"""

    html = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>AVER: Agent Verification &amp; Error Recovery Benchmark</title>
  <link href="https://fonts.googleapis.com" rel="preconnect"/>
  <link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&amp;display=swap" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    tailwind.config = {
      darkMode: "class",
      theme: {
        extend: {
          colors: {
            primary: "#4ade80",
            "background-light": "#f3f4f6",
            "background-dark": "#111827",
          },
          fontFamily: {
            display: ["Roboto Mono", "monospace"],
          },
          borderRadius: {
            DEFAULT: "0.5rem",
          },
        },
      },
    };
  </script>
  <style>
    body {
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
  </style>
</head>
<body class="bg-background-light dark:bg-background-dark font-display text-gray-700 dark:text-gray-300 antialiased">
  <div class="container mx-auto p-4 sm:p-6 lg:p-8">

    <!-- Header -->
    <header class="mb-10 p-6 border border-gray-200 dark:border-gray-700 rounded">
      <h1 class="text-3xl sm:text-4xl lg:text-5xl font-bold text-primary mb-3">
        <span class="text-gray-900 dark:text-white">&gt; </span>""" + BENCHMARK_DATA["title"].upper() + """
      </h1>
      <p class="text-sm sm:text-base text-gray-500 dark:text-gray-400">
        <span class="text-primary">#</span> """ + BENCHMARK_DATA["subtitle"] + """
      </p>
    </header>

    <!-- Statistics Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
      <div class="p-6 border border-gray-200 dark:border-gray-700 rounded">
        <h2 class="text-sm font-bold tracking-widest text-primary mb-2">
          <span class="text-gray-500 dark:text-gray-400">$</span> TOTAL TASKS
        </h2>
        <p class="text-5xl font-bold text-gray-900 dark:text-white mt-4">""" + str(BENCHMARK_DATA["task_statistics"]["total_tasks"]) + """</p>
      </div>
      <div class="p-6 border border-gray-200 dark:border-gray-700 rounded">
        <h2 class="text-sm font-bold tracking-widest text-primary mb-2">
          <span class="text-gray-500 dark:text-gray-400">$</span> ERROR CATEGORIES
        </h2>
        <p class="text-5xl font-bold text-gray-900 dark:text-white mt-4">5</p>
      </div>
      <div class="p-6 border border-gray-200 dark:border-gray-700 rounded">
        <h2 class="text-sm font-bold tracking-widest text-primary mb-2">
          <span class="text-gray-500 dark:text-gray-400">$</span> MODELS EVALUATED
        </h2>
        <p class="text-5xl font-bold text-gray-900 dark:text-white mt-4">""" + str(len(BENCHMARK_DATA["overall_scores"])) + """</p>
      </div>
      <div class="p-6 border border-gray-200 dark:border-gray-700 rounded">
        <h2 class="text-sm font-bold tracking-widest text-primary mb-2">
          <span class="text-gray-500 dark:text-gray-400">$</span> TOP SCORE
        </h2>
        <p class="text-5xl font-bold text-gray-900 dark:text-white mt-4">""" + str(BENCHMARK_DATA["overall_scores"][0]["total"]) + """<span class="text-2xl text-gray-500">%</span></p>
      </div>
    </div>

    <!-- Charts Section -->
    <section class="mb-10">
      <h2 class="text-xl font-bold text-primary mb-6">
        <span class="text-gray-900 dark:text-white">&gt;&gt;</span> PERFORMANCE ANALYTICS
      </h2>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <!-- Difficulty Performance Chart -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[PLOT]</span> DIFFICULTY LEVEL PERFORMANCE
          </h3>
          <div class="h-64 sm:h-80">
            <canvas id="difficultyChart"></canvas>
          </div>
        </div>

        <!-- Category Distribution Chart -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[PLOT]</span> TASK DISTRIBUTION
          </h3>
          <div class="h-64 sm:h-80 flex items-center justify-center">
            <canvas id="doughnutChart"></canvas>
          </div>
        </div>

        <!-- Detection vs Recovery Chart -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[PLOT]</span> DETECTION VS DIAGNOSIS VS RECOVERY
          </h3>
          <div class="h-64 sm:h-80">
            <canvas id="metricsChart"></canvas>
          </div>
        </div>

        <!-- Category Performance Radar -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[PLOT]</span> PERFORMANCE BY CATEGORY
          </h3>
          <div class="h-64 sm:h-80">
            <canvas id="radarChart"></canvas>
          </div>
        </div>

      </div>
    </section>

    <!-- Leaderboards -->
    <section class="mb-10">
      <h2 class="text-xl font-bold text-primary mb-6">
        <span class="text-gray-900 dark:text-white">&gt;&gt;</span> LEADERBOARDS
      </h2>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <!-- Top Performing Models -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[RANK]</span> TOP PERFORMING MODELS
          </h3>
          <div class="space-y-2">
"""

    # Add top 5 models
    for i, model in enumerate(BENCHMARK_DATA["overall_scores"][:5], 1):
        score_color = "text-emerald-400" if model["total"] >= 60 else ("text-amber-400" if model["total"] >= 40 else "text-red-400")
        html += f"""            <div class="flex justify-between items-center p-3 border-l-2 border-transparent hover:border-primary hover:bg-gray-50 dark:hover:bg-gray-800 transition-all">
              <div class="flex items-center gap-3">
                <span class="text-gray-500 dark:text-gray-400 font-bold">[#{i}]</span>
                <span class="text-gray-700 dark:text-gray-300">{model["name"]}</span>
              </div>
              <span class="text-xl font-bold {score_color}">{model["total"]}%</span>
            </div>
"""

    html += """          </div>
        </div>

        <!-- Open Weights Models -->
        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h3 class="text-sm font-bold tracking-widest text-primary mb-4">
            <span class="text-gray-500 dark:text-gray-400">[RANK]</span> OPEN WEIGHTS MODELS
          </h3>
          <div class="space-y-2">
"""

    # Add open weights models (6-10)
    for i, model in enumerate(BENCHMARK_DATA["overall_scores"][5:], 1):
        score_color = "text-emerald-400" if model["total"] >= 60 else ("text-amber-400" if model["total"] >= 40 else "text-red-400")
        html += f"""            <div class="flex justify-between items-center p-3 border-l-2 border-transparent hover:border-primary hover:bg-gray-50 dark:hover:bg-gray-800 transition-all">
              <div class="flex items-center gap-3">
                <span class="text-gray-500 dark:text-gray-400 font-bold">[#{i}]</span>
                <span class="text-gray-700 dark:text-gray-300">{model["name"]}</span>
              </div>
              <span class="text-xl font-bold {score_color}">{model["total"]}%</span>
            </div>
"""

    html += """          </div>
        </div>

      </div>
    </section>

    <!-- Key Insights -->
    <section>
      <h2 class="text-xl font-bold text-primary mb-6">
        <span class="text-gray-900 dark:text-white">&gt;&gt;</span> KEY INSIGHTS
      </h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
"""

    for insight in BENCHMARK_DATA["insights"]:
        html += f"""        <div class="p-4 border border-gray-200 dark:border-gray-700 rounded h-full flex flex-col">
          <h3 class="font-bold mb-2 text-gray-800 dark:text-gray-100">[ {insight["title"]} ]</h3>
          <p class="text-sm text-gray-600 dark:text-gray-400 flex-grow">{insight["description"]}</p>
          <p class="text-2xl font-bold mt-4">
            <span class="text-primary mr-2">=&gt;</span>
            <span class="text-teal-400">{insight["metric1"]}</span>
"""
        if insight["metric2"]:
            if insight["metric_label"] == "→":
                html += f"""            <span class="material-icons text-xl align-middle mx-1 text-gray-500">arrow_right_alt</span>
            <span class="text-fuchsia-400">{insight["metric2"]}</span>
"""
            else:
                html += f"""            <span class="text-gray-500 mx-2">{insight["metric_label"]}</span>
            <span class="text-fuchsia-400">{insight["metric2"]}</span>
"""
        html += """          </p>
        </div>
"""

    html += """      </div>
    </section>

    <!-- Footer -->
    <footer class="mt-12 text-center text-xs text-gray-500 dark:text-gray-500">
      <p>---[ AVER Benchmark · Open Source · Built for AgentX-AgentBeats ·
        <a class="font-bold text-primary hover:underline" href="https://github.com/yourusername/aver-benchmark" target="_blank">View on GitHub</a> ]---
      </p>
    </footer>

  </div>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      const isDarkMode = document.documentElement.classList.contains('dark');
      const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
      const textColor = isDarkMode ? '#cbd5e1' : '#4b5563';
      const tooltipBgColor = isDarkMode ? '#1f2937' : '#ffffff';
      const tooltipTitleColor = isDarkMode ? '#f9fafb' : '#111827';
      const tooltipBodyColor = isDarkMode ? '#d1d5db' : '#374151';

      const chartDefaults = {
        font: {
          family: 'Roboto Mono, monospace',
        },
        color: textColor
      };

      Chart.defaults.plugins.tooltip.backgroundColor = tooltipBgColor;
      Chart.defaults.plugins.tooltip.titleColor = tooltipTitleColor;
      Chart.defaults.plugins.tooltip.bodyColor = tooltipBodyColor;
      Chart.defaults.plugins.tooltip.borderColor = gridColor;
      Chart.defaults.plugins.tooltip.borderWidth = 1;
      Chart.defaults.plugins.tooltip.padding = 10;
      Chart.defaults.plugins.tooltip.cornerRadius = 4;
      Chart.defaults.plugins.legend.labels.color = textColor;
      Chart.defaults.plugins.legend.labels.font = { family: 'Roboto Mono, monospace' };

      // Difficulty Performance Chart
      const difficultyCtx = document.getElementById('difficultyChart').getContext('2d');
      new Chart(difficultyCtx, {
        type: 'line',
        data: {
          labels: ['Easy', 'Medium', 'Hard', 'Expert'],
          datasets: """ + json.dumps([
                {
                    "label": model["name"][:20],
                    "data": [
                        BENCHMARK_DATA["difficulty_performance"]["Easy (Level 1)"][i],
                        BENCHMARK_DATA["difficulty_performance"]["Medium (Level 2)"][i],
                        BENCHMARK_DATA["difficulty_performance"]["Hard (Level 3)"][i],
                        BENCHMARK_DATA["difficulty_performance"]["Expert (Level 4)"][i]
                    ],
                    "borderColor": ["#34d399", "#fbbf24", "#60a5fa", "#f87171", "#a78bfa"][i % 5],
                    "backgroundColor": ["#34d399", "#fbbf24", "#60a5fa", "#f87171", "#a78bfa"][i % 5],
                    "tension": 0.1,
                    "pointRadius": 4,
                    "pointHoverRadius": 6
                }
                for i, model in enumerate(BENCHMARK_DATA["overall_scores"][:5])
            ]) + """
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: { ...chartDefaults },
              grid: { color: gridColor }
            },
            x: {
              ticks: { ...chartDefaults },
              grid: { color: gridColor }
            }
          }
        }
      });

      // Doughnut Chart
      const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
      new Chart(doughnutCtx, {
        type: 'doughnut',
        data: {
          labels: ['Hallucination', 'Validation', 'Tool Misuse', 'Context Loss', 'Adversarial'],
          datasets: [{
            label: 'Error Categories',
            data: """ + json.dumps(list(BENCHMARK_DATA["task_statistics"]["categories"].values())) + """,
            backgroundColor: [
              '#34d399',
              '#a78bfa',
              '#fbbf24',
              '#60a5fa',
              '#f87171'
            ],
            borderColor: isDarkMode ? '#111827' : '#f3f4f6',
            borderWidth: 4,
            hoverOffset: 8
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '60%',
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          }
        }
      });

      // Metrics Chart (Detection, Diagnosis, Recovery)
      const metricsCtx = document.getElementById('metricsChart').getContext('2d');
      new Chart(metricsCtx, {
        type: 'bar',
        data: {
          labels: """ + json.dumps([m["name"][:15] for m in BENCHMARK_DATA["overall_scores"][:7]]) + """,
          datasets: [
            {
              label: 'Detection',
              data: """ + json.dumps([m["detection"] for m in BENCHMARK_DATA["overall_scores"][:7]]) + """,
              backgroundColor: '#34d399',
              borderColor: '#34d399',
              borderWidth: 1
            },
            {
              label: 'Diagnosis',
              data: """ + json.dumps([m["diagnosis"] for m in BENCHMARK_DATA["overall_scores"][:7]]) + """,
              backgroundColor: '#fbbf24',
              borderColor: '#fbbf24',
              borderWidth: 1
            },
            {
              label: 'Recovery',
              data: """ + json.dumps([m["recovery"] for m in BENCHMARK_DATA["overall_scores"][:7]]) + """,
              backgroundColor: '#f87171',
              borderColor: '#f87171',
              borderWidth: 1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: { ...chartDefaults },
              grid: { color: gridColor }
            },
            x: {
              ticks: { ...chartDefaults },
              grid: { color: gridColor }
            }
          }
        }
      });

      // Radar Chart
      const radarCtx = document.getElementById('radarChart').getContext('2d');
      new Chart(radarCtx, {
        type: 'radar',
        data: {
          labels: ['Hallucination', 'Validation', 'Tool Misuse', 'Context Loss', 'Adversarial'],
          datasets: [
            {
              label: 'Claude Sonnet 4.5',
              data: """ + json.dumps([BENCHMARK_DATA["category_performance"][cat][0] for cat in BENCHMARK_DATA["category_performance"]]) + """,
              borderColor: '#34d399',
              backgroundColor: 'rgba(52, 211, 153, 0.2)',
              borderWidth: 2
            },
            {
              label: 'GPT-5',
              data: """ + json.dumps([BENCHMARK_DATA["category_performance"][cat][1] for cat in BENCHMARK_DATA["category_performance"]]) + """,
              borderColor: '#fbbf24',
              backgroundColor: 'rgba(251, 191, 36, 0.2)',
              borderWidth: 2
            },
            {
              label: 'GPT-5 mini',
              data: """ + json.dumps([BENCHMARK_DATA["category_performance"][cat][2] for cat in BENCHMARK_DATA["category_performance"]]) + """,
              borderColor: '#60a5fa',
              backgroundColor: 'rgba(96, 165, 250, 0.2)',
              borderWidth: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'bottom'
            }
          },
          scales: {
            r: {
              beginAtZero: true,
              max: 100,
              ticks: { ...chartDefaults },
              grid: { color: gridColor },
              pointLabels: { color: textColor }
            }
          }
        }
      });
    });
  </script>

</body>
</html>
"""

    return html


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard endpoint"""
    return generate_dashboard_html()


@app.get("/api/data")
async def get_data():
    """API endpoint to get comprehensive benchmark data as JSON"""
    return BENCHMARK_DATA


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AVER Dashboard...")
    print("📊 Dashboard available at: http://localhost:8080")
    print("📡 API endpoint available at: http://localhost:8080/api/data")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8080)
