#!/bin/bash
# Run the AVER Dashboard

echo "🚀 Starting AVER Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8080"
echo ""

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "aver-venv/bin" ]; then
    source aver-venv/bin/activate
fi

# Run the dashboard
python src/aver/dashboard.py
