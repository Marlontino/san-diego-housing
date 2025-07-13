#!/bin/bash

echo "🏠 Setting up San Diego Housing Market Analysis Project"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download the data files and place them in the data/ directory:"
echo "   - listings.csv.gz"
echo "   - calendar.csv.gz" 
echo "   - reviews.csv.gz"
echo "   - neighbourhoods.csv"
echo "   - neighbourhoods.geojson"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Test the setup:"
echo "   python quick_start.py"
echo ""
echo "4. Start Jupyter notebook:"
echo "   jupyter notebook"
echo ""
echo "5. Open notebooks/01_data_exploration.ipynb" 