#!/bin/bash
# CHISEL Setup Script

echo "Setting up CHISEL..."

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "2. Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "3. Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "4. Downloading NLTK data..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download Spider dataset from https://yale-lily.github.io/spider"
echo "2. Extract to data/spider/"
echo "3. Activate venv: source venv/bin/activate"
echo "4. Run: python -m chisel.pipeline"
