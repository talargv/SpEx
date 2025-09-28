#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Install ExKMC ---
echo "Cloning and installing ExKMC..."
git clone https://github.com/navefr/ExKMC.git
cd ExKMC
python setup.py build_ext --inplace --cython
pip install .
cd ..
rm -rf ExKMC
echo "ExKMC installed successfully."

# --- Install CLIP ---
echo "Installing CLIP from OpenAI..."
pip install git+https://github.com/openai/CLIP.git
echo "CLIP installed successfully."

echo "All external dependencies installed."