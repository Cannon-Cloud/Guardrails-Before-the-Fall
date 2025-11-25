#!/bin/bash
# Setup script for Guardrails Before the Fall repository
# Author: Clarence H. Cannon, IV

set -e

echo "=========================================="
echo "Guardrails Before the Fall - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo -e "${RED}Error: Python 3.10+ required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p paper/figures
mkdir -p experiments/exp_a_jailbreak/results
mkdir -p experiments/exp_b_drift/results
mkdir -p experiments/exp_c_efficiency/results
mkdir -p experiments/exp_d_compositional/results
mkdir -p experiments/exp_d_compositional/sandbox_data
mkdir -p experiments/exp_d_compositional/mock_website
mkdir -p experiments/exp_d_compositional/logs
mkdir -p data
touch data/.gitkeep
echo -e "${GREEN}✓ Directories created${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate and install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null 2>&1 || {
    echo -e "${YELLOW}Some optional dependencies may not have installed. Core functionality should work.${NC}"
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check for .env file
echo -e "${YELLOW}Checking environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ Created .env from template. Please edit with your API keys.${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Check for LaTeX (optional)
echo -e "${YELLOW}Checking LaTeX installation (optional)...${NC}"
if command -v pdflatex &> /dev/null; then
    echo -e "${GREEN}✓ LaTeX found (pdflatex available)${NC}"
else
    echo -e "${YELLOW}⚠ LaTeX not found. Use Overleaf for PDF generation.${NC}"
fi

# Check for Docker (optional, for Experiment D)
echo -e "${YELLOW}Checking Docker installation (optional)...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
else
    echo -e "${YELLOW}⚠ Docker not found. Experiment D sandbox will run in simulation mode.${NC}"
fi

# Make experiment scripts executable
echo -e "${YELLOW}Setting permissions...${NC}"
chmod +x experiments/exp_a_jailbreak/run.py 2>/dev/null || true
chmod +x experiments/exp_b_drift/run.py 2>/dev/null || true
chmod +x experiments/exp_c_efficiency/run.py 2>/dev/null || true
chmod +x experiments/exp_d_compositional/run.py 2>/dev/null || true
echo -e "${GREEN}✓ Permissions set${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys:"
echo "     - OPENAI_API_KEY"
echo "     - ANTHROPIC_API_KEY"
echo ""
echo "  2. Build the PDF:"
echo "     make pdf"
echo "     # Or upload paper/ folder to Overleaf"
echo ""
echo "  3. Run experiments:"
echo "     source venv/bin/activate"
echo "     make experiments"
echo ""
echo "  4. Run individual experiment:"
echo "     python experiments/exp_a_jailbreak/run.py"
echo ""
echo "For help: make help"
echo ""