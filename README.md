# CATA Analysis Tool

Python tool for analyzing Check-All-That-Apply (CATA) data with multiple statistical approaches.

## Features

- Frequency analysis of CATA attributes
- Cochran's Q tests for attribute differences
- Penalty analysis for liking scores
- Correspondence analysis visualization
- PCA biplot visualization
- Ideal product comparison

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/daidzein/cata-analysis-tool-py.git
   cd cata-analysis-tool-py
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

Place your CATA data in CSV format in the data/ folder, then run:
```bash
python src/cata_analysis.py
```

## Data Format
Your CSV should have columns:
- Consumer: Participant ID
- Product: Product Name
- Liking: Hedonic score (numeric)
- [Attribute 1], [Attribute 2], ...: Boolean columns for CATA Attributes
