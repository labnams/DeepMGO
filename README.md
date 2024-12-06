# DeepMGO Application Tool

A web-based tool for predicting MGO affinity using deep learning.

## Features

- Predict MGO affinity from SMILES strings
- Process multiple compounds simultaneously
- Interactive visualization of results
- Export results in CSV format
- Download prediction plots as SVG

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepMGO.git
cd DeepMGO
```

2. Create environment using conda:
```bash
conda env create -f environment.yml
conda activate deepmgo
```

Or install using pip:
```bash
pip install -r requirements.txt
```

## Required Directory Structure

```
DeepMGO/
├── data/
│   └── descriptors/
│       ├── padel_columns.csv
│       └── feature_indices.csv
├── Model/
│   ├── DeepMGO_fs90.json
│   └── DeepMGO_fs90.h5
└── scaler/
    └── minmax_scaler_add_68.pkl
```

## Usage

Run the application:
```bash
python DeepMGO_shiny.py
```

The web interface will be available at `http://localhost:8000` by default.

## Input Format

- SMILES strings: One per line (comments after # are ignored)
- Concentrations: One per line (in µM)

## Citation

If you use this tool in your research, please cite:
[Add your citation information here]

## License

MIT License

Copyright (c) 2024 LabNams

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
