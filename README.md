# DeepMGO Application Web Tool

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
git clone https://github.com/labnams/DeepMGO.git
cd DeepMGO
```

2. Install using pip:
```bash
pip install -r requirements.txt
```

## Required Directory Structure

```
DeepMGO/
├── DeepMGO_shiny.py
├── data/
│   └── descriptors/
│       ├── padel_columns.csv
│       └── feature_indices.csv
│   └── preprocessing/
│       └── minmax_scaler.pkl
└── Model/
    ├── DeepMGO.json
    └── DeepMGO.h5 
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
[DeepMGO reference information]

## License

MIT License
Copyright (c) 2024 LabNams

## Contact
### If you have any questions, please contact below.
- Dr. Aron Park (parkar13@gmail.com)
- Prof. Seungyoon Nam (nams@gachon.ac.kr)


Dr. Aron Park (parkar13@gmail.com)
Prof. Seungyoon Nam (seungyoon.nam@gmail.com)

