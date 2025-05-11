# Tautomer Enumerator

A Streamlit web application for enumerating, visualizing, and calculating energies of molecular tautomers using RDKit.

## Overview

This application provides a powerful tool for chemists and computational researchers to explore tautomeric forms of molecules. It uses RDKit's tautomer enumeration functionality, enhanced with custom processing to ensure chemically reasonable structures and reliable energy calculations.

![Tautomer Enumerator Screenshot](https://github.com/your-username/tautomer-enumerator/raw/main/screenshot.png)

## Features

- **Tautomer Enumeration**: Generate all possible tautomers for a given molecule
- **Chemical Reasonability Rules**: Apply pattern-based and SMARTS-based cleanup to ensure valid chemistry
- **Energy Calculation**: Calculate MMFF energies for compatible tautomers
- **Interactive Visualization**: View 2D structures of all tautomers
- **3D Visualization**: Explore 3D conformations with py3DMol integration
- **Energy Comparison**: Compare relative energies of tautomers with sortable tables and charts

## How It Works

The application follows a sophisticated pipeline:

1. **Tautomer Generation**: Uses RDKit's TautomerEnumerator to generate all possible tautomers
2. **Display Preparation**: Ensures all tautomers can be visualized regardless of chemical validity
3. **Chemical Cleanup**: Applies both SMILES pattern replacements and SMARTS-based structural fixes
4. **Energy Calculation**: Identifies which tautomers can have valid energy calculations
5. **Visualization**: Displays all tautomers with their SMILES and energy values where available

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/tautomer-enumerator.git
cd tautomer-enumerator

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- Python 3.7+
- RDKit
- Streamlit
- NumPy
- Pandas
- PIL

Full requirements are listed in `requirements.txt`.

## Usage

Run the Streamlit app:

```bash
streamlit run tautomer-enumerator.py
```

Then open your web browser to the URL shown (typically http://localhost:8501).

### Input Options

1. Enter a SMILES string in the text input
2. Adjust maximum number of tautomers to generate (slider)
3. Toggle 3D structure visualization
4. Toggle energy calculation

### Example SMILES

The app comes with several example SMILES strings to try:
- Pyrazolone: `O=c1cc[nH][nH]1`
- Diketone: `CC(=O)C(=O)CC`
- Azoxy compound: `c1ccc2c(c1)C(=NN=C2O)N`
- Acetaldehyde: `CC=O`

## Troubleshooting

If tautomer generation or energy calculation fails:
1. Try simplifying your input molecule
2. Try different SMILES representations of the same molecule
3. Adjust the maximum number of tautomers

## Technical Details

### Chemical Reasonability Rules

The app applies several cleanup rules to ensure chemically valid tautomers:

1. **SMILES Pattern-Based Cleanup**:
   - `C(=O[H])` → `C(=O)` (Remove H from carbonyl oxygen)
   - `CO([H])[H]` → `CO[H]` (Remove extra H from hydroxyl)
   - `CN([H])([H])[H]` → `CN([H])[H]` (Remove extra H from amine)
   - And many others

2. **SMARTS-Based Structural Cleanup**:
   - `[N;X4;H3;v4]` - Nitrogen with 4 connections, 3 hydrogens, valence 4
   - `[O;X3;H2;v3]` - Oxygen with 3 connections, 2 hydrogens, valence 3
   - `[O;X3;H2]-[C,c]` - Oxygen with 2 H attached to carbon
   - `[O;X2;H1]=[C,c]` - OH attached to C by double bond

### Energy Calculation

The app uses RDKit's MMFF implementation for energy calculations, with special handling:
- Generates 3D conformations for energy calculation
- Tests energy calculation capability for each tautomer
- Presents relative energies for valid tautomers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RDKit team for their excellent cheminformatics toolkit
- Streamlit for the interactive web framework
- py3DMol for 3D visualization capabilities
