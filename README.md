<div align="center">
  <img src="https://github.com/lulelaboratory/MD_Fragment_Analyzer/blob/main/icon/MD_Frag.png" alt="MD Fragment Analyzer Icon" width="150" style="margin-bottom: 20px;" />
  <h1>MD Fragment Analyzer</h1>
  <p><em>Fragment analysis for molecular dynamics simulations</em></p>
</div>

## Introduction

**MD Fragment Analyzer** is a state-of-the-art Python tool that streamlines the analysis of molecular dynamics (MD) simulations. By automatically identifying chemically connected fragments across simulation frames, it provides clear insights into the structural evolution of your system. Whether you're studying protein dynamics or material science, this tool offers the perfect blend of efficiency, flexibility, and user-friendly interaction.

---

## Overview

Fragment analysis is essential for understanding the structural evolution in MD simulations. This tool computes interatomic distances and determines bonds based on a threshold defined as a scaling factor (default: 1.2) times the sum of atomic radii. Connected atoms are then grouped into fragments using an efficient union-find (disjoint-set) algorithm.

---

## Key Features

- **Fragment Detection**
  - Computes interatomic distances from MD simulation data.
  - Detects bonds using a customizable scale factor and selected atomic radii.

- **Radii Options**
  - Choose between **Covalent Radii** and **Van der Waals Radii** for bond determination.

- **Periodic Boundary Conditions (PBC)**
  - Optionally adjusts interatomic distances to account for periodic boundary conditions using simulation cell dimensions.

- **Performance Optimizations**
  - Accelerated computations with Numba’s just-in-time (JIT) compilation.
  - Multiprocessing support for handling multiple frames.
  - Optional GPU acceleration (in selected versions) for processing large datasets.

- **User-Friendly Interface**
  - A Tkinter-based graphical user interface (GUI) that simplifies file selection, parameter configuration, and provides real-time status updates.

---

## Methodology (v1.4)

1. **Input Processing**  
   Reads MD trajectories from XYZ files and extracts lattice parameters (if available) for periodic systems.

2. **Distance Calculation**  
   Computes the Euclidean distance between every pair of atoms. When PBC is enabled, distances are adjusted according to the simulation cell dimensions.

3. **Bond Determination**  
   A bond is determined if the interatomic distance is less than:

cutoff = scale * (atomic_radius_atom1 + atomic_radius_atom2)


4. **Fragment Grouping**  
Uses a union-find algorithm to cluster bonded atoms into distinct fragments.

5. **Output Generation**  
Summarizes each fragment by counting constituent elements and formatting the results as a molecular formula.

---

## Version Features and Changes

- **v1.0**
- *Initial Release:* Basic fragment analysis using ASE for XYZ file reading and union-find for fragment grouping.
- Provided both CPU (via Numba JIT) and GPU (via Numba CUDA) acceleration.

- **v1.1**
- *GUI Enhancements:* Improved file selection and parameter configuration interface.
- Introduced multiprocessing support for handling multiple frames.

- **v1.2**
- *Radii Options & PBC:* Added support for Van der Waals radii and implemented periodic boundary conditions in distance calculations.

- **v1.3**
- *User Interface Upgrade:* Enhanced GUI with a menu bar (including “About” and “Methodology”) and a detailed status message area.
- Improved fragment grouping and error handling.

- **v1.4**
- *Streamlined Methodology:* Focused on stable CPU-based computations using essential submodules for better performance and accuracy.

- **v1.5**
- *Expanded Support:* Integrated a custom `Atoms` class and XYZ reader to remove the dependency on ASE.
- Enhanced multiprocessing and optional GPU acceleration to better handle large datasets.

---

## Requirements

- **Python 3.x**
- [NumPy](https://numpy.org/)
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) *(v1.5 now uses a custom reader)*
- [Numba](http://numba.pydata.org/)
- **Tkinter** (usually bundled with Python)
- *(Optional)* CUDA-compatible GPU for accelerated computations

---

## Installation

 git clone https://github.com/yourusername/MD-Fragment-Analyzer.git
 cd MD-Fragment-Analyzer
 pip install -r requirements.txt

---

## Usage

 python MD_FA_v1.4.py

For advanced batch processing or to leverage the latest features, consider using:

 python MD_FA_v1.5.py

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Developed by Le Lu ([lulelaboratory@gmail.com](mailto:lulelaboratory@gmail.com)) in 2025. For further details or support, please contact the author.
