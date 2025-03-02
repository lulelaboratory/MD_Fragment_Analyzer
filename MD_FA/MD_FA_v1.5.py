#!/usr/bin/env python3
import os
import csv
import time
from math import sqrt
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
from multiprocessing import Pool

import numpy as np
# Numba imports
from numba import jit, cuda, float64, int32

# Check for CUDA availability
gpu_available = cuda.is_available()
if not gpu_available:
    print("CUDA not available. GPU acceleration will be disabled.")

# --------------------------
# Preloaded atomic data for the entire periodic table (elements 1 to 118)
# Atomic numbers mapping
atomic_numbers = {
    "H": 1,  "He": 2, "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,   "F": 9,   "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,  "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53,  "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,  "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101,"No": 102,"Lr": 103,"Rf": 104,"Db": 105,"Sg": 106,"Bh": 107,"Hs": 108,"Mt": 109,"Ds": 110,
    "Rg": 111,"Cn": 112,"Nh": 113,"Fl": 114,"Mc": 115,"Lv": 116,"Ts": 117,"Og": 118
}

# Covalent radii in Å (approximate; based on Pyykkö & Atsumi, 2009) :contentReference[oaicite:0]{index=0}
covalent_radii = {
     1: 0.31,  2: 0.28,  3: 1.28,  4: 0.96,  5: 0.84,  6: 0.76,  7: 0.71,  8: 0.66,  9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.16, 38: 1.92, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.33, 54: 1.31, 55: 2.35, 56: 1.98, 57: 1.87, 58: 1.82, 59: 1.80, 60: 1.80,
    61: 1.80, 62: 1.80, 63: 1.80, 64: 1.78, 65: 1.75, 66: 1.75, 67: 1.75, 68: 1.75, 69: 1.75, 70: 1.74,
    71: 1.73, 72: 1.66, 73: 1.51, 74: 1.46, 75: 1.42, 76: 1.40, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32,
    81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50, 87: 2.50, 88: 2.21, 89: 2.15, 90: 2.06,
    91: 2.00, 92: 1.96, 93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69, 97: 1.60, 98: 1.60, 99: 1.60,100: 1.60,
   101: 1.60,102: 1.60,103: 1.60,104: 1.57,105: 1.50,106: 1.50,107: 1.50,108: 1.50,109: 1.50,
   110: 1.50,111: 1.50,112: 1.50,113: 1.50,114: 1.50,115: 1.50,116: 1.50,117: 1.50,118: 1.50
}

# Van der Waals radii in Å (approximate; based on Bondi’s data) :contentReference[oaicite:1]{index=1}
vdw_radii = {
     1: 1.20,  2: 1.40,  3: 1.82,  4: 1.53,  5: 1.92,  6: 1.70,  7: 1.55,  8: 1.52,  9: 1.47, 10: 1.54,
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88, 19: 2.75, 20: 2.31,
    21: 2.11, 22: 2.00, 23: 1.97, 24: 1.92, 25: 1.91, 26: 1.92, 27: 1.89, 28: 1.91, 29: 1.80, 30: 1.83,
    31: 1.87, 32: 2.11, 33: 1.85, 34: 1.90, 35: 1.98, 36: 2.02, 37: 3.03, 38: 2.49, 39: 2.27, 40: 2.13,
    41: 2.03, 42: 2.01, 43: 2.00, 44: 2.00, 45: 2.00, 46: 1.63, 47: 1.72, 48: 1.58, 49: 1.93, 50: 2.17,
    51: 2.06, 52: 2.06, 53: 1.98, 54: 2.16, 55: 3.43, 56: 2.68, 57: 2.50, 58: 2.48, 59: 2.47, 60: 2.45,
    61: 2.43, 62: 2.42, 63: 2.40, 64: 2.38, 65: 2.37, 66: 2.35, 67: 2.33, 68: 2.32, 69: 2.30, 70: 2.28,
    71: 2.27, 72: 2.25, 73: 2.20, 74: 2.10, 75: 2.05, 76: 2.00, 77: 2.00, 78: 1.75, 79: 1.66, 80: 1.55,
    81: 1.96, 82: 2.02, 83: 2.07, 84: 1.97, 85: 2.02, 86: 2.20, 87: 3.48, 88: 2.83, 89: 2.70, 90: 2.60,
    91: 2.50, 92: 2.40, 93: 2.30, 94: 2.20, 95: 2.10, 96: 2.00, 97: 1.90, 98: 1.90, 99: 1.90,100: 1.90,
   101: 1.90,102: 1.90,103: 1.90,104: 1.80,105: 1.80,106: 1.80,107: 1.80,108: 1.80,109: 1.80,
   110: 1.80,111: 1.80,112: 1.80,113: 1.80,114: 1.80,115: 1.80,116: 1.80,117: 1.80,118: 1.80
}

# --------------------------
# Minimal Atoms class and XYZ reader (replacing ASE)
# --------------------------
class Atoms:
    def __init__(self, symbols, positions, cell=None):
        self.symbols = symbols
        self.positions = positions  # numpy array (n, 3)
        self.cell = cell  # If provided, should be a 3x3 numpy array

    def get_positions(self):
        return self.positions

    def get_chemical_symbols(self):
        return self.symbols

    def get_cell(self):
        if self.cell is None:
            return np.zeros((3, 3), dtype=np.float64)
        return self.cell

def read_xyz(file_path, index=":"):
    """
    Reads an XYZ file and returns a list of Atoms objects (one per frame).
    If the comment line contains a 'Lattice=' string, the cell dimensions are extracted.
    """
    frames = []
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            comment = f.readline().strip()
            # Try to extract cell info from the comment if present
            cell = None
            if "Lattice=" in comment:
                start = comment.find("Lattice=")
                quote_start = comment.find('"', start)
                quote_end = comment.find('"', quote_start + 1)
                if quote_start != -1 and quote_end != -1:
                    lattice_str = comment[quote_start+1:quote_end]
                    try:
                        numbers = list(map(float, lattice_str.split()))
                        if len(numbers) >= 9:
                            cell = np.array(numbers[:9]).reshape((3, 3))
                    except Exception:
                        cell = None
            symbols = []
            positions = []
            for i in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    continue
                symbols.append(parts[0])
                positions.append(list(map(float, parts[1:4])))
            frames.append(Atoms(symbols, np.array(positions, dtype=np.float64), cell=cell))
    return frames if index == ":" else frames[int(index)]

# --------------------------
# Union-Find (disjoint-set) algorithm
# --------------------------
def union_find(n, pairs):
    parent = list(range(n))
    rank = [0] * n

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1

    for i, j in pairs:
        union(i, j)

    groups = {}
    for i in range(n):
        rep = find(i)
        groups.setdefault(rep, []).append(i)
    return list(groups.values())

# --------------------------
# Compute fragments for one frame (CPU - Numba JIT compiled)
# --------------------------
@jit(nopython=True)
def compute_fragments_cpu(positions, radii, scale, n, periodic, cell):
    bonds = []
    if periodic:
        Lx = cell[0, 0]
        Ly = cell[1, 1]
        Lz = cell[2, 2]
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            if periodic:
                dx -= round(dx / Lx) * Lx
                dy -= round(dy / Ly) * Ly
                dz -= round(dz / Lz) * Lz
            r = sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = scale * (radii[i] + radii[j])
            if r < cutoff:
                bonds.append((i, j))
    return bonds

# --------------------------
# Compute fragments for one frame (GPU accelerated - Numba)
# --------------------------
@cuda.jit
def compute_fragments_gpu(positions, radii, scale, bond_matrix):
    x = cuda.grid(1)
    n = positions.shape[0]
    if x < n:
        for y in range(x + 1, n):
            dx = positions[x, 0] - positions[y, 0]
            dy = positions[x, 1] - positions[y, 1]
            dz = positions[x, 2] - positions[y, 2]
            r = sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = scale * (radii[x] + radii[y])
            if r < cutoff:
                bond_matrix[x, y] = 1

# --------------------------
# Compute fragments (dispatcher)
# --------------------------
def compute_fragments(atoms, scale=1.2, use_gpu=False, periodic=False, radius_type="covalent"):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    if radius_type.lower() == "vdw":
        radii = np.array([vdw_radii[atomic_numbers[s]] for s in symbols], dtype=np.float64)
    else:
        radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols], dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)

    # Check cell dimensions for periodic conditions
    if periodic:
        cell = np.array(atoms.get_cell(), dtype=np.float64)
        if cell[0, 0] == 0 or cell[1, 1] == 0 or cell[2, 2] == 0:
            print("Warning: Periodic boundary conditions enabled but cell dimensions are zero. Falling back to non-periodic analysis.")
            periodic = False

    if use_gpu and periodic:
        print("GPU periodic analysis not implemented. Falling back to CPU.")
        use_gpu = False

    if use_gpu:
        d_positions = cuda.to_device(positions)
        d_radii = cuda.to_device(radii)
        bond_matrix = cuda.device_array((n, n), dtype=np.int32)
        threadsperblock = 256
        blockspergrid = (n + threadsperblock - 1) // threadsperblock
        compute_fragments_gpu[blockspergrid, threadsperblock](d_positions, d_radii, scale, bond_matrix)
        cuda.synchronize()
        bond_matrix_cpu = bond_matrix.copy_to_host()
        bonds = []
        for i in range(n):
            for j in range(i + 1, n):
                if bond_matrix_cpu[i, j] == 1:
                    bonds.append((i, j))
    else:
        cell = np.array(atoms.get_cell(), dtype=np.float64) if periodic else np.zeros((3, 3), dtype=np.float64)
        bonds = compute_fragments_cpu(positions, radii, scale, n, periodic, cell)

    groups = union_find(n, bonds)
    fragments = []
    for group in groups:
        comp = Counter([symbols[i] for i in group])
        formula = "".join(f"{el}{comp[el] if comp[el] > 1 else ''}" for el in sorted(comp.keys()))
        fragments.append(formula)
    return fragments

# --------------------------
# Process one frame (for multiprocessing)
# --------------------------
def process_frame(args):
    i, atoms, use_gpu, gpu_id, periodic, radius_type = args
    if use_gpu and gpu_id is not None:
        cuda.select_device(gpu_id)
    frags = compute_fragments(atoms, use_gpu=use_gpu, periodic=periodic, radius_type=radius_type)
    frag_counts = Counter(frags)
    return i, list(frag_counts.items())

# --------------------------
# GUI Class using Tkinter
# --------------------------
class FragmentCounterGUI:
    def __init__(self, master):
        self.master = master
        master.title("MD Fragment Analyzer")
        master.geometry("550x500")
        self.create_menu()

        self.file_path = None
        self.num_cores = tk.IntVar(value=1)
        self.use_gpu = tk.BooleanVar(value=False)
        self.use_pbc = tk.BooleanVar(value=False)
        self.selected_gpu = tk.StringVar(value="All GPUs")
        self.radius_type = tk.StringVar(value="Covalent Radii")
        self.file_var = tk.StringVar(value="")

        main_frame = tk.Frame(master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        headline = tk.Label(main_frame, text="MD Fragment Analyzer", font=("Helvetica", 20, "bold"))
        headline.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        file_frame = tk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        self.load_button = tk.Button(file_frame, text="Load File", command=self.load_file, width=15)
        self.load_button.grid(row=0, column=0, padx=5)
        self.file_entry = tk.Entry(file_frame, textvariable=self.file_var, width=50, state='readonly')
        self.file_entry.grid(row=0, column=1, padx=5, sticky="w")

        core_frame = tk.Frame(main_frame)
        core_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        core_label = tk.Label(core_frame, text="Number of cores:")
        core_label.grid(row=0, column=0, padx=5, sticky="e")
        self.core_entry = tk.Entry(core_frame, textvariable=self.num_cores, width=5)
        self.core_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.core_button = tk.Button(core_frame, text="Set Cores", command=self.validate_cores, width=10)
        self.core_button.grid(row=0, column=2, padx=5)

        gpu_frame = tk.Frame(main_frame)
        gpu_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        self.gpu_check = tk.Checkbutton(gpu_frame, text="Use GPU", variable=self.use_gpu, command=self.check_gpu_availability)
        self.gpu_check.grid(row=0, column=0, padx=5, pady=5)
        if gpu_available:
            num_gpus = len(cuda.gpus)
            gpu_options = ["All GPUs"] + [f"GPU {i}" for i in range(num_gpus)]
        else:
            gpu_options = ["No GPU Available"]
        gpu_label = tk.Label(gpu_frame, text="Select GPU:")
        gpu_label.grid(row=0, column=1, padx=5)
        self.gpu_optionmenu = tk.OptionMenu(gpu_frame, self.selected_gpu, *gpu_options)
        self.gpu_optionmenu.grid(row=0, column=2, padx=5)
        if not gpu_available:
            self.gpu_optionmenu.config(state=tk.DISABLED)
            self.use_gpu.set(False)

        radii_frame = tk.Frame(main_frame)
        radii_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        radii_label = tk.Label(radii_frame, text="Select Radii Type:")
        radii_label.grid(row=0, column=0, padx=5, sticky="e")
        radii_options = ["Covalent Radii", "Van der Waals Radii"]
        self.radii_optionmenu = tk.OptionMenu(radii_frame, self.radius_type, *radii_options)
        self.radii_optionmenu.grid(row=0, column=1, padx=5, sticky="w")

        pbc_frame = tk.Frame(main_frame)
        pbc_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        self.pbc_check = tk.Checkbutton(pbc_frame, text="Enable Periodic Boundary Conditions", variable=self.use_pbc)
        self.pbc_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        self.run_button = tk.Button(button_frame, text="Run Counting", command=self.run_counting, width=20)
        self.run_button.grid(row=0, column=0, padx=10)
        self.close_button = tk.Button(button_frame, text="Close", command=self.master.destroy, width=10)
        self.close_button.grid(row=0, column=1, padx=10)
        self.run_button.config(state=tk.DISABLED)

        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=7, column=0, columnspan=3, pady=10, sticky="nsew")
        status_label = tk.Label(status_frame, text="Status Messages:")
        status_label.pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        main_frame.grid_rowconfigure(7, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def create_menu(self):
        menu_bar = tk.Menu(self.master)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Methodology", command=self.show_methodology)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.master.config(menu=menu_bar)

    def show_about(self):
        about_text = (
            "MD Fragment Analyzer\n\n"
            "Author: Le Lu (lulelaboratory@gmail.com)\n"
            "Copyright (c) 2025 Le Lu\n\n"
            "MIT License\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies\n"
            "of the Software. THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND.\n\n"
            "For full details, see: https://en.wikipedia.org/wiki/MIT_License"
        )
        about_win = tk.Toplevel(self.master)
        about_win.title("About")
        about_win.geometry("600x400")
        txt = tk.Text(about_win, wrap="word")
        txt.insert("1.0", about_text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        btn_close = tk.Button(about_win, text="OK", command=about_win.destroy)
        btn_close.pack(pady=5)

    def show_methodology(self):
        methodology_text = (
            "Methodology:\n\n"
            "This tool analyzes MD trajectories stored in XYZ format by processing each frame to identify fragments—\n"
            "groups of atoms bonded together based on a distance cutoff. It uses a union-find algorithm to group\n"
            "connected atoms. Radii data (covalent or van der Waals) are preloaded for all elements (Z=1–118).\n\n"
            "Optimizations:\n"
            " - Numba JIT for CPU routines\n"
            " - GPU acceleration via Numba CUDA (non-periodic only)\n"
            " - Multiprocessing for frame processing\n"
        )
        meth_win = tk.Toplevel(self.master)
        meth_win.title("Methodology")
        meth_win.geometry("600x400")
        txt = tk.Text(meth_win, wrap="word")
        txt.insert("1.0", methodology_text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        btn_close = tk.Button(meth_win, text="OK", command=meth_win.destroy)
        btn_close.pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select .xyz file", filetypes=[("XYZ files", "*.xyz")]
        )
        if file_path:
            self.file_path = file_path
            self.file_var.set(file_path)
            self.run_button.config(state=tk.NORMAL)
            self.append_status(f"File loaded: {file_path}\n")
        else:
            self.file_var.set("No file loaded")
            self.run_button.config(state=tk.DISABLED)

    def validate_cores(self):
        try:
            cores = int(self.core_entry.get())
            if cores <= 0:
                raise ValueError("Number of cores must be positive.")
            self.num_cores.set(cores)
            messagebox.showinfo("Cores Set", f"Number of cores set to {cores}")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid core number: {e}")
            self.num_cores.set(1)

    def check_gpu_availability(self):
        if self.use_gpu.get():
            if not gpu_available:
                messagebox.showerror("CUDA Not Found", "CUDA is not available. GPU acceleration cannot be enabled.")
                self.use_gpu.set(False)
                return
            try:
                cuda.cudadrv.driver.driver.get_version()
            except cuda.CudaSupportError as e:
                messagebox.showerror("CUDA Error", f"CUDA driver error: {e}.\nGPU acceleration will be disabled.")
                self.use_gpu.set(False)

    def append_status(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def run_counting(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please load a file first.")
            return

        try:
            num_cores = self.num_cores.get()
            if num_cores <= 0:
                raise ValueError("Number of cores must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid core number: {e}")
            return

        selected_radius = self.radius_type.get()
        radius_type = "vdw" if "vdw" in selected_radius.lower() else "covalent"

        start_time = time.time()
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.append_status("Reading trajectory...\n")
        self.master.update_idletasks()
        try:
            frames = read_xyz(self.file_path, index=":")
            self.append_status(f"{len(frames)} frames loaded.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return

        out_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        out_csv = os.path.join(out_dir, f"{base_name}_fragments.csv")
        self.append_status("Processing frames...\n")
        self.master.update_idletasks()

        periodic = self.use_pbc.get()
        args = []
        if self.use_gpu.get():
            selected = self.selected_gpu.get()
            if selected == "All GPUs":
                num_gpus = len(cuda.gpus)
                args = [(i, atoms, True, i % num_gpus, periodic, radius_type) for i, atoms in enumerate(frames)]
            else:
                gpu_id = int(selected.split()[1])
                args = [(i, atoms, True, gpu_id, periodic, radius_type) for i, atoms in enumerate(frames)]
        else:
            args = [(i, atoms, False, None, periodic, radius_type) for i, atoms in enumerate(frames)]

        with open(out_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Fragment_Type", "Count"])

            with Pool(processes=num_cores) as pool:
                for i, frag_counts_list in pool.imap(process_frame, args):
                    for frag, count in frag_counts_list:
                        writer.writerow([i, frag, count])
                    if (i + 1) % 5 == 0:
                        self.append_status(f"Processed frame {i+1} of {len(frames)}\n")
                        self.master.update_idletasks()
                pool.close()
                pool.join()

        total_time = time.time() - start_time
        self.append_status(f"Done! CSV saved to: {out_csv}\n")
        self.append_status(f"Total time used: {total_time:.2f} seconds\n")
        messagebox.showinfo("Done", f"Fragment counting complete.\nResults saved to:\n{out_csv}\nTotal time used: {total_time:.2f} seconds")

# --------------------------
# Run the GUI
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = FragmentCounterGUI(root)
    root.mainloop()
