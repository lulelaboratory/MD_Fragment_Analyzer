#!/usr/bin/env python3
import os
import csv
import time
from math import sqrt
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox
from multiprocessing import Pool

import numpy as np
from ase.io import read
from ase.data import covalent_radii

# Numba imports
from numba import jit, cuda, float64, int32

# Check for CUDA availability
gpu_available = cuda.is_available()
if not gpu_available:
    print("CUDA not available. GPU acceleration will be disabled.")

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
# This version now optionally applies periodic boundary conditions.
# Assumes an orthorhombic cell.
# --------------------------
@jit(nopython=True)
def compute_fragments_cpu(positions, radii, scale, n, periodic, cell):
    bonds = []
    if periodic:
        # Assume cell is orthorhombic; extract box lengths from diagonal elements
        Lx = cell[0, 0]
        Ly = cell[1, 1]
        Lz = cell[2, 2]
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            if periodic:
                dx = dx - round(dx / Lx) * Lx
                dy = dy - round(dy / Ly) * Ly
                dz = dz - round(dz / Lz) * Lz
            r = sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = scale * (radii[i] + radii[j])
            if r < cutoff:
                bonds.append((i, j))
    return bonds

# --------------------------
# Compute fragments for one frame (GPU accelerated - Numba)
# (Note: GPU periodic handling is not implemented; if periodic=True and GPU is requested,
# the code will fall back to CPU.)
# --------------------------
@cuda.jit
def compute_fragments_gpu(positions, radii, scale, bond_matrix):
    x = cuda.grid(1)  # 1D grid for simplicity
    n = positions.shape[0]
    if x < n:
        for y in range(x + 1, n):  # Only compute upper triangle
            dx = positions[x, 0] - positions[y, 0]
            dy = positions[x, 1] - positions[y, 1]
            dz = positions[x, 2] - positions[y, 2]
            r = sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = scale * (radii[x] + radii[y])
            if r < cutoff:
                bond_matrix[x, y] = 1  # Mark as bonded

# --------------------------
# Compute fragments for one frame (dispatcher)
# --------------------------
def compute_fragments(atoms, scale=1.2, use_gpu=False, periodic=False):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    radii = np.array([covalent_radii[atoms[i].number] for i in range(n)], dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)

    # If periodic is enabled and GPU is requested, fall back to CPU (GPU PBC not implemented)
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
        if periodic:
            cell = np.array(atoms.get_cell(), dtype=np.float64)
        else:
            cell = np.zeros((3, 3), dtype=np.float64)
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
    i, atoms, use_gpu, gpu_id, periodic = args
    if use_gpu and gpu_id is not None:
        cuda.select_device(gpu_id)
    frags = compute_fragments(atoms, use_gpu=use_gpu, periodic=periodic)
    frag_counts = Counter(frags)
    return i, list(frag_counts.items())

# --------------------------
# GUI Class using Tkinter (Revised Layout with GPU and PBC Options)
# --------------------------
class FragmentCounterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Fragment Counter")
        master.geometry("500x550")

        # Variables
        self.file_path = None
        self.num_cores = tk.IntVar(value=1)
        self.use_gpu = tk.BooleanVar(value=False)
        self.use_pbc = tk.BooleanVar(value=False)
        self.selected_gpu = tk.StringVar(value="All GPUs")

        # Main frame
        main_frame = tk.Frame(master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Headline
        headline = tk.Label(main_frame, text="Fragment Counter", font=("Helvetica", 18, "bold"))
        headline.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # File selection frame
        file_frame = tk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        self.load_button = tk.Button(file_frame, text="Load File", command=self.load_file, width=15)
        self.load_button.grid(row=0, column=0, padx=5)
        self.file_label = tk.Label(file_frame, text="No file loaded", fg="red")
        self.file_label.grid(row=0, column=1, padx=5, sticky="w")

        # Core selection frame
        core_frame = tk.Frame(main_frame)
        core_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        core_label = tk.Label(core_frame, text="Number of cores:")
        core_label.grid(row=0, column=0, padx=5, sticky="e")
        self.core_entry = tk.Entry(core_frame, textvariable=self.num_cores, width=5)
        self.core_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.core_button = tk.Button(core_frame, text="Set Cores", command=self.validate_cores, width=10)
        self.core_button.grid(row=0, column=2, padx=5)

        # GPU options frame
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

        # PBC options frame
        pbc_frame = tk.Frame(main_frame)
        pbc_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        self.pbc_check = tk.Checkbutton(pbc_frame, text="Enable Periodic Boundary Conditions", variable=self.use_pbc)
        self.pbc_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Run and Close buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        self.run_button = tk.Button(button_frame, text="Run Counting", command=self.run_counting, width=20)
        self.run_button.grid(row=0, column=0, padx=10)
        self.close_button = tk.Button(button_frame, text="Close", command=self.master.destroy, width=10)
        self.close_button.grid(row=0, column=1, padx=10)
        self.run_button.config(state=tk.DISABLED)

        # Status label
        self.status_label = tk.Label(main_frame, text="", fg="blue")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select .xyz file", filetypes=[("XYZ files", "*.xyz")]
        )
        if self.file_path:
            self.file_label.config(text=self.file_path, fg="green")
            self.run_button.config(state=tk.NORMAL)
        else:
            self.file_label.config(text="No file loaded", fg="red")
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

        start_time = time.time()
        self.status_label.config(text="Reading trajectory...")
        self.master.update_idletasks()
        try:
            frames = read(self.file_path, index=":")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            self.status_label.config(text="")
            return

        out_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        out_csv = os.path.join(out_dir, f"{base_name}_fragments.csv")

        self.status_label.config(text="Processing frames...")
        self.master.update_idletasks()

        # Prepare arguments for each frame with GPU assignment if enabled
        periodic = self.use_pbc.get()
        args = []
        if self.use_gpu.get():
            selected = self.selected_gpu.get()
            if selected == "All GPUs":
                num_gpus = len(cuda.gpus)
                args = [(i, atoms, True, i % num_gpus, periodic) for i, atoms in enumerate(frames)]
            else:
                gpu_id = int(selected.split()[1])
                args = [(i, atoms, True, gpu_id, periodic) for i, atoms in enumerate(frames)]
        else:
            args = [(i, atoms, False, None, periodic) for i, atoms in enumerate(frames)]

        with open(out_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Fragment_Type", "Count"])

            with Pool(processes=num_cores) as pool:
                for i, frag_counts_list in pool.imap(process_frame, args):
                    for frag, count in frag_counts_list:
                        writer.writerow([i, frag, count])
                    self.status_label.config(text=f"Processed frame {i+1} of {len(frames)}")
                    self.master.update_idletasks()
                pool.close()
                pool.join()

        total_time = time.time() - start_time
        self.status_label.config(text=f"Done! CSV saved to:\n{out_csv}")
        messagebox.showinfo("Done", f"Fragment counting complete.\nResults saved to:\n{out_csv}\nTotal time used: {total_time:.2f} seconds")

# --------------------------
# Run the GUI
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = FragmentCounterGUI(root)
    root.mainloop()
