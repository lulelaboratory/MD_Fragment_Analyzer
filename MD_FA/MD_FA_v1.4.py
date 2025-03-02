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
from ase.io import read
from ase.data import covalent_radii, vdw_radii  # only the needed submodules

# Numba imports (only CPU version used)
from numba import jit, float64, int32

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
# This version optionally applies periodic boundary conditions.
# Assumes an orthorhombic cell.
# --------------------------
@jit(nopython=True)
def compute_fragments_cpu(positions, radii, scale, n, periodic, cell):
    bonds = []
    if periodic:
        # Assume cell is orthorhombic; get box lengths from diagonal elements
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
# Compute fragments for one frame (dispatcher)
# --------------------------
def compute_fragments(atoms, scale=1.2, periodic=False, radius_type="covalent"):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    # Select which radii to use based on user choice
    if radius_type.lower() == "vdw":
        radii = np.array([vdw_radii[atoms[i].number] for i in range(n)], dtype=np.float64)
    else:
        radii = np.array([covalent_radii[atoms[i].number] for i in range(n)], dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)

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
    i, atoms, periodic, radius_type = args
    frags = compute_fragments(atoms, periodic=periodic, radius_type=radius_type)
    frag_counts = Counter(frags)
    return i, list(frag_counts.items())

# --------------------------
# GUI Class using Tkinter (with Menu Bar and Status Box)
# --------------------------
class FragmentCounterGUI:
    def __init__(self, master):
        self.master = master
        master.title("MD Fragment Analyzer")
        master.geometry("550x500")
        
        # Create Menu Bar
        self.create_menu()

        # Variables
        self.file_path = None
        self.num_cores = tk.IntVar(value=1)
        self.use_pbc = tk.BooleanVar(value=False)
        self.radius_type = tk.StringVar(value="Covalent Radii")
        self.file_var = tk.StringVar(value="")

        # Main frame
        main_frame = tk.Frame(master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Headline
        headline = tk.Label(main_frame, text="MD Fragment Analyzer", font=("Helvetica", 20, "bold"))
        headline.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # File selection frame with copyable Entry
        file_frame = tk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.load_button = tk.Button(file_frame, text="Load File", command=self.load_file, width=15)
        self.load_button.grid(row=0, column=0, padx=5)
        self.file_entry = tk.Entry(file_frame, textvariable=self.file_var, width=50, state='readonly')
        self.file_entry.grid(row=0, column=1, padx=5, sticky="w")

        # Core selection frame
        core_frame = tk.Frame(main_frame)
        core_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        core_label = tk.Label(core_frame, text="Number of cores:")
        core_label.grid(row=0, column=0, padx=5, sticky="e")
        self.core_entry = tk.Entry(core_frame, textvariable=self.num_cores, width=5)
        self.core_entry.grid(row=0, column=1, padx=5, sticky="w")
        self.core_button = tk.Button(core_frame, text="Set Cores", command=self.validate_cores, width=10)
        self.core_button.grid(row=0, column=2, padx=5)

        # Radii selection frame
        radii_frame = tk.Frame(main_frame)
        radii_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        radii_label = tk.Label(radii_frame, text="Select Radii Type:")
        radii_label.grid(row=0, column=0, padx=5, sticky="e")
        radii_options = ["Covalent Radii", "Van der Waals Radii"]
        self.radii_optionmenu = tk.OptionMenu(radii_frame, self.radius_type, *radii_options)
        self.radii_optionmenu.grid(row=0, column=1, padx=5, sticky="w")

        # PBC options frame
        pbc_frame = tk.Frame(main_frame)
        pbc_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        self.pbc_check = tk.Checkbutton(pbc_frame, text="Enable Periodic Boundary Conditions", variable=self.use_pbc)
        self.pbc_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Run and Close buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        self.run_button = tk.Button(button_frame, text="Run Counting", command=self.run_counting, width=20)
        self.run_button.grid(row=0, column=0, padx=10)
        self.close_button = tk.Button(button_frame, text="Close", command=self.master.destroy, width=10)
        self.close_button.grid(row=0, column=1, padx=10)
        self.run_button.config(state=tk.DISABLED)

        # Status message area (ScrolledText)
        status_frame = tk.Frame(main_frame)
        status_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky="nsew")
        status_label = tk.Label(status_frame, text="Status Messages:")
        status_label.pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weight so status box expands
        main_frame.grid_rowconfigure(6, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

    def create_menu(self):
        # Create a menu bar with "About" and "Methodology"
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
            "Copyright (c) 2025 Le Lu (lulelaboratory@gmail.com)\n\n"
            "MIT License\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies\n"
            "of the Software, and to permit persons to whom the Software is furnished to do so,\n"
            "subject to the following conditions:\n\n"
            "The above copyright notice and this permission notice shall be included in all\n"
            "copies or substantial portions of the Software.\n\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,\n"
            "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR\n"
            "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE\n"
            "FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,\n"
            "ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"
            "For full details, see: https://en.wikipedia.org/wiki/MIT_License"
        )
        # Create a custom About window using Toplevel
        about_win = tk.Toplevel(self.master)
        about_win.title("About")
        about_win.geometry("600x400")  # Set width=600, height=400
        # Create a Text widget to display the about text
        txt = tk.Text(about_win, wrap="word")
        txt.insert("1.0", about_text)
        txt.config(state="disabled")  # Make the text read-only
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        # Add a button to close the About window
        btn_close = tk.Button(about_win, text="OK", command=about_win.destroy)
        btn_close.pack(pady=5)

    def show_methodology(self):
        # Updated methodology text with union-find and naive find complexity details
        methodology_text = (
            "Methodology:\n\n"
            "This tool analyzes MD trajectories stored in XYZ format by processing each frame to identify fragments—\n"
            "groups of atoms that are bonded together based on a distance cutoff. Once bonds are determined,\n"
            "the union-find (disjoint-set) algorithm is employed to group connected atoms into fragments.\n\n"
            "Union-Find Details:\n"
            "-------------------\n"
            "The union-find algorithm supports two primary operations:\n\n"
            "  1. Find Operation:\n"
            "     - Determines the representative (root) of the set containing a given node.\n"
            "     - A naive implementation might scan all nodes in the set, resulting in O(n) time per operation.\n"
            "     - With optimizations such as path compression, the worst-case time is reduced to O(log(n))\n"
            "       (or nearly O(1) amortized), with O(1) extra space per operation.\n\n"
            "  2. Union Operation:\n"
            "     - Merges two disjoint sets using union by rank.\n"
            "     - Amortized Time Complexity: O(1) with union by rank, with O(1) space overhead.\n\n"
            "Overall Code Complexity:\n"
            "  - Time: O(n log(n)) in the worst-case without full optimizations; with path compression and union by rank,\n"
            "    the total cost is nearly linear (O(n * α(n))), where α(n) is the inverse Ackermann function.\n"
            "  - Space: O(n) to maintain the parent and rank arrays, where n is the total number of nodes.\n\n"
            "Why is Union-Find Better?\n"
            "--------------------------\n"
            "A naive find operation could take O(n) time per query by scanning every node in the set,\n"
            "while the optimized union-find algorithm uses path compression to flatten the tree and\n"
            "union by rank to merge sets efficiently, resulting in nearly constant time per operation.\n\n"
        )
        # Create a custom Methodology window using Toplevel
        meth_win = tk.Toplevel(self.master)
        meth_win.title("Methodology")
        meth_win.geometry("600x400")  # Set width=600, height=400
        # Create a Text widget to display the methodology text
        txt = tk.Text(meth_win, wrap="word")
        txt.insert("1.0", methodology_text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        # Add a button to close the Methodology window
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
        if "vdw" in selected_radius.lower():
            radius_type = "vdw"
        else:
            radius_type = "covalent"

        start_time = time.time()
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.append_status("Reading trajectory...\n")
        self.master.update_idletasks()
        try:
            frames = read(self.file_path, index=":")
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
        args = [(i, atoms, periodic, radius_type) for i, atoms in enumerate(frames)]

        with open(out_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Fragment_Type", "Count"])

            with Pool(processes=num_cores) as pool:
                for i, frag_counts_list in pool.imap(process_frame, args):
                    for frag, count in frag_counts_list:
                        writer.writerow([i, frag, count])
                    # Update status every 5 frames to reduce overhead
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
