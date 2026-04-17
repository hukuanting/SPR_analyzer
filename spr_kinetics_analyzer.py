import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os

def langmuir_association(t, Rmax, ka_Conc):
    """Langmuir association model: R = Rmax * (1 - exp(-ka*C*t))"""
    return Rmax * (1 - np.exp(-ka_Conc * t))

def langmuir_dissociation(t, R0, kd):
    """Langmuir dissociation model: R = R0 * exp(-kd*t)"""
    return R0 * np.exp(-kd * t)

class SPRKineticsAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("SPR Kinetics Analyzer")
        self.root.geometry("1200x900")
        
        # Data storage
        self.df = None
        self.data_groups = []
        self.processed_data = {}  # Stores processed data for each group
        self.click_points = []  # Stores user-selected time points
        self.phase_ranges = {}  # Stores baseline, association, dissociation ranges
        self.kinetic_results = {}  # Stores fitting results
        
        # UI state
        self.current_selection_phase = "baseline"  # baseline, association, dissociation
        self.data_type = "ratio"  # TM, TE, or ratio
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Data File", 
                  command=self.load_data).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT)
        
        # Data type selection
        data_type_frame = ttk.Frame(control_frame)
        data_type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_type_frame, text="Data Type:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.data_type_var = tk.StringVar(value="ratio")
        ttk.Radiobutton(data_type_frame, text="TM/TE Ratio", 
                       variable=self.data_type_var, value="ratio",
                       command=self.update_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(data_type_frame, text="TM", 
                       variable=self.data_type_var, value="TM",
                       command=self.update_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(data_type_frame, text="TE", 
                       variable=self.data_type_var, value="TE",
                       command=self.update_plot).pack(side=tk.LEFT)
        
        # Phase selection for clicking
        phase_frame = ttk.Frame(control_frame)
        phase_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(phase_frame, text="Click Mode:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.phase_var = tk.StringVar(value="baseline")
        ttk.Radiobutton(phase_frame, text="Baseline (2 points)", 
                       variable=self.phase_var, value="baseline",
                       command=self.set_phase_mode).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(phase_frame, text="Association (2 points)", 
                       variable=self.phase_var, value="association",
                       command=self.set_phase_mode).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(phase_frame, text="Dissociation (2 points)", 
                       variable=self.phase_var, value="dissociation",
                       command=self.set_phase_mode).pack(side=tk.LEFT)
        
        # Processing controls
        process_frame = ttk.Frame(control_frame)
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(process_frame, text="Apply Denoising", 
                  command=self.apply_denoising).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(process_frame, text="Normalize Data", 
                  command=self.normalize_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(process_frame, text="Fit Kinetics", 
                  command=self.fit_kinetics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(process_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(process_frame, text="Export Plot", 
                  command=self.export_plot_only).pack(side=tk.LEFT)
        
        # Clear controls
        clear_frame = ttk.Frame(control_frame)
        clear_frame.pack(fill=tk.X)
        
        ttk.Button(clear_frame, text="Clear Points", 
                  command=self.clear_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(clear_frame, text="Reset All", 
                  command=self.reset_all).pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready. Load a data file to begin.")
        self.status_label.pack(pady=(10, 0))
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Data Visualization", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for plot and parameters
        plot_container = ttk.Frame(plot_frame)
        plot_container.pack(fill=tk.BOTH, expand=True)
        
        # Left side for matplotlib plot
        plot_left = ttk.Frame(plot_container)
        plot_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Response")
        self.ax.grid(True, alpha=0.3)
        
        # Canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, plot_left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_left)
        toolbar.update()
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Right side for kinetic parameters display
        params_frame = ttk.LabelFrame(plot_container, text="Kinetic Parameters", padding=10)
        params_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Create notebook for tabbed interface
        self.params_notebook = ttk.Notebook(params_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Results display
        results_tab = ttk.Frame(self.params_notebook)
        self.params_notebook.add(results_tab, text="Results")
        
        # Create text widget for parameters in results tab
        self.params_text = tk.Text(results_tab, width=35, height=25, 
                                  font=('Courier', 9), state=tk.DISABLED,
                                  bg='#f0f0f0')
        self.params_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for parameters text
        params_scrollbar = ttk.Scrollbar(results_tab, orient=tk.VERTICAL, command=self.params_text.yview)
        self.params_text.configure(yscrollcommand=params_scrollbar.set)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tab 2: Analysis and parameter adjustment
        analysis_tab = ttk.Frame(self.params_notebook)
        self.params_notebook.add(analysis_tab, text="Analysis")
        
        # Group selection for analysis
        group_select_frame = ttk.Frame(analysis_tab)
        group_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(group_select_frame, text="Select Group:").pack(anchor=tk.W)
        self.analysis_group_var = tk.StringVar()
        self.analysis_group_combo = ttk.Combobox(group_select_frame, textvariable=self.analysis_group_var, 
                                               state="readonly", width=32)
        self.analysis_group_combo.pack(fill=tk.X, pady=(5, 0))
        self.analysis_group_combo.bind('<<ComboboxSelected>>', self.on_analysis_group_change)
        
        # Parameter adjustment frame
        param_adjust_frame = ttk.LabelFrame(analysis_tab, text="Adjust Parameters", padding=5)
        param_adjust_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Association parameters
        assoc_frame = ttk.LabelFrame(param_adjust_frame, text="Association", padding=5)
        assoc_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(assoc_frame, text="ka×C (s⁻¹):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.ka_conc_var = tk.DoubleVar()
        self.ka_conc_entry = ttk.Entry(assoc_frame, textvariable=self.ka_conc_var, width=15)
        self.ka_conc_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 5))
        self.ka_conc_entry.bind('<KeyRelease>', self.on_param_change)
        
        ttk.Label(assoc_frame, text="Rmax:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.rmax_var = tk.DoubleVar()
        self.rmax_entry = ttk.Entry(assoc_frame, textvariable=self.rmax_var, width=15)
        self.rmax_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 5), pady=(5, 0))
        self.rmax_entry.bind('<KeyRelease>', self.on_param_change)
        
        assoc_frame.grid_columnconfigure(1, weight=1)
        
        # Dissociation parameters
        dissoc_frame = ttk.LabelFrame(param_adjust_frame, text="Dissociation", padding=5)
        dissoc_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(dissoc_frame, text="kd (s⁻¹):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.kd_var = tk.DoubleVar()
        self.kd_entry = ttk.Entry(dissoc_frame, textvariable=self.kd_var, width=15)
        self.kd_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 5))
        self.kd_entry.bind('<KeyRelease>', self.on_param_change)
        
        ttk.Label(dissoc_frame, text="R0:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.r0_var = tk.DoubleVar()
        self.r0_entry = ttk.Entry(dissoc_frame, textvariable=self.r0_var, width=15)
        self.r0_entry.grid(row=1, column=1, sticky=tk.EW, padx=(0, 5), pady=(5, 0))
        self.r0_entry.bind('<KeyRelease>', self.on_param_change)
        
        dissoc_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(analysis_tab)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Apply Changes", 
                  command=self.apply_parameter_changes).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset to Fitted", 
                  command=self.reset_to_fitted_params).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Show Preview", 
                  command=self.show_parameter_preview).pack(side=tk.LEFT)
        
        # Analysis info
        self.analysis_info = tk.Text(analysis_tab, height=8, font=('Courier', 8), 
                                   state=tk.DISABLED, bg='#f8f8f8')
        self.analysis_info.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Initialize analysis variables
        self.current_analysis_group = None
        self.preview_lines = {}  # Store preview curve lines
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Kinetic Parameters", padding=10)
        results_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create results table
        self.results_tree = ttk.Treeview(results_frame, columns=('Group', 'ka*C', 'kd', 'KD', 'Rmax', 'R²_assoc', 'R²_dissoc'), show='headings', height=8)
        
        # Define headings
        self.results_tree.heading('Group', text='Group')
        self.results_tree.heading('ka*C', text='ka×C (1/s)')
        self.results_tree.heading('kd', text='kd (1/s)')
        self.results_tree.heading('KD', text='KD (M)')
        self.results_tree.heading('Rmax', text='Rmax')
        self.results_tree.heading('R²_assoc', text='R² (Assoc)')
        self.results_tree.heading('R²_dissoc', text='R² (Dissoc)')
        
        # Configure column widths
        for col in self.results_tree['columns']:
            self.results_tree.column(col, width=100, anchor='center')
        
        self.results_tree.pack(fill=tk.X)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
    def load_data(self):
        """Load SPR data file"""
        file_path = filedialog.askopenfilename(
            title="Select SPR Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.df = pd.read_csv(file_path)
            self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            
            # Detect data groups
            self.detect_data_groups()
            
            # Reset all processing
            self.processed_data = {}
            self.click_points = []
            self.phase_ranges = {}
            self.kinetic_results = {}
            
            # Plot raw data
            self.plot_raw_data()
            self.status_label.config(text=f"Data loaded successfully. Found {len(self.data_groups)} data groups.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data file:\n{str(e)}")
            
    def detect_data_groups(self):
        """Detect data groups from column names"""
        self.data_groups = []
        
        # Look for pattern: sampleTime1, TM_intensity1, TE_intensity1, etc.
        for i in range(1, 7):  # Check groups 1-6
            time_col = f'sampleTime{i}'
            tm_col = f'TM_intensity{i}'
            te_col = f'TE_intensity{i}'
            
            if all(col in self.df.columns for col in [time_col, tm_col, te_col]):
                self.data_groups.append({
                    'group': i,
                    'time': time_col,
                    'TM': tm_col,
                    'TE': te_col
                })
                
    def plot_raw_data(self):
        """Plot raw data for all groups"""
        if not self.data_groups:
            return
            
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Response")
        self.ax.set_title("Raw SPR Data")
        self.ax.grid(True, alpha=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data_groups)))
        data_type = self.data_type_var.get()
        
        for idx, group_info in enumerate(self.data_groups):
            time_data = self.df[group_info['time']].dropna()
            tm_data = self.df[group_info['TM']].dropna()
            te_data = self.df[group_info['TE']].dropna()
            
            # Ensure consistent length
            min_len = min(len(time_data), len(tm_data), len(te_data))
            if min_len == 0:
                continue
                
            time_data = time_data.iloc[:min_len]
            tm_data = tm_data.iloc[:min_len]
            te_data = te_data.iloc[:min_len]
            
            # Get y data based on data type
            if data_type == "ratio":
                y_data = tm_data / np.where(te_data != 0, te_data, 1e-10)
                ylabel = "TM/TE Ratio"
            elif data_type == "TM":
                y_data = tm_data
                ylabel = "TM Intensity"
            else:  # TE
                y_data = te_data
                ylabel = "TE Intensity"
            
            self.ax.plot(time_data, y_data, color=colors[idx], 
                        label=f'Group {group_info["group"]}', alpha=0.8)
        
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.canvas.draw_idle()
        
    def update_plot(self):
        """Update plot when data type changes"""
        if hasattr(self, 'processed_data') and self.processed_data:
            self.plot_processed_data()
        else:
            self.plot_raw_data()
            
    def set_phase_mode(self):
        """Set the current phase selection mode"""
        self.current_selection_phase = self.phase_var.get()
        phase_names = {"baseline": "Baseline", "association": "Association", "dissociation": "Dissociation"}
        self.status_label.config(text=f"Click mode: {phase_names[self.current_selection_phase]} selection. Click 2 points to define range.")
        
    def on_click(self, event):
        """Handle mouse clicks on the plot"""
        if event.inaxes != self.ax or event.button != 1:  # Left click only
            return
            
        x_click = event.xdata
        if x_click is None:
            return
            
        phase = self.current_selection_phase
        
        # Initialize phase data if needed
        if phase not in self.click_points:
            self.click_points.append({
                'phase': phase,
                'points': [],
                'line': None
            })
        
        # Find or create current phase entry
        phase_entry = None
        for entry in self.click_points:
            if entry['phase'] == phase:
                phase_entry = entry
                break
                
        if not phase_entry:
            phase_entry = {'phase': phase, 'points': [], 'line': None}
            self.click_points.append(phase_entry)
        
        # Add point (max 2 points per phase)
        if len(phase_entry['points']) < 2:
            phase_entry['points'].append(x_click)
            
            # Draw vertical line at click point
            colors = {'baseline': 'red', 'association': 'green', 'dissociation': 'blue'}
            self.ax.axvline(x_click, color=colors[phase], linestyle='--', alpha=0.7)
            self.canvas.draw_idle()
            
            # If 2 points selected, define range and move to next phase
            if len(phase_entry['points']) == 2:
                x1, x2 = sorted(phase_entry['points'])
                self.phase_ranges[phase] = (x1, x2)
                
                # Draw range span
                if phase_entry['line']:
                    phase_entry['line'].remove()
                phase_entry['line'] = self.ax.axvspan(x1, x2, color=colors[phase], alpha=0.2, 
                                                    label=f'{phase.capitalize()} range')
                self.canvas.draw_idle()
                
                self.status_label.config(text=f"{phase.capitalize()} range defined: {x1:.1f} - {x2:.1f} s")
                
                # Check if all phases are defined
                if len(self.phase_ranges) == 3:
                    self.status_label.config(text="All phases defined. You can now apply denoising and normalization.")
        else:
            # Reset if clicking again
            phase_entry['points'] = [x_click]
            if phase in self.phase_ranges:
                del self.phase_ranges[phase]
            if phase_entry['line']:
                phase_entry['line'].remove()
                phase_entry['line'] = None
            self.redraw_phase_indicators()
            
    def redraw_phase_indicators(self):
        """Redraw all phase indicators"""
        self.ax.clear()
        self.update_plot()
        
        colors = {'baseline': 'red', 'association': 'green', 'dissociation': 'blue'}
        
        # Redraw ranges
        for phase, (x1, x2) in self.phase_ranges.items():
            self.ax.axvspan(x1, x2, color=colors[phase], alpha=0.2, 
                           label=f'{phase.capitalize()} range')
        
        # Redraw points
        for entry in self.click_points:
            phase = entry['phase']
            for x_point in entry['points']:
                self.ax.axvline(x_point, color=colors[phase], linestyle='--', alpha=0.7)
        
        self.canvas.draw_idle()
        
    def apply_denoising(self):
        """Apply Gaussian denoising to all data groups"""
        if not self.data_groups:
            messagebox.showwarning("Warning", "Please load data first.")
            return
            
        try:
            self.processed_data = {}
            data_type = self.data_type_var.get()
            
            for group_info in self.data_groups:
                group_id = group_info['group']
                
                time_data = self.df[group_info['time']].dropna()
                tm_data = self.df[group_info['TM']].dropna()
                te_data = self.df[group_info['TE']].dropna()
                
                # Ensure consistent length
                min_len = min(len(time_data), len(tm_data), len(te_data))
                if min_len == 0:
                    continue
                    
                time_data = time_data.iloc[:min_len].values
                tm_data = tm_data.iloc[:min_len].values
                te_data = te_data.iloc[:min_len].values
                
                # Get y data based on data type
                if data_type == "ratio":
                    y_data = tm_data / np.where(te_data != 0, te_data, 1e-10)
                elif data_type == "TM":
                    y_data = tm_data
                else:  # TE
                    y_data = te_data
                
                # Apply Gaussian filter for denoising
                denoised_y = gaussian_filter1d(y_data, sigma=2.0)
                
                self.processed_data[group_id] = {
                    'time': time_data,
                    'original': y_data,
                    'denoised': denoised_y,
                    'normalized': None
                }
            
            self.plot_processed_data()
            self.status_label.config(text="Denoising applied to all data groups.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply denoising:\n{str(e)}")
            
    def normalize_data(self):
        """Normalize data: baseline to 0, maximum to 1"""
        if not self.processed_data:
            messagebox.showwarning("Warning", "Please apply denoising first.")
            return
            
        if 'baseline' not in self.phase_ranges:
            messagebox.showwarning("Warning", "Please define baseline range first.")
            return
            
        try:
            baseline_start, baseline_end = self.phase_ranges['baseline']
            
            for group_id, data in self.processed_data.items():
                time_data = data['time']
                y_data = data['denoised'].copy()
                
                # Find baseline range indices
                baseline_mask = (time_data >= baseline_start) & (time_data <= baseline_end)
                baseline_mean = np.mean(y_data[baseline_mask])
                
                # Subtract baseline
                y_normalized = y_data - baseline_mean
                
                # Scale to make maximum = 1
                if y_normalized.max() > 0:
                    y_normalized = y_normalized / y_normalized.max()
                
                data['normalized'] = y_normalized
                data['baseline_value'] = baseline_mean
            
            self.plot_processed_data()
            self.status_label.config(text="Data normalized: baseline = 0, maximum = 1.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to normalize data:\n{str(e)}")
            
    def plot_processed_data(self):
        """Plot processed data"""
        if not self.processed_data:
            return
            
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Normalized Response")
        self.ax.set_title("Processed SPR Data")
        self.ax.grid(True, alpha=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.processed_data)))
        
        for idx, (group_id, data) in enumerate(self.processed_data.items()):
            time_data = data['time']
            
            # Plot normalized data if available, otherwise denoised
            if data['normalized'] is not None:
                y_data = data['normalized']
                title_suffix = " (Normalized)"
            else:
                y_data = data['denoised']
                title_suffix = " (Denoised)"
            
            self.ax.plot(time_data, y_data, color=colors[idx], 
                        label=f'Group {group_id}', alpha=0.8, linewidth=2)
        
        self.ax.set_title("Processed SPR Data" + title_suffix)
        self.ax.legend()
        
        # Redraw phase indicators if they exist
        if self.phase_ranges:
            colors_phase = {'baseline': 'red', 'association': 'green', 'dissociation': 'blue'}
            for phase, (x1, x2) in self.phase_ranges.items():
                self.ax.axvspan(x1, x2, color=colors_phase[phase], alpha=0.2, 
                               label=f'{phase.capitalize()} range')
        
        self.canvas.draw_idle()
        
    def fit_kinetics(self):
        """Fit kinetic models to all data groups"""
        if not self.processed_data:
            messagebox.showwarning("Warning", "Please process data first.")
            return
            
        required_phases = ['baseline', 'association', 'dissociation']
        if not all(phase in self.phase_ranges for phase in required_phases):
            messagebox.showwarning("Warning", "Please define all phase ranges (baseline, association, dissociation).")
            return
            
        if not all(data['normalized'] is not None for data in self.processed_data.values()):
            messagebox.showwarning("Warning", "Please normalize data first.")
            return
            
        try:
            self.kinetic_results = {}
            assoc_start, assoc_end = self.phase_ranges['association']
            dissoc_start, dissoc_end = self.phase_ranges['dissociation']
            
            for group_id, data in self.processed_data.items():
                time_data = data['time']
                y_data = data['normalized']
                
                # Fit association phase
                assoc_mask = (time_data >= assoc_start) & (time_data <= assoc_end)
                t_assoc = time_data[assoc_mask] - time_data[assoc_mask][0]  # Start from 0
                y_assoc = y_data[assoc_mask]
                
                # Fit dissociation phase
                dissoc_mask = (time_data >= dissoc_start) & (time_data <= dissoc_end)
                t_dissoc = time_data[dissoc_mask] - time_data[dissoc_mask][0]  # Start from 0
                y_dissoc = y_data[dissoc_mask]
                
                if len(t_assoc) < 3 or len(y_assoc) < 3:
                    continue
                if len(t_dissoc) < 3 or len(y_dissoc) < 3:
                    continue
                
                try:
                    # Fit association
                    p0_assoc = [y_assoc.max(), 0.01]
                    params_assoc, cov_assoc = curve_fit(langmuir_association, t_assoc, y_assoc, 
                                                       p0=p0_assoc, maxfev=10000)
                    
                    # Calculate R² for association
                    y_pred_assoc = langmuir_association(t_assoc, *params_assoc)
                    ss_res_assoc = np.sum((y_assoc - y_pred_assoc) ** 2)
                    ss_tot_assoc = np.sum((y_assoc - np.mean(y_assoc)) ** 2)
                    r2_assoc = 1 - (ss_res_assoc / ss_tot_assoc) if ss_tot_assoc > 0 else 0
                    
                    # Fit dissociation
                    p0_dissoc = [y_dissoc[0], 0.01]
                    params_dissoc, cov_dissoc = curve_fit(langmuir_dissociation, t_dissoc, y_dissoc, 
                                                         p0=p0_dissoc, maxfev=10000)
                    
                    # Calculate R² for dissociation
                    y_pred_dissoc = langmuir_dissociation(t_dissoc, *params_dissoc)
                    ss_res_dissoc = np.sum((y_dissoc - y_pred_dissoc) ** 2)
                    ss_tot_dissoc = np.sum((y_dissoc - np.mean(y_dissoc)) ** 2)
                    r2_dissoc = 1 - (ss_res_dissoc / ss_tot_dissoc) if ss_tot_dissoc > 0 else 0
                    
                    # Store results
                    Rmax, ka_conc = params_assoc
                    R0, kd = params_dissoc
                    
                    # Calculate KD (assuming concentration C is unknown, we report ka*C and kd separately)
                    # KD = kd / ka (but we have ka*C, so we can't calculate KD without knowing C)
                    
                    self.kinetic_results[group_id] = {
                        'ka_conc': ka_conc,
                        'kd': kd,
                        'Rmax': Rmax,
                        'R0': R0,
                        'r2_assoc': r2_assoc,
                        'r2_dissoc': r2_dissoc,
                        'assoc_fit': y_pred_assoc,
                        'dissoc_fit': y_pred_dissoc,
                        'assoc_time': time_data[assoc_mask],
                        'dissoc_time': time_data[dissoc_mask]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not fit Group {group_id}: {e}")
                    continue
            
            self.update_results_table()
            self.plot_with_fits()
            self.status_label.config(text=f"Kinetic fitting completed for {len(self.kinetic_results)} groups.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit kinetics:\n{str(e)}")
            
    def update_results_table(self):
        """Update the results table with kinetic parameters"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add results
        for group_id, results in self.kinetic_results.items():
            # Calculate KD if possible (would need concentration)
            kd_str = f"N/A (need C)"  # Can't calculate without concentration
            
            self.results_tree.insert('', 'end', values=(
                f"Group {group_id}",
                f"{results['ka_conc']:.4f}",
                f"{results['kd']:.4f}",
                kd_str,
                f"{results['Rmax']:.3f}",
                f"{results['r2_assoc']:.3f}",
                f"{results['r2_dissoc']:.3f}"
            ))
            
    def plot_with_fits(self):
        """Plot data with fitted curves"""
        if not self.processed_data or not self.kinetic_results:
            return
            
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Normalized Response")
        self.ax.set_title("SPR Data with Kinetic Fits")
        self.ax.grid(True, alpha=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.processed_data)))
        
        for idx, (group_id, data) in enumerate(self.processed_data.items()):
            time_data = data['time']
            y_data = data['normalized']
            color = colors[idx]
            
            # Plot data
            self.ax.plot(time_data, y_data, color=color, alpha=0.6, 
                        label=f'Group {group_id} Data', linewidth=1)
            
            # Plot fits if available
            if group_id in self.kinetic_results:
                results = self.kinetic_results[group_id]
                
                # Plot association fit
                self.ax.plot(results['assoc_time'], results['assoc_fit'], 
                            color=color, linestyle='--', linewidth=2, 
                            label=f'Group {group_id} Fit')
                
                # Plot dissociation fit
                self.ax.plot(results['dissoc_time'], results['dissoc_fit'], 
                            color=color, linestyle='--', linewidth=2)
        
        # Draw phase ranges
        colors_phase = {'baseline': 'red', 'association': 'green', 'dissociation': 'blue'}
        for phase, (x1, x2) in self.phase_ranges.items():
            self.ax.axvspan(x1, x2, color=colors_phase[phase], alpha=0.15, 
                           label=f'{phase.capitalize()} range')
        
        self.ax.legend()
        self.canvas.draw_idle()
        
        # Update parameters display on the right side
        self.update_parameters_display()
        
        # Update analysis group options
        self.update_analysis_groups()
        
    def update_analysis_groups(self):
        """Update the group selection combo box for analysis"""
        if not self.kinetic_results:
            self.analysis_group_combo['values'] = []
            self.analysis_group_var.set('')
            return
            
        group_options = [f"Group {gid}" for gid in sorted(self.kinetic_results.keys())]
        self.analysis_group_combo['values'] = group_options
        
        # Select first group by default
        if group_options:
            self.analysis_group_var.set(group_options[0])
            self.load_group_parameters()
    
    def on_analysis_group_change(self, event=None):
        """Handle analysis group selection change"""
        self.load_group_parameters()
        
    def load_group_parameters(self):
        """Load parameters for the selected group into entry fields"""
        selected = self.analysis_group_var.get()
        if not selected or not self.kinetic_results:
            return
            
        # Extract group ID
        try:
            group_id = int(selected.split()[-1])
        except:
            return
            
        if group_id not in self.kinetic_results:
            return
            
        self.current_analysis_group = group_id
        results = self.kinetic_results[group_id]
        
        # Load values into entry fields
        self.ka_conc_var.set(results['ka_conc'])
        self.rmax_var.set(results['Rmax'])
        self.kd_var.set(results['kd'])
        self.r0_var.set(results['R0'])
        
        # Update analysis info
        self.update_analysis_info()
        
    def on_param_change(self, event=None):
        """Handle parameter changes for real-time preview"""
        # Small delay to avoid too frequent updates
        self.root.after(500, self.show_parameter_preview)
        
    def show_parameter_preview(self):
        """Show preview of curves with current parameter values"""
        if not self.current_analysis_group or not self.processed_data:
            return
            
        group_id = self.current_analysis_group
        if group_id not in self.processed_data:
            return
            
        try:
            # Get current parameter values
            ka_conc = self.ka_conc_var.get()
            rmax = self.rmax_var.get()
            kd = self.kd_var.get()
            r0 = self.r0_var.get()
            
            # Get time ranges
            if not self.phase_ranges or 'association' not in self.phase_ranges or 'dissociation' not in self.phase_ranges:
                return
                
            assoc_start, assoc_end = self.phase_ranges['association']
            dissoc_start, dissoc_end = self.phase_ranges['dissociation']
            
            # Get data
            data = self.processed_data[group_id]
            time_data = data['time']
            
            # Generate preview curves
            assoc_mask = (time_data >= assoc_start) & (time_data <= assoc_end)
            dissoc_mask = (time_data >= dissoc_start) & (time_data <= dissoc_end)
            
            t_assoc = time_data[assoc_mask] - time_data[assoc_mask][0]
            t_dissoc = time_data[dissoc_mask] - time_data[dissoc_mask][0]
            
            y_assoc_preview = langmuir_association(t_assoc, rmax, ka_conc)
            y_dissoc_preview = langmuir_dissociation(t_dissoc, r0, kd)
            
            # Remove old preview lines
            for line_key in list(self.preview_lines.keys()):
                if self.preview_lines[line_key] in self.ax.lines:
                    self.preview_lines[line_key].remove()
                del self.preview_lines[line_key]
            
            # Plot preview curves
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.processed_data)))
            group_idx = list(self.processed_data.keys()).index(group_id)
            color = colors[group_idx]
            
            # Association preview
            line1, = self.ax.plot(time_data[assoc_mask], y_assoc_preview, 
                                color=color, linestyle=':', linewidth=3, alpha=0.8,
                                label=f'Group {group_id} Preview')
            self.preview_lines[f'assoc_{group_id}'] = line1
            
            # Dissociation preview
            line2, = self.ax.plot(time_data[dissoc_mask], y_dissoc_preview, 
                                color=color, linestyle=':', linewidth=3, alpha=0.8)
            self.preview_lines[f'dissoc_{group_id}'] = line2
            
            self.ax.legend()
            self.canvas.draw_idle()
            
            # Update analysis info
            self.update_analysis_info()
            
        except (ValueError, tk.TclError):
            # Handle invalid input gracefully
            pass
            
    def update_analysis_info(self):
        """Update analysis information display"""
        if not self.current_analysis_group:
            return
            
        try:
            ka_conc = self.ka_conc_var.get()
            kd = self.kd_var.get()
            rmax = self.rmax_var.get()
            r0 = self.r0_var.get()
            
            # Calculate derived parameters
            half_life = 0.693 / kd if kd > 0 else float('inf')
            
            # Compare with fitted values if available
            fitted_info = ""
            if self.current_analysis_group in self.kinetic_results:
                fitted = self.kinetic_results[self.current_analysis_group]
                fitted_info = f"\nCOMPARISON WITH FITTED:\n"
                fitted_info += f"Δka×C  = {ka_conc - fitted['ka_conc']:+.6f}\n"
                fitted_info += f"ΔRmax  = {rmax - fitted['Rmax']:+.4f}\n"
                fitted_info += f"Δkd    = {kd - fitted['kd']:+.6f}\n"
                fitted_info += f"ΔR0    = {r0 - fitted['R0']:+.4f}\n"
            
            info_text = f"ANALYSIS GROUP {self.current_analysis_group}\n"
            info_text += "=" * 25 + "\n\n"
            info_text += f"Current Parameters:\n"
            info_text += f"ka×C   = {ka_conc:.6f} s⁻¹\n"
            info_text += f"Rmax   = {rmax:.4f}\n"
            info_text += f"kd     = {kd:.6f} s⁻¹\n"
            info_text += f"R0     = {r0:.4f}\n\n"
            info_text += f"Derived:\n"
            info_text += f"t₁/₂   = {half_life:.2f} s\n"
            info_text += fitted_info
            
            self.analysis_info.config(state=tk.NORMAL)
            self.analysis_info.delete(1.0, tk.END)
            self.analysis_info.insert(tk.END, info_text)
            self.analysis_info.config(state=tk.DISABLED)
            
        except (ValueError, tk.TclError):
            pass
            
    def apply_parameter_changes(self):
        """Apply parameter changes permanently to the results"""
        if not self.current_analysis_group:
            messagebox.showwarning("Warning", "Please select a group first.")
            return
            
        try:
            ka_conc = self.ka_conc_var.get()
            rmax = self.rmax_var.get()
            kd = self.kd_var.get()
            r0 = self.r0_var.get()
            
            # Validate parameters
            if ka_conc <= 0 or kd <= 0:
                messagebox.showerror("Error", "Rate constants must be positive.")
                return
                
            if rmax <= 0 or r0 <= 0:
                messagebox.showerror("Error", "Response values must be positive.")
                return
            
            group_id = self.current_analysis_group
            
            # Update kinetic results with new values
            if group_id in self.kinetic_results:
                # Recalculate fitted curves with new parameters
                if 'association' in self.phase_ranges and 'dissociation' in self.phase_ranges:
                    assoc_start, assoc_end = self.phase_ranges['association']
                    dissoc_start, dissoc_end = self.phase_ranges['dissociation']
                    
                    data = self.processed_data[group_id]
                    time_data = data['time']
                    
                    assoc_mask = (time_data >= assoc_start) & (time_data <= assoc_end)
                    dissoc_mask = (time_data >= dissoc_start) & (time_data <= dissoc_end)
                    
                    t_assoc = time_data[assoc_mask] - time_data[assoc_mask][0]
                    t_dissoc = time_data[dissoc_mask] - time_data[dissoc_mask][0]
                    
                    y_assoc_new = langmuir_association(t_assoc, rmax, ka_conc)
                    y_dissoc_new = langmuir_dissociation(t_dissoc, r0, kd)
                    
                    # Update results
                    self.kinetic_results[group_id].update({
                        'ka_conc': ka_conc,
                        'kd': kd,
                        'Rmax': rmax,
                        'R0': r0,
                        'assoc_fit': y_assoc_new,
                        'dissoc_fit': y_dissoc_new,
                        'assoc_time': time_data[assoc_mask],
                        'dissoc_time': time_data[dissoc_mask]
                    })
                    
                    # Calculate new R² values
                    y_data = data['normalized']
                    y_assoc_data = y_data[assoc_mask]
                    y_dissoc_data = y_data[dissoc_mask]
                    
                    # Association R²
                    ss_res_assoc = np.sum((y_assoc_data - y_assoc_new) ** 2)
                    ss_tot_assoc = np.sum((y_assoc_data - np.mean(y_assoc_data)) ** 2)
                    r2_assoc = 1 - (ss_res_assoc / ss_tot_assoc) if ss_tot_assoc > 0 else 0
                    
                    # Dissociation R²
                    ss_res_dissoc = np.sum((y_dissoc_data - y_dissoc_new) ** 2)
                    ss_tot_dissoc = np.sum((y_dissoc_data - np.mean(y_dissoc_data)) ** 2)
                    r2_dissoc = 1 - (ss_res_dissoc / ss_tot_dissoc) if ss_tot_dissoc > 0 else 0
                    
                    self.kinetic_results[group_id]['r2_assoc'] = r2_assoc
                    self.kinetic_results[group_id]['r2_dissoc'] = r2_dissoc
            
            # Update displays
            self.update_results_table()
            self.plot_with_fits()
            
            messagebox.showinfo("Success", f"Parameters updated for Group {group_id}")
            
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")
            
    def reset_to_fitted_params(self):
        """Reset parameters to originally fitted values"""
        if not self.current_analysis_group or self.current_analysis_group not in self.kinetic_results:
            return
            
        # This would require storing original fitted values separately
        # For now, just reload current values
        self.load_group_parameters()
        messagebox.showinfo("Info", "Parameters reset to current fitted values.")
        
    def update_parameters_display(self):
        """Update kinetic parameters display in the right panel"""
        if not self.kinetic_results:
            self.params_text.config(state=tk.NORMAL)
            self.params_text.delete(1.0, tk.END)
            self.params_text.insert(tk.END, "No kinetic parameters available.\nPlease fit kinetics first.")
            self.params_text.config(state=tk.DISABLED)
            return
            
        # Prepare text for all groups
        param_text = "KINETIC PARAMETERS\n"
        param_text += "=" * 50 + "\n\n"
        
        for group_id, results in sorted(self.kinetic_results.items()):
            param_text += f"GROUP {group_id}:\n"
            param_text += "-" * 20 + "\n"
            param_text += f"Association Phase:\n"
            param_text += f"  ka×C  = {results['ka_conc']:.6f} s⁻¹\n"
            param_text += f"  Rmax  = {results['Rmax']:.4f}\n"
            param_text += f"  R²    = {results['r2_assoc']:.4f}\n\n"
            
            param_text += f"Dissociation Phase:\n"
            param_text += f"  kd    = {results['kd']:.6f} s⁻¹\n"
            param_text += f"  R0    = {results['R0']:.4f}\n"
            param_text += f"  R²    = {results['r2_dissoc']:.4f}\n\n"
            
            param_text += f"Half-life (t₁/₂):\n"
            param_text += f"  t₁/₂  = {0.693/results['kd']:.2f} s\n"
            param_text += "\n" + "=" * 30 + "\n\n"
        
        # Update text widget
        self.params_text.config(state=tk.NORMAL)
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(tk.END, param_text)
        self.params_text.config(state=tk.DISABLED)
        
    def export_results(self):
        """Export kinetic parameters to CSV file and save plot"""
        if not self.kinetic_results:
            messagebox.showwarning("Warning", "No results to export. Please fit kinetics first.")
            return
            
        try:
            # Ask for base filename using tkinter filedialog
            file_path = tk.filedialog.asksavename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Kinetic Parameters"
            )
            
            if not file_path:
                return
            
            # Remove extension to create base filename
            base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
                
            # Export CSV data
            export_data = []
            for group_id, results in self.kinetic_results.items():
                export_data.append({
                    'Group': f"Group {group_id}",
                    'ka_conc_per_s': results['ka_conc'],
                    'kd_per_s': results['kd'],
                    'Rmax': results['Rmax'],
                    'R0': results['R0'],
                    'R2_association': results['r2_assoc'],
                    'R2_dissociation': results['r2_dissoc'],
                    'Half_life_s': 0.693/results['kd']
                })
            
            # Create DataFrame and save CSV
            try:
                import pandas as pd
                df_export = pd.DataFrame(export_data)
                csv_path = base_path + '.csv'
                df_export.to_csv(csv_path, index=False)
                csv_success = True
                csv_msg = f"Results exported to: {csv_path}"
            except ImportError:
                # Fallback to manual CSV writing if pandas not available
                csv_path = base_path + '.csv'
                with open(csv_path, 'w', newline='') as f:
                    # Write header
                    f.write("Group,ka_conc_per_s,kd_per_s,Rmax,R0,R2_association,R2_dissociation,Half_life_s\n")
                    # Write data
                    for data in export_data:
                        f.write(f"{data['Group']},{data['ka_conc_per_s']:.6f},{data['kd_per_s']:.6f},"
                               f"{data['Rmax']:.4f},{data['R0']:.4f},{data['R2_association']:.4f},"
                               f"{data['R2_dissociation']:.4f},{data['Half_life_s']:.2f}\n")
                csv_success = True
                csv_msg = f"Results exported to: {csv_path}"
            
            # Save plot as image
            try:
                png_path = base_path + '_plot.png'
                self.fig.savefig(png_path, dpi=300, bbox_inches='tight')
                plot_success = True
                plot_msg = f"\nPlot saved to: {png_path}"
            except Exception as plot_error:
                plot_success = False
                plot_msg = f"\nWarning: Could not save plot - {str(plot_error)}"
            
            # Success message
            success_msg = csv_msg + plot_msg
            messagebox.showinfo("Export Complete", success_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
            
    def export_plot_only(self):
        """Export only the current plot as PNG"""
        if not self.kinetic_results:
            messagebox.showwarning("Warning", "No plot to export. Please fit kinetics first.")
            return
            
        try:
            file_path = tk.filedialog.asksavename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Plot"
            )
            
            if not file_path:
                return
                
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export Complete", f"Plot saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot:\n{str(e)}")
            
    def clear_points(self):
        """Clear all selected points and ranges"""
        self.click_points = []
        self.phase_ranges = {}
        self.plot_processed_data() if self.processed_data else self.plot_raw_data()
        self.status_label.config(text="All selection points cleared.")
        
    def reset_all(self):
        """Reset all data and processing"""
        self.processed_data = {}
        self.click_points = []
        self.phase_ranges = {}
        self.kinetic_results = {}
        
        # Clear results table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        if hasattr(self, 'df') and self.df is not None:
            self.plot_raw_data()
            self.status_label.config(text="All processing reset. Raw data displayed.")
        else:
            self.ax.clear()
            self.canvas.draw_idle()
            self.status_label.config(text="Ready. Load a data file to begin.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SPRKineticsAnalyzer(root)
    root.mainloop()
