import tkinter as tk
from tkinter import messagebox
from optimization import solve_optimization
from modeling import analyze_data, plot_graphs
from data_loader import load_data

def run_optimization():
    data = load_data("data.json")["optimization"]
    results = solve_optimization(data)
    messagebox.showinfo("Optimization Results", f"Objective: {results['objective']}\nVariables: {results['variables']}")

def run_modeling():
    data = load_data("data.json")["modeling"]
    results = analyze_data(data["dates"], data["prices"])
    plot_graphs(results)

app = tk.Tk()
app.title("Math and Modeling GUI")

tk.Button(app, text="Run Optimization", command=run_optimization).pack(pady=10)
tk.Button(app, text="Run Modeling", command=run_modeling).pack(pady=10)

app.mainloop()
