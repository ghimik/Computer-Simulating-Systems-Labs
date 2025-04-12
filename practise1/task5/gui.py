import tkinter as tk
from tkinter import messagebox
import numpy as np
from visualization import plot_solution

def parse_params(params_str):
    """
    Преобразует строку вида "a=1.0, b=0.1" в словарь {'a':1.0, 'b':0.1}.
    Если строка пуста – возвращает пустой словарь.
    """
    params = {}
    params_str = params_str.strip()
    if params_str:
        # Разделяем по запятым
        pairs = params_str.split(',')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=')
                key = key.strip()
                try:
                    # Преобразуем значение в float (можно дополнить для более сложных типов)
                    value = float(value)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Невозможно преобразовать значение параметра {key}")
                    return None
                params[key] = value
            else:
                messagebox.showerror("Ошибка", "Неверный формат параметров. Используйте формат a=1.0, b=0.1")
                return None
    return params

def parse_list(text):
    """
    Преобразует строку, разделённую запятыми, в список, отбрасывая лишние пробелы.
    Например: "t, y" -> ['t', 'y'].
    """
    return [elem.strip() for elem in text.split(',') if elem.strip()]

def parse_initial_conditions(ic_str):
    """
    Преобразует строку, разделённую запятыми, в список чисел.
    Например: "1, 0" -> [1, 0].
    """
    ic = []
    ic_str = ic_str.strip()
    if ic_str:
        parts = ic_str.split(',')
        for part in parts:
            try:
                ic.append(float(part))
            except ValueError:
                messagebox.showerror("Ошибка", f"Невозможно преобразовать '{part}' в число для начальных условий")
                return None
    return ic

def build_gui():
    root = tk.Tk()
    root.title("ODE Solver GUI")
    root.geometry("600x500")

    # Рамка для уравнений
    frame_eq = tk.Frame(root)
    frame_eq.pack(padx=10, pady=5, fill="x")
    tk.Label(frame_eq, text="Уравнения (каждое с новой строки):").pack(anchor="w")
    text_eq = tk.Text(frame_eq, height=6)
    text_eq.pack(fill="both", padx=5, pady=5)
    # Пример: y' = y**2 - t*y
    text_eq.insert("end", "y' = y**2 - t*y\n")   # можно убрать или оставить пример

    # Рамка для переменных
    frame_vars = tk.Frame(root)
    frame_vars.pack(padx=10, pady=5, fill="x")
    tk.Label(frame_vars, text="Переменные (через запятую, первый всегда независимая, например, t, y):").pack(anchor="w")
    entry_vars = tk.Entry(frame_vars)
    entry_vars.pack(fill="x", padx=5, pady=5)
    entry_vars.insert(0, "t, y")

    # Рамка для параметров
    frame_params = tk.Frame(root)
    frame_params.pack(padx=10, pady=5, fill="x")
    tk.Label(frame_params, text="Параметры (формат: a=1.0, b=0.1):").pack(anchor="w")
    entry_params = tk.Entry(frame_params)
    entry_params.pack(fill="x", padx=5, pady=5)

    # Рамка для начальных условий
    frame_ic = tk.Frame(root)
    frame_ic.pack(padx=10, pady=5, fill="x")
    tk.Label(frame_ic, text="Начальные условия (через запятую, например, 1, 0):").pack(anchor="w")
    entry_ic = tk.Entry(frame_ic)
    entry_ic.pack(fill="x", padx=5, pady=5)

    # Рамка для интервала интегрирования
    frame_tspan = tk.Frame(root)
    frame_tspan.pack(padx=10, pady=5, fill="x")
    tk.Label(frame_tspan, text="Интервал времени (t0, tf):").pack(anchor="w")
    entry_t0 = tk.Entry(frame_tspan, width=10)
    entry_t0.pack(side="left", padx=5, pady=5)
    entry_t0.insert(0, "0")
    entry_tf = tk.Entry(frame_tspan, width=10)
    entry_tf.pack(side="left", padx=5, pady=5)
    entry_tf.insert(0, "10")

    def on_plot():
        # Чтение введённых данных
        eq_text = text_eq.get("1.0", "end").strip()
        if not eq_text:
            messagebox.showerror("Ошибка", "Введите уравнения")
            return
        equations = [line.strip() for line in eq_text.splitlines() if line.strip()]
        
        vars_list = parse_list(entry_vars.get())
        if not vars_list:
            messagebox.showerror("Ошибка", "Введите переменные")
            return
        params = parse_params(entry_params.get())
        if params is None:
            return
        ic = parse_initial_conditions(entry_ic.get())
        if ic is None:
            return
        try:
            t0 = float(entry_t0.get())
            tf = float(entry_tf.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения для интервала времени")
            return
        
        t_span = (t0, tf)
        # Вызов функции построения графика
        try:
            plot_solution(equations, vars_list, params, t_span, ic)
        except Exception as e:
            messagebox.showerror("Ошибка", f"При построении графика произошла ошибка:\n{e}")

    btn_plot = tk.Button(root, text="Построить график", command=on_plot)
    btn_plot.pack(padx=10, pady=10)

    # Запуск главного цикла
    root.mainloop()

if __name__ == '__main__':
    build_gui()
