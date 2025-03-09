import tkinter as tk
from tkinter import ttk, messagebox
from generator import LemerGenerator
from task1_pure_python import SingleServerWithBlocking
from task1_simpy import simulate as simulate_single
from task1_simpy import theoretical_statistics as theoretical_statistics_single
from task2_pure_python import MultiServerQueue
from task2_simpy import simulate as simulate_multi
from task2_simpy import theoretical_statistics as theoretical_statistics_multi

class CMOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("СМО Симулятор")
        self.root.geometry("900x400")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.single_server_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_server_tab, text="Одноканальная СМО")
        self.setup_single_server_tab()

        self.multi_server_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.multi_server_tab, text="Многоканальная СМО")
        self.setup_multi_server_tab()

    def setup_single_server_tab(self):
        """Настройка вкладки для одноканальной СМО."""
        tk.Label(self.single_server_tab, text="Интенсивность (λ):").grid(row=0, column=0, padx=10, pady=5)
        self.single_lambda_entry = tk.Entry(self.single_server_tab)
        self.single_lambda_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.single_server_tab, text="Время обслуживания:").grid(row=1, column=0, padx=10, pady=5)
        self.single_service_time_entry = tk.Entry(self.single_server_tab)
        self.single_service_time_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.single_server_tab, text="Время симуляции:").grid(row=2, column=0, padx=10, pady=5)
        self.single_max_time_entry = tk.Entry(self.single_server_tab)
        self.single_max_time_entry.grid(row=2, column=1, padx=10, pady=5)

        self.single_run_button = tk.Button(self.single_server_tab, text="Запустить симуляцию", command=self.run_single_simulation)
        self.single_run_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.single_result_tree = ttk.Treeview(self.single_server_tab, columns=("Type", "P_reject", "P_service", "Ratio"), show="headings")
        self.single_result_tree.heading("Type", text="Тип")
        self.single_result_tree.heading("P_reject", text="Вероятность отказа")
        self.single_result_tree.heading("P_service", text="Вероятность обслуживания")
        self.single_result_tree.heading("Ratio", text="Отношение обслуж/отказ")
        self.single_result_tree.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def setup_multi_server_tab(self):
        """Настройка вкладки для многоканальной СМО."""
        tk.Label(self.multi_server_tab, text="Интенсивность (λ):").grid(row=0, column=0, padx=10, pady=5)
        self.multi_lambda_entry = tk.Entry(self.multi_server_tab)
        self.multi_lambda_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.multi_server_tab, text="Время обслуживания:").grid(row=1, column=0, padx=10, pady=5)
        self.multi_service_time_entry = tk.Entry(self.multi_server_tab)
        self.multi_service_time_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.multi_server_tab, text="Время симуляции:").grid(row=2, column=0, padx=10, pady=5)
        self.multi_max_time_entry = tk.Entry(self.multi_server_tab)
        self.multi_max_time_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.multi_server_tab, text="Количество серверов:").grid(row=3, column=0, padx=10, pady=5)
        self.multi_num_servers_entry = tk.Entry(self.multi_server_tab)
        self.multi_num_servers_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.multi_server_tab, text="Ограничение очереди:").grid(row=4, column=0, padx=10, pady=5)
        self.multi_queue_capacity_entry = tk.Entry(self.multi_server_tab)
        self.multi_queue_capacity_entry.grid(row=4, column=1, padx=10, pady=5)

        self.multi_run_button = tk.Button(self.multi_server_tab, text="Запустить симуляцию", command=self.run_multi_simulation)
        self.multi_run_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.multi_result_tree = ttk.Treeview(self.multi_server_tab, columns=("Type", "P_reject", "P_service", "Avg_queue"), show="headings")
        self.multi_result_tree.heading("Type", text="Тип")
        self.multi_result_tree.heading("P_reject", text="Вероятность отказа")
        self.multi_result_tree.heading("P_service", text="Вероятность обслуживания")
        self.multi_result_tree.heading("Avg_queue", text="Средняя длина очереди")
        self.multi_result_tree.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def run_single_simulation(self):
        """Запуск симуляции для одноканальной СМО."""
        try:
            lambda_value = float(self.single_lambda_entry.get())
            service_time = float(self.single_service_time_entry.get())
            max_time = float(self.single_max_time_entry.get())

            seed = 42  
            generator = LemerGenerator(seed)

            custom_smo = SingleServerWithBlocking(generator, lambda_value, service_time)
            custom_smo.simulate(max_time)
            custom_reject, custom_service, custom_ratio = custom_smo.get_statistics()

            simpy_stats = simulate_single(lambda_value, service_time, max_time, generator)
            total_clients = simpy_stats['served'] + simpy_stats['rejected']
            simpy_reject = simpy_stats['rejected'] / total_clients if total_clients > 0 else 0
            simpy_service = simpy_stats['served'] / total_clients if total_clients > 0 else 0
            simpy_ratio = simpy_stats['served'] / simpy_stats['rejected'] if simpy_stats['rejected'] > 0 else float('inf')

            theory_reject, theory_service, theory_ratio = theoretical_statistics_single(lambda_value, service_time)

            for row in self.single_result_tree.get_children():
                self.single_result_tree.delete(row)

            self.single_result_tree.insert("", "end", values=("Кастомная", f"{custom_reject:.4f}", f"{custom_service:.4f}", f"{custom_ratio:.4f}"))
            self.single_result_tree.insert("", "end", values=("Simpy", f"{simpy_reject:.4f}", f"{simpy_service:.4f}", f"{simpy_ratio:.4f}"))
            self.single_result_tree.insert("", "end", values=("Теория", f"{theory_reject:.4f}", f"{theory_service:.4f}", f"{theory_ratio:.4f}"))

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    def run_multi_simulation(self):
        """Запуск симуляции для многоканальной СМО."""
        try:
            lambda_value = float(self.multi_lambda_entry.get())
            service_time = float(self.multi_service_time_entry.get())
            max_time = float(self.multi_max_time_entry.get())
            num_servers = int(self.multi_num_servers_entry.get())
            queue_capacity = self.multi_queue_capacity_entry.get()
            queue_capacity = int(queue_capacity) if queue_capacity else None

            seed = 42 
            generator = LemerGenerator(seed)

            custom_smo = MultiServerQueue(generator, lambda_value, service_time, num_servers, queue_capacity)
            custom_smo.simulate(max_time)
            custom_reject, custom_service, custom_avg_queue = custom_smo.get_statistics()

            simpy_stats = simulate_multi(lambda_value, service_time, num_servers, max_time, generator, queue_capacity)
            total_clients = simpy_stats['served'] + simpy_stats['rejected']
            simpy_reject = simpy_stats['rejected'] / total_clients if total_clients > 0 else 0
            simpy_service = simpy_stats['served'] / total_clients if total_clients > 0 else 0
            simpy_avg_queue = simpy_stats['avg_queue_length']

            theory_reject, theory_service, theory_avg_queue = theoretical_statistics_multi(lambda_value, service_time, num_servers, queue_capacity)

            for row in self.multi_result_tree.get_children():
                self.multi_result_tree.delete(row)

            self.multi_result_tree.insert("", "end", values=("Кастомная", f"{custom_reject:.4f}", f"{custom_service:.4f}", f"{custom_avg_queue:.4f}"))
            self.multi_result_tree.insert("", "end", values=("Simpy", f"{simpy_reject:.4f}", f"{simpy_service:.4f}", f"{simpy_avg_queue:.4f}"))
            self.multi_result_tree.insert("", "end", values=("Теория", f"{theory_reject:.4f}", f"{theory_service:.4f}", f"{theory_avg_queue:.4f}"))

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CMOApp(root)
    root.mainloop()