import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  
from matplotlib.figure import Figure
from solver import RealisticViscousFallSimulator  


class FallSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulate Viscous Fall")
        self.setGeometry(100, 100, 1200, 800)

        self.simulator = None 

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout() 
        main_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)

        self.input_layout = QGridLayout()
        left_layout.addLayout(self.input_layout)

        self.create_input_fields()

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.run_button)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Velocity (m/s)", "Height (m)"])
        left_layout.addWidget(self.table)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 3) 

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

    def create_input_fields(self):
        """
        Создаем поля ввода для параметров симуляции.
        """
        labels = [
            "Radius (m):", "Object Density (kg/m³):", "Medium Density (kg/m³):",
            "Viscosity (Pa·s):", "Gravity (m/s²):", "Max Time (s):", "Time Step (s):"
        ]
        self.inputs = {}
        defaults = [0.01, 1000, 998, 0.001, 9.81, 20, 0.1]

        for i, (label, default) in enumerate(zip(labels, defaults)):
            self.input_layout.addWidget(QLabel(label), i, 0)
            line_edit = QLineEdit(str(default))
            self.input_layout.addWidget(line_edit, i, 1)
            self.inputs[label] = line_edit

    def run_simulation(self):
        """
        Запуск симуляции.
        """
        try:
            radius = self.get_input_value("Radius (m):")
            object_density = self.get_input_value("Object Density (kg/m³):")
            medium_density = self.get_input_value("Medium Density (kg/m³):")
            viscosity = self.get_input_value("Viscosity (Pa·s):")
            g = self.get_input_value("Gravity (m/s²):")
            t_max = self.get_input_value("Max Time (s):")
            dt = self.get_input_value("Time Step (s):")

            self.simulator = RealisticViscousFallSimulator(
                radius, object_density, medium_density, g, t_max, dt
            )

            time, velocity, height = self.simulator.solve(viscosity)

            self.update_plots(time, velocity, height)

            self.update_table(time, velocity, height)

        except ValueError as e:
            self.show_error_message(str(e))

    def get_input_value(self, label):
        """
        Получает и проверяет значение ввода.
        """
        try:
            return float(self.inputs[label].text())
        except ValueError:
            raise ValueError(f"Неверное значение для '{label}'")

    def show_error_message(self, message):
        """
        Показывает сообщение об ошибке.
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Ошибка ввода")
        msg_box.setText(message)
        msg_box.exec()

    def update_plots(self, time, velocity, height):
        """
        Обновление графиков.
        """
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        ax1.plot(time, velocity, label="Velocity", color="blue")
        ax1.set_title("Velocity vs Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.legend()
        ax1.grid()

        ax2.plot(time, height, label="Height", color="green")
        ax2.set_title("Height vs Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Height (m)")
        ax2.legend()
        ax2.grid()

        self.figure.subplots_adjust(hspace=0.4)

        self.canvas.draw()

    def update_table(self, time, velocity, height):
        """
        Обновление таблицы.
        """
        self.table.setRowCount(len(time))
        for i in range(len(time)):
            self.table.setItem(i, 0, QTableWidgetItem(f"{time[i]:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{velocity[i]:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{height[i]:.4f}"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FallSimulatorApp()
    window.show()
    sys.exit(app.exec())
