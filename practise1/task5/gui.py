import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QTextEdit, QLineEdit, 
                              QPushButton, QMessageBox)

from visualization import plot_solution

def parse_params(params_str):
    """
    Преобразует строку вида "a=1.0, b=0.1" в словарь {'a':1.0, 'b':0.1}.
    Если строка пуста – возвращает пустой словарь.
    """
    params = {}
    params_str = params_str.strip()
    if params_str:
        pairs = params_str.split(',')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=')
                key = key.strip()
                try:
                    value = float(value)
                except ValueError:
                    QMessageBox.critical(None, "Ошибка", f"Невозможно преобразовать значение параметра {key}")
                    return None
                params[key] = value
            else:
                QMessageBox.critical(None, "Ошибка", "Неверный формат параметров. Используйте формат a=1.0, b=0.1")
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
                QMessageBox.critical(None, "Ошибка", f"Невозможно преобразовать '{part}' в число для начальных условий")
                return None
    return ic

class ODESolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODE Solver GUI")
        self.setGeometry(100, 100, 600, 500)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        
        # Уравнения
        self.eq_label = QLabel("Уравнения (каждое с новой строки):")
        self.text_eq = QTextEdit()
        self.text_eq.setPlainText("y' = y**2 - t*y\n")
        self.text_eq.setMinimumHeight(120)
        
        # Переменные
        self.vars_label = QLabel("Переменные (через запятую, первый всегда независимая, например, t, y):")
        self.entry_vars = QLineEdit("t, y")
        
        # Параметры
        self.params_label = QLabel("Параметры (формат: a=1.0, b=0.1):")
        self.entry_params = QLineEdit()
        
        # Начальные условия
        self.ic_label = QLabel("Начальные условия (через запятую, например, 1, 0):")
        self.entry_ic = QLineEdit()
        self.entry_ic.setText("1")
        
        # Интервал времени
        self.tspan_label = QLabel("Интервал времени (t0, tf):")
        self.tspan_layout = QHBoxLayout()
        self.entry_t0 = QLineEdit("0")
        self.entry_t0.setMaximumWidth(100)
        self.entry_tf = QLineEdit("10")
        self.entry_tf.setMaximumWidth(100)
        self.tspan_layout.addWidget(self.entry_t0)
        self.tspan_layout.addWidget(self.entry_tf)
        self.tspan_layout.addStretch()
        
        # Кнопка построения графика
        self.btn_plot = QPushButton("Построить график")
        self.btn_plot.clicked.connect(self.on_plot)
        
        # Добавление виджетов в основной layout
        self.layout.addWidget(self.eq_label)
        self.layout.addWidget(self.text_eq)
        self.layout.addWidget(self.vars_label)
        self.layout.addWidget(self.entry_vars)
        self.layout.addWidget(self.params_label)
        self.layout.addWidget(self.entry_params)
        self.layout.addWidget(self.ic_label)
        self.layout.addWidget(self.entry_ic)
        self.layout.addWidget(self.tspan_label)
        self.layout.addLayout(self.tspan_layout)
        self.layout.addWidget(self.btn_plot)
        self.layout.addStretch()
    
    def on_plot(self):
        eq_text = self.text_eq.toPlainText().strip()
        if not eq_text:
            QMessageBox.critical(self, "Ошибка", "Введите уравнения")
            return
        equations = [line.strip() for line in eq_text.splitlines() if line.strip()]
        
        vars_list = parse_list(self.entry_vars.text())
        if not vars_list:
            QMessageBox.critical(self, "Ошибка", "Введите переменные")
            return
        params = parse_params(self.entry_params.text())
        if params is None:
            return
        ic = parse_initial_conditions(self.entry_ic.text())
        if ic is None:
            return
        try:
            t0 = float(self.entry_t0.text())
            tf = float(self.entry_tf.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения для интервала времени")
            return
        
        t_span = (t0, tf)
        try:
            plot_solution(equations, vars_list, params, t_span, ic)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"При построении графика произошла ошибка:\n{e}")

def build_gui():
    app = QApplication(sys.argv)
    window = ODESolverGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    build_gui()