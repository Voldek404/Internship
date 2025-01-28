import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import logging
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Данные насосов
def generate_pump(id):
    power = random.randint(5, 30)  # Мощность от 5 до 30 кВт
    capacity = random.randint(50, 100)  # Производительность от 50 до 100 м³/ч
    efficiency = round(random.uniform(0.7, 1.0), 2)  # Эффективность от 0.7 до 1.0
    return {'type': f'Насос {id}', 'power': power, 'capacity': capacity, 'efficiency': efficiency}

# Создаем 15 насосов с различными параметрами
pumps = [generate_pump(i) for i in range(1, 8)]

# Начальный спрос на воду
total_capacity = sum([pump['capacity'] for pump in pumps])
initial_demand = total_capacity
# Очередь для моделирования спроса (с диапазоном изменения)
queue = deque()
# Заполняем очередь случайными значениями спроса в диапазоне от 50% до 150% от начального
for i in range(100):  # 100 шагов (например, для 100 минут или секунд)
    demand_random_changer = np.random.choice(np.arange(0, 1.0, 0.05))
    demand = demand_random_changer *  initial_demand  # варьируем спрос
    if total_capacity < demand:
        continue
    elif demand == 0:
        continue
    queue.append(demand)


# Функция для перераспределения нагрузки между насосами в зависимости от спроса
def calculate_load(demand):
    logging.info(f"Получен новый спрос: {demand} м³/ч")  # Логируем текущий спрос
    current_demand = demand
    selected_pumps = []

    # Сортировка насосов по эффективности (более эффективные насосы работают первыми)
    for pump in sorted(pumps, key=lambda x: x['efficiency'], reverse=True):
        if current_demand <= 0:
            break
        water_supply = pump['capacity']
        if water_supply <= current_demand:
            # Рассчитываем регулировку мощности на основе загрузки
            load = water_supply
            power_usage = pump['power'] * (load / pump['capacity'])  # Пропорциональная мощность
            selected_pumps.append(
                {'pump': pump['type'], 'power': power_usage, 'load': load, 'efficiency': pump['efficiency']})
            current_demand -= water_supply
        else:
            load = current_demand
            power_usage = pump['power'] * (load / pump['capacity'])
            selected_pumps.append(
                {'pump': pump['type'], 'power': power_usage, 'load': load, 'efficiency': pump['efficiency']})
            current_demand = 0

    total_power_consumption = sum([pump['power'] for pump in selected_pumps])
    logging.info(
        f"Распределено {len(selected_pumps)} насосов для удовлетворения спроса, суммарное потребление мощности: {total_power_consumption} кВт")

    return selected_pumps, total_power_consumption


# Создаем график
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(0, total_capacity)

# Инициализируем бары для отображения (с пустыми значениями)
bars = [] * len(pumps)
ax.set_ylabel('Загруженность (м³/ч)')
ax.set_xlabel('Тип насоса')
ax.set_title('Динамическая загрузка насосов')

# Линия для суммарного потребления мощности
line, = ax.plot([], [], color='red', linestyle='--', label="Суммарное потребление мощности")
ax.legend()

# Создаем поле для вывода логов
log_text = ax.text(0.1, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')

# Функция обновления графика для анимации
def update(frame):
    global bars  # Используем глобальную переменную для bars

    # Извлекаем следующий элемент из очереди
    if len(queue) > 0:
        demand = queue.popleft()  # получаем значение из очереди
    else:
        demand = initial_demand  # если очередь пуста, используем базовое значение

    # Рассчитываем загрузку насосов для текущего спроса
    selected_pumps, total_power_consumption = calculate_load(demand)

    # Обновляем данные для столбцов (баров)
    labels = [pump['pump'] for pump in selected_pumps]
    load_values = [pump['load'] for pump in selected_pumps]
    all_labels = [pump['type'] for pump in pumps]

    # Инициализируем загрузки для всех насосов (включая неработающие)
    all_load_values = [0] * len(pumps)  # Задаем нулевую загрузку для всех насосов

    # Обновляем загрузки для насосов, которые участвуют в текущем расчете
    for pump in selected_pumps:
        idx = next(i for i, p in enumerate(pumps) if p['type'] == pump['pump'])
        all_load_values[idx] = pump['load']

    # Если количество баров меньше чем количество насосов, создаем новые
    if len(bars) < len(all_load_values):
        for i in range(len(bars), len(all_load_values)):
            # Создаем новый бар и добавляем его в список bars
            bars.append(ax.bar(all_labels[i], all_load_values[i], color='skyblue')[0])

    for i, rect in enumerate(bars):
        if i < len(all_load_values):
            rect.set_height(all_load_values[i])  # Обновляем высоту бара
        else:
            rect.set_height(0)  # Если бар не нужен, устанавливаем его высоту в 0

    # Обновляем линию суммарного потребления мощности
    line.set_data(range(len(selected_pumps)), [total_power_consumption] * len(selected_pumps))

    # Обновляем метки оси X
    ax.set_xticks(range(len(selected_pumps)))  # индексы для оси X
    ax.set_xticklabels(labels)  # строки для меток оси X

    # Логируем обновление
    logging.info(
        f'Получен новый спрос: {demand} м³/ч", суммарное потребление мощности: {total_power_consumption} кВт')
    # Обновляем текстовое поле с логами
    log_text.set_text(
        f'Получен новый спрос: {demand} м³/ч"\nСуммарное потребление мощности: {total_power_consumption:.2f} кВт')

    return bars, line, log_text



# Анимация
try:
    ani = FuncAnimation(fig, update, frames=np.arange(0, 50, 1), blit=False,
                        interval=1000)  # Увеличиваем интервал для уменьшения нагрузки
    plt.show()
except KeyboardInterrupt:
    print("Анимация была остановлена вручную.")
