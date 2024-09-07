import tkinter as tk
from tkinter import font, messagebox
import keyboard
import pyperclip
import time
import threading
from pynput import keyboard as pynput_keyboard

def write_birds_count(number):
    units = {1: 'одна', 2: 'две', 3: 'три', 4: 'четыре', 5: 'пять', 6: 'шесть', 7: 'семь', 8: 'восемь', 9: 'девять'}
    units_exclusion = {1: 'один', 2: 'два'}
    teens = {10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать', 14: 'четырнадцать', 15: 'пятнадцать',
             16: 'шестнадцать', 17: 'семнадцать', 18: 'восемнадцать', 19: 'девятнадцать'}
    tens = {20: 'двадцать', 30: 'тридцать', 40: 'сорок', 50: 'пятьдесят', 60: 'шестьдесят', 70: 'семьдесят',
            80: 'восемьдесят', 90: 'девяносто'}
    hundreds = {100: 'сто', 200: 'двести', 300: 'триста', 400: 'четыреста', 500: 'пятьсот', 600: 'шестьсот',
                700: 'семьсот', 800: 'восемьсот', 900: 'девятьсот'}
    bird_forms = ['сорока', 'сороки', 'сорок']

    thousands_forms = ['тысяча', 'тысячи', 'тысяч']
    millions_forms = ['миллион', 'миллиона', 'миллионов']
    billions_forms = ['миллиард', 'миллиарда', 'миллиардов']

    if number < 0 or number > 10 ** 9 or isinstance(number, float):
        return "Invariant Value"

    def get_form(n, forms):
        if 11 <= n % 100 <= 19:
            return forms[2]
        last_digit = n % 10
        if last_digit == 1:
            return forms[0]
        elif 2 <= last_digit <= 4:
            return forms[1]
        else:
            return forms[2]

    def number_to_words(n, is_thousand=False, is_million=False):
        if n == 0:
            return ''
        elif n < 10:
            if is_thousand and n in units:
                return units[n]
            elif is_million and n in units_exclusion:
                return units_exclusion[n]
            return units[n]
        elif n < 20:
            return teens[n]
        elif n < 100:
            return tens[n // 10 * 10] + (f' {units[n % 10]}' if n % 10 != 0 else '')
        elif n < 1000:
            if n in hundreds:
                return hundreds[n]
            return hundreds[n // 100 * 100] + (
                f' {number_to_words(n % 100, is_thousand, is_million)}' if n % 100 != 0 else '')

    result = []

    chunks = [
        (10 ** 9, billions_forms),
        (10 ** 6, millions_forms, True),
        (10 ** 3, thousands_forms)
    ]

    for divisor, forms, *rest in chunks:
        chunk = number // divisor
        if chunk > 0:
            is_thousand = (divisor == 10 ** 3)
            is_million = (divisor == 10 ** 6)
            result.append(f'{number_to_words(chunk, is_thousand, is_million)} {get_form(chunk, forms)}')
        number %= divisor

    if number > 0:
        result.append(number_to_words(number))

    if len(result) == 0:
        result.append('Ноль')

    bird_form = get_form(number, bird_forms)
    result.append(bird_form)

    return ' '.join(result).capitalize() + '.'

def clear_clipboard():
    pyperclip.copy("")  # Очищаем буфер обмена

def periodic_clear_clipboard():
    global last_clipboard
    last_clipboard = ""
    while True:
        time.sleep(1)  # Очищаем буфер обмена каждую секунду
        clear_clipboard()

def check_clipboard():
    global last_clipboard
    try:
        current_clipboard = pyperclip.paste()
        if current_clipboard != last_clipboard:
            last_clipboard = current_clipboard
            messagebox.showwarning("Предупреждение", "Для использования буфера обмена закройте программу.")
            clear_clipboard()
    except Exception as e:
        print(f"Ошибка проверки буфера обмена: {e}")

def clipboard_monitor():
    while True:
        time.sleep(0.1)  # Проверяем буфер обмена каждые 0.1 секунды
        check_clipboard()

def delayed_clear_clipboard():
    time.sleep(1)
    clear_clipboard()

def on_print_screen(key=None):
    threading.Thread(target=delayed_clear_clipboard).start()

def on_activate_snip_and_sketch():
    threading.Thread(target=delayed_clear_clipboard).start()

def on_press(key):
    try:
        if key == pynput_keyboard.Key.print_screen:
            on_print_screen()
        elif key == pynput_keyboard.KeyCode.from_char('s') and pynput_keyboard.Key.cmd in pressed_keys:
            on_activate_snip_and_sketch()
    except AttributeError:
        pass

def on_release(key):
    if key == pynput_keyboard.Key.cmd:
        pressed_keys.discard(key)

pressed_keys = set()

def display_result(event=None):
    try:
        number = int(entry.get())
        result = write_birds_count(number)
        result_label.config(text=result, fg="white")  # Показать результат
    except ValueError:
        result_label.config(text="Введите целое неотрицательное число", fg="white")

def hide_text(event=None):
    if event and event.keysym == "Return":
        return  # Игнорируем Enter
    result_label.config(fg=root["bg"])  # Скрываем текст

def show_text(event=None):
    result_label.config(fg="white")  # Восстанавливаем текст

def on_focus_out(event):
    result_label.config(fg=root["bg"])  # Скрываем текст при потере фокуса

def on_focus_in(event):
    result_label.config(fg="white")  # Восстанавливаем текст при возвращении фокуса

# Создание основного окна
root = tk.Tk()
root.title("Число сорок прописью")
root.configure(bg="#121400")

# Настройка шрифта
font_title = tk.font.Font(family="Times New Roman", size=24, weight="bold")
font_result = tk.font.Font(family="Times New Roman", size=18, weight="bold")

# Заголовок
title_label = tk.Label(root, text="Число сорок прописью", bg="#121405", fg="white", font=font_title)
title_label.pack(pady=20)

# Поле ввода
entry_label = tk.Label(root, text="Введите целое неотрицательное число", bg="#121400", fg="white", font=font_result)
entry_label.pack(pady=10)

entry = tk.Entry(root, font=font_result)
entry.pack(pady=10)

# Кнопка для выполнения преобразования
convert_button = tk.Button(root, text="Преобразовать", command=display_result, font=font_result)
convert_button.pack(pady=20)

# Поле вывода
result_label = tk.Label(root, text="", bg="#121400", fg="white", font=font_result)
result_label.pack(pady=20)

# Привязка событий
entry.bind("<Return>", display_result)
root.bind("<ButtonPress>", hide_text)  # Скрытие текста при нажатии кнопок мыши
root.bind("<ButtonRelease>", show_text)  # Восстановление текста при отпускании кнопок мыши
root.bind("<KeyPress>", hide_text)  # Скрытие текста при нажатии клавиш (кроме Enter)
root.bind("<KeyRelease>", show_text)  # Восстановление текста при отпускании клавиш
root.bind("<FocusOut>", on_focus_out)  # Скрытие текста при потере фокуса
root.bind("<FocusIn>", on_focus_in)  # Восстановление текста при возвращении фокуса

# Запуск проверки Print Screen и Snip & Sketch
listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Запуск потоков для очистки буфера и мониторинга
last_clipboard = ""
threading.Thread(target=periodic_clear_clipboard, daemon=True).start()
threading.Thread(target=clipboard_monitor, daemon=True).start()

root.mainloop()
