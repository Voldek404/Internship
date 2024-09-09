import requests
import scipy.special as sp


def get_random_numbers(count):
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length={count}&type=uint16"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['success']:
            random_numbers = data['data']  # Список случайных чисел
            # Преобразуем каждое число в 16-битную строку и соединяем в одну строку
            bit_string = ''.join(format(num, '016b') for num in random_numbers)
            bit_string_length = len(bit_string)
            return bit_string, bit_string_length  # Возвращаем строку битов
        else:
            return "Error: Could not retrieve data"
    else:
        return f"Error: HTTP {response.status_code}"


def frequency_test(bit_string, bit_string_length):
    numberOfBits = bit_string_length
    numbers = [-1 if bit == '0' else 1 for bit in bit_string]
    nth_PatrialSum = sum(numbers)
    observedValue = abs(nth_PatrialSum) / (numberOfBits) ** 0.5
    p_Value = sp.erfc(observedValue / 2 ** 0.5)
    frequency_test_conclusion = (p_Value > 0.01)
    return frequency_test_conclusion


# Получаем 10 случайных чисел и их битовое представление
bit_string = get_random_numbers(7)
print(bit_string)
