import requests
import scipy.special as sp


def get_random_numbers():
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length={7}&type=uint16"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            random_numbers = data['data']  # Список случайных чисел
            # Преобразуем каждое число в 16-битную строку и соединяем в одну строку
            bit_string = ''.join(format(num, '016b') for num in random_numbers)
            bit_string_length = len(bit_string)
            return random_numbers, bit_string, bit_string_length  # Возвращаем строку битов
        else:
            return None, None, None
    else:
        return f"Error: HTTP {response.status_code}"""


def frequency_test(bit_string, bit_string_length):
    numberOfBits = bit_string_length
    numbers = [-1 if bit == '0' else 1 for bit in bit_string]
    nth_PatrialSum = sum(numbers)
    observedValue = abs(nth_PatrialSum) / (numberOfBits) ** 0.5
    p_Value = sp.erfc(observedValue / 2 ** 0.5)
    frequency_test_conclusion = (p_Value > 0.01)
    if frequency_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста: {frequency_test_conclusion}"
    if not frequency_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста: {frequency_test_conclusion}"


random_numbers, bit_string, bit_string_length = get_random_numbers()
print(f'Список полученных от сервера чисел {random_numbers}')
print(f'Битовое представления последовательности чисел {bit_string}')
if bit_string is not None and bit_string_length is not None:
    print(frequency_test(bit_string, bit_string_length))
else:
    print("Error: Could not perform frequency test due to invalid random number data")
