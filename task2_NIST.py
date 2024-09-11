import requests
import scipy.special as sp


def get_random_numbers():
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length={7}&type=uint16"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            random_numbers = data['data']
            bit_string = ''.join(format(num, '016b') for num in random_numbers)
            bit_string_length = len(bit_string)
            return random_numbers, bit_string, bit_string_length
        else:
            return None, None, None
    else:
        return f"Error: HTTP {response.status_code}"""


def frequency_test_1(bit_string, bit_string_length):
    numberOfBits = bit_string_length
    numbers = [-1 if bit == '0' else 1 for bit in bit_string]
    nth_PatrialSum = sum(numbers)
    observedValue = abs(nth_PatrialSum) / (numberOfBits) ** 0.5
    p_Value = sp.erfc(observedValue / 2 ** 0.5)
    frequency_test_conclusion = (p_Value > 0.01)
    if frequency_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста frequency_test_1 : {frequency_test_conclusion}"
    if not frequency_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста frequency_test_1: {frequency_test_conclusion}"


def frequency_test_within_a_Block_2(bit_string):
    bit_string = bit_string[:105]
    block_size = 15
    number_of_blocks = 7
    pi_values = []
    for i in range(number_of_blocks):
        block = bit_string[i * block_size:(i + 1) * block_size]
        pi_i = block.count('1') / block_size
        pi_values.append(pi_i)
    chi_square = 4 * block_size * sum((pi_i - 0.5) ** 2 for pi_i in pi_values)
    p_Value = sp.gammaincc(number_of_blocks / 2, chi_square / 2)
    frequency_test_conclusion = (p_Value > 0.01)
    if frequency_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста frequency_test_within_a_Block_2: {frequency_test_conclusion}"
    if not frequency_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста frequency_test_within_a_Block_2: {frequency_test_conclusion}"


def run_test_3(bit_string, bit_string_length):
    numberOf1 = bit_string.count('1')
    proportionOf1 = numberOf1 / bit_string_length
    tau = 2 / bit_string_length ** 0.5
    observedNumber = 1
    for i in range(1, bit_string_length):
        if bit_string[i] != bit_string[i - 1]:
            observedNumber += 1
    p_Value = sp.erfc(abs(observedNumber - 2 * bit_string_length * proportionOf1 * (1 - proportionOf1)) / (2 * (2 * bit_string_length) ** 0.5 * proportionOf1 * ( 1 - proportionOf1)))
    frequency_test_conclusion = (p_Value > 0.01)
    if frequency_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста run_test_3: {frequency_test_conclusion}"
    if not frequency_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста run_test_3: {frequency_test_conclusion}"

random_numbers, bit_string, bit_string_length = get_random_numbers()
print(f'Список полученных от сервера чисел {random_numbers}')
print(f'Битовое представления последовательности чисел {bit_string}')
print(f'Длина строки {bit_string_length}')
if bit_string is not None and bit_string_length is not None:
    print(frequency_test_1(bit_string, bit_string_length))
else:
    print("Error: Could not perform frequency test due to invalid random number data")
print(frequency_test_within_a_Block_2(bit_string))
print(run_test_3(bit_string, bit_string_length))
