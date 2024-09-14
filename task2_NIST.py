import requests
import scipy.special as sp
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import chi2
from scipy.fftpack import fft
from scipy.special import erfc


def get_random_numbers():
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length=8&type=uint16"
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
        return f"Error: HTTP {response.status_code}"


def get_random_numbers_local():
    random_numbers = [random.randint(0, 65535) for _ in range(8)]
    bit_string = ''.join(format(num, '016b') for num in random_numbers)
    bit_string_length = len(bit_string)
    return random_numbers, bit_string, bit_string_length


def frequency_test_1(bit_string, bit_string_length):
    numbers = [-1 if bit == '0' else 1 for bit in bit_string]
    nth_PatrialSum = sum(numbers)
    observedValue = abs(nth_PatrialSum) / (bit_string_length) ** 0.5
    p_Value = sp.erfc(observedValue / 2 ** 0.5)
    frequency_test_conclusion = (p_Value >= 0.01)
    if frequency_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста frequency_test_1 : {frequency_test_conclusion}"
    if not frequency_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста frequency_test_1: {frequency_test_conclusion}"


def frequency_test_within_a_Block_2(bit_string):
    block_size = 16
    number_of_blocks = 8
    pi_values = []
    for i in range(number_of_blocks):
        block = bit_string[i * block_size:(i + 1) * block_size]
        pi_i = block.count('1') / block_size
        pi_values.append(pi_i)
    chi_square = 4 * block_size * sum((pi_i - 0.5) ** 2 for pi_i in pi_values)
    p_Value = sp.gammaincc(number_of_blocks / 2, chi_square / 2)
    frequency_test_conclusion = (p_Value >= 0.01)
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
    p_Value = sp.erfc(abs(observedNumber - 2 * bit_string_length * proportionOf1 * (1 - proportionOf1)) / (
            2 * (2 * bit_string_length) ** 0.5 * proportionOf1 * (1 - proportionOf1)))
    run_test_conclusion = (p_Value >= 0.01)
    if run_test_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста run_test_3: {run_test_conclusion}"
    if not run_test_conclusion:
        return f"Последовательность чисел не  является случайной, статус прохождения теста run_test_3: {run_test_conclusion}"


def run_test_within_a_Block_4(bit_string, bit_string_length):
    def longest_run_of_ones(bit_string):
        max_run = 0
        current_run = 0
        for bit in bit_string:
            if bit == '1':
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run

    block_size = 8
    number_of_blocks = bit_string_length // block_size  # Число блоков
    if number_of_blocks == 0:
        return "Ошибка: длина битовой строки слишком мала для выбранного размера блока."
    bit_string = bit_string[:number_of_blocks * block_size]
    max_runs = []
    for i in range(number_of_blocks):
        block = bit_string[i * block_size:(i + 1) * block_size]
        max_run = longest_run_of_ones(block)
        max_runs.append(max_run)
    v = [0] * 7
    for run in max_runs:
        if run <= 1:
            v[0] += 1
        elif run == 2:
            v[1] += 1
        elif run == 3:
            v[2] += 1
        elif run == 4:
            v[3] += 1
        elif run == 5:
            v[4] += 1
        elif run == 6:
            v[5] += 1
        else:
            v[6] += 1
    proportionOf1 = [0.2857, 0.4286, 0.2143, 0.0714, 0.0143, 0.0014, 0.0001]  # Обновлена для блока 8 бит
    chi_square = sum(
        [(v[i] - number_of_blocks * proportionOf1[i]) ** 2 / (number_of_blocks * proportionOf1[i]) for i in range(7)])
    p_Value = sp.gammaincc(6 / 2, chi_square / 2)
    test_within_a_Block_conclusion = (p_Value >= 0.01)
    if test_within_a_Block_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста run_test_within_a_Block_4 : {test_within_a_Block_conclusion}"
    else:
        return f"Последовательность чисел не является случайной, статус прохождения теста run_test_within_a_Block_4: {test_within_a_Block_conclusion}"


def binary_matrix_rank_test_5(bit_string):
    M = 4
    Q = 4
    numberOfBitBlocks = 8

    def rank(binary_matrix):
        _, u = lu(binary_matrix, permute_l=True)
        rank = np.sum(np.abs(np.diag(u)) > 1e-10)
        return rank

    prob_full_rank = 0.2888
    prob_one_less_rank = 0.5776
    prob_two_less_rank = 0.1336
    full_rank_count = 0
    one_less_rank_count = 0
    two_less_rank_count = 0
    for i in range(numberOfBitBlocks):
        start = i * M * Q
        end = start + M * Q
        bit_block = bit_string[start:end]
        matrix = np.array([int(bit) for bit in bit_block]).reshape(M, Q)
        matrix_rank = rank(matrix)
        if matrix_rank == min(M, Q):
            full_rank_count += 1
        elif matrix_rank == min(M, Q) - 1:
            one_less_rank_count += 1
        else:
            two_less_rank_count += 1
    expected_full_rank = numberOfBitBlocks * prob_full_rank
    expected_one_less_rank = numberOfBitBlocks * prob_one_less_rank
    expected_two_less_rank = numberOfBitBlocks * prob_two_less_rank
    chi_square_stat = ((full_rank_count - expected_full_rank) ** 2) / expected_full_rank
    chi_square_stat += ((one_less_rank_count - expected_one_less_rank) ** 2) / expected_one_less_rank
    chi_square_stat += ((two_less_rank_count - expected_two_less_rank) ** 2) / expected_two_less_rank
    p_Value = 1 - chi2.cdf(chi_square_stat, df=2)
    test_within_a_Block_conclusion = (p_Value >= 0.01)
    if test_within_a_Block_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста binary_matrix_rank_test_5: {test_within_a_Block_conclusion}"
    else:
        return f"Последовательность чисел не является случайной, статус прохождения теста binary_matrix_rank_test_5: {test_within_a_Block_conclusion}"


def discrete_fourier_transform_test_6(bit_string, bit_string_length):
    bit_array = np.array([1 if bit == '1' else -1 for bit in bit_string])
    s = fft(bit_array)
    modulus = np.abs(s[0:bit_string_length // 2])
    threshold = np.sqrt(bit_string_length * np.log(1 / 0.05))
    count_below_threshold = np.sum(modulus < threshold)
    expected_count = 0.95 * (bit_string_length / 2)
    dStat = (count_below_threshold - expected_count) / np.sqrt(bit_string_length * 0.95 * 0.05 / 4)
    p_Value = erfc(np.abs(dStat) / np.sqrt(2))
    test_discrete_fourier_transform_conclusion = (p_Value >= 0.01)
    if test_discrete_fourier_transform_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста discrete_fourier_transform_test_6: {test_discrete_fourier_transform_conclusion}"
    else:
        return f"Последовательность чисел не является случайной, статус прохождения теста discrete_fourier_transform_test_6: {test_discrete_fourier_transform_conclusion}"


def non_overlapping_template_machine_test7(bit_string, bit_string_length):
    template = "111"
    block_size = bit_string_length // numberOfBlocks
    template_length = len(template)
    occurrences = []
    for i in range(numberOfBlocks):
        block = bit_string[i * block_size:(i + 1) * block_size]
        count = 0
        j = 0
        while j <= block_size - template_length:
            if block[j:j + template_length] == template:
                count += 1
                j += template_length
            else:
                j += 1
    occurrences.append(count)
    M = block_size
    m = template_length
    mu = (M - m + 1) / (2 ** m)
    sigma_squared = M * ((1 / 2 ** m) - (2 * m - 1) / 2 ** (2 * m))
    chi_square = sum(((W_i - mu) ** 2) / sigma_squared for W_i in occurrences)
    p_Value = sp.gammaincc(numberOfBlocks / 2, chi_square / 2)
    v = (p_Value >= 0.01)
    if non_overlapping_template_machine_conclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста non_overlapping_template_machine_conclusion_test_7: {non_overlapping_template_machine_conclusion}"
    else:
        return f"Последовательность чисел не является случайной, статус прохождения теста non_overlapping_template_machine_conclusion_test_7: {non_overlapping_template_machine_conclusion}"



print("Введите номер источника случайных чисел. 1 - QRNG, 2 - random library")
choise = int(input())
if choise == 1:
    random_numbers, bit_string, bit_string_length = get_random_numbers()
elif choise == 2:
    random_numbers, bit_string, bit_string_length = get_random_numbers_local()
else:
    print("Некорректный номер источника случаных чисел")

print(f'Список полученных от сервера чисел {random_numbers}')
print(f'Битовое представления последовательности чисел {bit_string}')
print(f'Длина строки {bit_string_length}')
if bit_string is not None and bit_string_length is not None:
    print(frequency_test_1(bit_string, bit_string_length))
else:
    print("Error: Could not perform frequency test due to invalid random number data")

print(frequency_test_within_a_Block_2(bit_string))

print(run_test_3(bit_string, bit_string_length))

print(run_test_within_a_Block_4(bit_string, bit_string_length))

print(binary_matrix_rank_test_5(bit_string))

print(discrete_fourier_transform_test_6(bit_string, bit_string_length))

print(non_overlapping_template_machine_test7(bit_string, bit_string_length))
