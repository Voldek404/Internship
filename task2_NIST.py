import requests
import scipy.special as sp
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import chi2
from scipy.fftpack import fft
from scipy.special import erfc


def getRandomNumbers():
    url = f"https://qrng.anu.edu.au/API/jsonI.php?length=8&type=uint16"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            randomNumbers = data['data']
            bitString = ''.join(format(num, '016b') for num in randomNumbers)
            bitStringLength = len(bitString)
            return randomNumbers, bitString, bitStringLength
        else:
            return None, None, None
    else:
        return f"Error: HTTP {response.status_code}"


def getRandomNumbersLocal():
    randomNumbers = [random.randint(0, 65535) for _ in range(8)]
    bitString = ''.join(format(num, '016b') for num in randomNumbers)
    bitStringLength = len(bitString)
    return randomNumbers, bitString, bitStringLength


def frequencyTest_1(bitString, bitStringLength):
    numbers = [-1 if bit == '0' else 1 for bit in bitString]
    nth_PatrialSum = sum(numbers)
    observedValue = abs(nth_PatrialSum) / (bitStringLength) ** 0.5
    pValue = sp.erfc(observedValue / 2 ** 0.5)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #1 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #1 status : {testConclusion}, pValue: {round(pValue, 5)}"


def frequencyWithinABlockTest_2(bitString):
    block_size = 16
    numberOfBlocks = 8
    pi_values = []
    for i in range(numberOfBlocks):
        block = bitString[i * block_size:(i + 1) * block_size]
        pi_i = block.count('1') / block_size
        pi_values.append(pi_i)
    chi_square = 4 * block_size * sum((pi_i - 0.5) ** 2 for pi_i in pi_values)
    pValue = sp.gammaincc(numberOfBlocks / 2, chi_square / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #2 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #2 status : {testConclusion}, pValue: {round(pValue, 5)}"


def runTest_3(bitString, bitStringLength):
    numberOf1 = bitString.count('1')
    proportionOf1 = numberOf1 / bitStringLength
    tau = 2 / bitStringLength ** 0.5
    observedNumber = 1
    for i in range(1, bitStringLength):
        if bitString[i] != bitString[i - 1]:
            observedNumber += 1
    pValue = sp.erfc(abs(observedNumber - 2 * bitStringLength * proportionOf1 * (1 - proportionOf1)) / (
            2 * (2 * bitStringLength) ** 0.5 * proportionOf1 * (1 - proportionOf1)))
    testConclusion = (pValue >= 0.01)
    if tau <= (proportionOf1 - 0.5):
        return f"Test_3 is not applicable. tau <= |pi - 0.5|. "
    if testConclusion:
        return f"Numbers are random. Test #3 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #3 status : {testConclusion}, pValue: {round(pValue, 5)}"


def runWithinABlockTest_4(bitString, bitStringLength):
    def longest_run_of_ones(bitString):
        max_run = 0
        current_run = 0
        for bit in bitString:
            if bit == '1':
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        return max_run

    block_size = 8
    numberOfBlocks = bitStringLength // block_size  # Число блоков
    if numberOfBlocks == 0:
        return "Ошибка: длина битовой строки слишком мала для выбранного размера блока."
    bitString = bitString[:numberOfBlocks * block_size]
    max_runs = []
    for i in range(numberOfBlocks):
        block = bitString[i * block_size:(i + 1) * block_size]
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
        [(v[i] - numberOfBlocks * proportionOf1[i]) ** 2 / (numberOfBlocks * proportionOf1[i]) for i in range(7)])
    pValue = sp.gammaincc(6 / 2, chi_square / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #4 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #4 status : {testConclusion}, pValue: {round(pValue, 5)}"


def binary_matrix_rank_test_5(bitString):
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
        bit_block = bitString[start:end]
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
    pValue = 1 - chi2.cdf(chi_square_stat, df=2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #5 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #5 status : {testConclusion}, pValue: {round(pValue, 5)}"


def discreteFourierTransformTest_6(bitString, bitStringLength):
    bit_array = np.array([1 if bit == '1' else -1 for bit in bitString])
    s = fft(bit_array)
    modulus = np.abs(s[0:bitStringLength // 2])
    threshold = np.sqrt(bitStringLength * np.log(1 / 0.05))
    count_below_threshold = np.sum(modulus < threshold)
    expected_count = 0.95 * (bitStringLength / 2)
    dStat = (count_below_threshold - expected_count) / np.sqrt(bitStringLength * 0.95 * 0.05 / 4)
    pValue = erfc(np.abs(dStat) / np.sqrt(2))
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #6 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #6 status : {testConclusion}, pValue: {round(pValue, 5)}"


def non_overlapping_template_machine_test7(bitString, bitStringLength):
    template = "111"
    numberOfBitBlocks = 8
    block_size = bitStringLength // numberOfBitBlocks
    template_length = len(template)
    occurrences = []
    for i in range(numberOfBitBlocks):
        block = bitString[i * block_size:(i + 1) * block_size]
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
    pValue = sp.gammaincc(numberOfBitBlocks / 2, chi_square / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"


def overlappingTemplateMachineTest_8(bitString, bitStringLength):
    template = "111"
    numberOfBitBlocks = 8
    block_size = bitStringLength // numberOfBitBlocks
    template_length = len(template)
    K = 5  # количество степеней свободы
    lambda_val = (block_size - template_length + 1) / 2 ** template_length
    pi = [np.exp(-lambda_val) * lambda_val ** i / np.math.factorial(i) for i in range(K)]
    pi.append(1 - sum(pi))

    def count_overlapping_template(block, template):
        count = 0
        for i in range(len(block) - len(template) + 1):
            if block[i:i + len(template)] == template:
                count += 1
        return count

    blocks = [bitString[i * block_size:(i + 1) * block_size] for i in range(numberOfBitBlocks)]
    observed_counts = [count_overlapping_template(block, template) for block in blocks]
    F = np.bincount(observed_counts, minlength=K + 1)
    expected_counts = [numberOfBitBlocks * p for p in pi]
    chi_square = sum((F[i] - expected_counts[i]) ** 2 / expected_counts[i] for i in range(K + 1))
    pValue = sp.gammaincc(K / 2, chi_square / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"


def universal_statistical_test_9(bitString, bitStringLength):
    if bitStringLength < 387840:
        raise ValueError("The bit string length must be at least 387,840 bits.")
    L = 7
    Q = 1280
    K = bitStringLength // L - Q
    table = [-1] * (2 ** L)
    for i in range(Q):
        block_value = int(bitString[i * L:(i + 1) * L], 2)
        table[block_value] = i
    sum_val = 0.0
    for i in range(Q, Q + K):
        block_value = int(bitString[i * L:(i + 1) * L], 2)
        last_position = table[block_value]
        table[block_value] = i
        if last_position != -1:
            sum_val += math.log2(i - last_position)
    fn = sum_val / K
    expected_value = 7.1836656  # Для L = 7
    variance = 3.238
    test_statistic = (fn - expected_value) / math.sqrt(variance)
    pValue = math.erfc(abs(test_statistic) / math.sqrt(2))
    universal_statistical_testConclusion = (pValue >= 0.01)
    if universal_statistical_testConclusion:
        return f"Последовательность чисел является случайной, статус прохождения теста universal_statistical_test_9: {universal_statistical_testConclusion},{pValue}"
    else:
        return f"Последовательность чисел не является случайной, статус прохождения теста universal_statistical_test_9: {universal_statistical_testConclusion}"

def linear_complexity_test_10(bitString, bitStringLength):
    M = 500  # длина блока
    if bitStringLength < M:
        raise ValueError("Длина битовой строки должна быть не менее 500 бит.")
    N = bitStringLength // M  # количество блоков
    expected_complexity = M / 2 + (9 + (-1) ** M) / 36

    # Функция для вычисления линейной сложности (алгоритм Берлекэмпа-Мэсси)
    def berlekamp_massey_algorithm(block):
        n = len(block)
        c = [0] * n
        b = [0] * n
        c[0], b[0] = 1, 1
        L, m, d = 0, -1, 0
        for i in range(n):
            d = (block[i] + sum([c[j] * block[i - j] for j in range(1, L + 1)])) % 2
            if d == 1:
                t = c[:]
                for j in range(i - m, n):
                    c[j] = (c[j] + b[j - (i - m)]) % 2
                if 2 * L <= i:
                    L = i + 1 - L
                    m = i
                    b = t
        return L

    # Разбиваем строку на блоки и вычисляем линейную сложность для каждого блока
    blocks = [bitString[i * M:(i + 1) * M] for i in range(N)]
    complexities = [berlekamp_massey_algorithm([int(bit) for bit in block]) for block in blocks]
    # Подсчет хи-квадрат статистики
    T = [(complexity - expected_complexity) for complexity in complexities]
    chi_square = sum([(t ** 2) / (M / 2) for t in T])
    pValue = sp.gammaincc(N / 2, chi_square / 2)


def serial_test_11(bitString: str, bitStringLength: int):
    # Определение длины подстроки m (например, log2(bitStringLength) - 2)
    m = max(2, math.floor(math.log2(bitStringLength)) - 2)
    # Минимальная длина последовательности для Serial Test
    if bitStringLength < 2 ** m:
        raise ValueError(f"Длина битовой последовательности должна быть больше {2 ** m} для m = {m}.")
    # Проверка на целостность битовой строки и ее соответствие заявленной длине
    if len(bitString) != bitStringLength:
        raise ValueError("Длина битовой строки не соответствует указанной длине.")

    # Вспомогательная функция для подсчета частот всех подстрок длины m
    def count_patterns(bitString, m):
        pattern_count = {}
        for i in range(2 ** m):
            pattern = bin(i)[2:].zfill(m)
            pattern_count[pattern] = 0
        for i in range(bitStringLength - m + 1):
            substring = bitString[i:i + m]
            pattern_count[substring] += 1
        return pattern_count

    # Частоты для подстрок длины m, m-1 и m-2
    V_m = count_patterns(bitString, m)
    V_m_1 = count_patterns(bitString, m - 1)
    V_m_2 = count_patterns(bitString, m - 2)

    # Подсчет статистик теста
    def compute_stat(V, m):
        N = bitStringLength
        sum_v = sum(v ** 2 for v in V.values())
        return (sum_v - N) / N

    # Вычисление статистики теста
    psi_m = compute_stat(V_m, m)
    psi_m_1 = compute_stat(V_m_1, m - 1)
    psi_m_2 = compute_stat(V_m_2, m - 2)
    # Статистика Serial Test
    delta_psi_m = psi_m - psi_m_1
    delta_psi_m_1 = psi_m_1 - psi_m_2

    # Вычисление p-value через неполное гамма-распределение
    def compute_pValue(statistic, df):
        # Вычисление гамма-функции и неполного гамма-распределения
        chi2_stat = statistic * (bitStringLength / 2)
        return 1 - sp.gammainc(df / 2, chi2_stat / 2)

    df_m = 2 ** m - 1
    df_m_1 = 2 ** (m - 1) - 1
    pValue_1 = compute_pValue(delta_psi_m, df_m)
    pValue_2 = compute_pValue(delta_psi_m_1, df_m_1)
    return pValue_1, pValue_2


def approximateEntropyTest_12(bitString, bitStringLength, m=2):
    if bitStringLength < 100:
        raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    def calculate_frequency(pattern_length):
        counts = {}
        for i in range(bitStringLength):
            pattern = bitString[i:i + pattern_length]
            if len(pattern) < pattern_length:
                pattern += bitString[:pattern_length - len(pattern)]
            if pattern in counts:
                counts[pattern] += 1
            else:
                counts[pattern] = 1
        total_patterns = bitStringLength
        for key in counts:
            counts[key] /= total_patterns
        return counts
    P_m = calculate_frequency(m)
    P_m1 = calculate_frequency(m + 1)
    def calculate_entropy(pattern_counts):
        return sum([-p * math.log(p, 2) for p in pattern_counts.values() if p > 0])
    entropy_m = calculate_entropy(P_m)
    entropy_m1 = calculate_entropy(P_m1)
    approx_entropy = entropy_m - entropy_m1
    chi_square = 2 * bitStringLength * (math.log(2) - approx_entropy)
    pValue = sp.gammaincc(2 ** (m - 1), chi_square / 2)
    

print("Введите номер источника случайных чисел. 1 - QRNG, 2 - random library")
choise = int(input())
if choise == 1:
    randomNumbers, bitString, bitStringLength = getRandomNumbers()
elif choise == 2:
    randomNumbers, bitString, bitStringLength = getRandomNumbersLocal()
else:
    print("Некорректный номер источника случаных чисел")

print(f'Список полученных от сервера чисел {randomNumbers}')
print(f'Битовое представления последовательности чисел {bitString}')
print(f'Длина строки {bitStringLength}')
if bitString is not None and bitStringLength is not None:
    print(frequencyTest_1(bitString, bitStringLength))
else:
    print("Error: Could not perform frequency test due to invalid random number data")

print(frequencyWithinABlockTest_2(bitString))

print(runTest_3(bitString, bitStringLength))

print(runWithinABlockTest_4(bitString, bitStringLength))

print(binary_matrix_rank_test_5(bitString))

print(discreteFourierTransformTest_6(bitString, bitStringLength))

print(non_overlapping_template_machine_test7(bitString, bitStringLength))

print(overlappingTemplateMachineTest_8(bitString, bitStringLength))
