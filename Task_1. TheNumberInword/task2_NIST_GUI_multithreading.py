import requests
import scipy.special as sp
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import chi2
from scipy.fftpack import fft
from scipy.special import erfc
from scipy.special import gammaincc, factorial
import tkinter as tk
from tkinter import messagebox, filedialog, font
import math
import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import queue
import logging
from multiprocessing import Manager


def getRandomNumbers(randomNumbersQRNG: int):
    try:
        url = f"https://www.random.org/integers/?num={randomNumbersQRNG}&min=0&max=65535&col=1&base=10&format=plain&rnd=new"
        response = requests.get(url)
        if response.status_code == 200:
            randomNumbers = list(map(int, response.text.strip().split()))
            bitString = ''.join(format(num, '016b') for num in randomNumbers)
            bitStringLength = len(bitString)
            return randomNumbers, bitString, bitStringLength
    except requests.Timeout:
        print("Ошибка: запрос превысил время ожидания.")
        return None, None, None
    except requests.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None, None, None


def getRandomNumbersLocal(randomNumbersLocal: int):
    randomNumbers = [random.randint(0, 65535) for _ in range(randomNumbersLocal)]
    bitString = ''.join(format(num, '016b') for num in randomNumbers)
    bitStringLength = len(bitString)
    return randomNumbers, bitString, bitStringLength


def process_line(line):
    try:
        # Преобразуем строку в число
        return int(line.strip())
    except ValueError:
        # Если не удается преобразовать строку в число, возвращаем None
        return None


def process_chunk(chunk):
    data_numbers = []
    for line in chunk:
        result = process_line(line)
        if result is not None:
            data_numbers.append(result)
    return data_numbers


def to_16bit_binary(num):
    if num < 0:
        # Для отрицательных чисел применяем дополнительный код (two's complement)
        return format((1 << 16) + num, '016b')  # Добавляем 2^16 для преобразования в дополнительный код
    else:
        # Для положительных чисел стандартное двоичное представление
        return format(num, '016b')

def getRandomNumbersUser(dataNumbers: str):
    # Убираем лишние пробелы с начала и конца строки, затем разделяем на строки
    lines = dataNumbers.strip().splitlines()

    # Применяем strip() к каждой строке списка
    dataNumbers = [line.strip() for line in lines]  # Здесь мы обрабатываем строки

    try:
        # Используем пул потоков для параллельной обработки данных
        chunk_size = len(dataNumbers) // 4  # Разделяем данные на 4 части для 4 потоков
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Разбиваем данные на блоки
            chunks = [dataNumbers[i:i + chunk_size] for i in range(0, len(dataNumbers), chunk_size)]

            # Используем tqdm для отслеживания прогресса
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

            # Отображаем прогресс выполнения
            dataNumbers = []
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                    desc="Обработка данных", unit="блок"):
                dataNumbers.extend(future.result())  # Собираем результаты

        # Преобразуем числа в 16-битные строки, с учетом отрицательных чисел
        bitString = ''.join(to_16bit_binary(num) for num in dataNumbers)
        return dataNumbers, bitString, len(bitString)

    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        raise


def process_file_in_chunks(file_path):
    dataNumbers = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Разделяем строки на блоки
    chunk_size = len(lines) // 4  # 4 потока
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Разбиваем данные на блоки
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        # Используем tqdm для отслеживания прогресса
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        # Собираем результаты с прогрессом
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Обработка файла",
                                unit="блок"):
            dataNumbers.extend(future.result())  # Собираем результаты

    # Преобразуем в битовую строку
    bitString = ''.join(format(num, '016b') for num in dataNumbers)
    return dataNumbers, bitString, len(bitString)





def frequencyTest_1(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #1 False"
    numbers = np.array([1 if bit == '1' else -1 for bit in bitString])
    nth_PatrialSum = np.sum(numbers)
    observedValue = abs(nth_PatrialSum) / (bitStringLength) ** 0.5
    pValue = sp.erfc(observedValue / 2 ** 0.5)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #1 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #1 status : {testConclusion}, pValue: {round(pValue, 5)}"


def frequencyWithinABlockTest_2(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 100:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #2 False"
    blockSize = bitStringLength // 10
    numberOfBlocks = 10
    blocks = np.array([bitString[i * blockSize:(i + 1) * blockSize] for i in range(numberOfBlocks)])
    pi_values = np.array([np.sum([1 for bit in block if bit == '1']) / blockSize for block in blocks])
    chiSquare = 4 * blockSize * sum((pi_i - 0.5) ** 2 for pi_i in pi_values)
    pValue = sp.gammaincc(numberOfBlocks / 2, chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #2 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #2 status : {testConclusion}, pValue: {round(pValue, 5)}"


def runTest_3(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 100:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #3 False"
    numberOf1 = bitString.count('1')
    proportionOf1 = numberOf1 / bitStringLength
    tau = 2 / bitStringLength ** 0.5
    observedNumber = sum(1 for a, b in zip(bitString, bitString[1:]) if a != b) + 1
    expectedNumber = 2 * bitStringLength * proportionOf1 * (1 - proportionOf1)
    variance = 2 * (2 * bitStringLength) ** 0.5 * proportionOf1 * (1 - proportionOf1)
    pValue = sp.erfc(abs(observedNumber - expectedNumber) / variance)
    testConclusion = (pValue >= 0.01)
    if tau <= (proportionOf1 - 0.5):
        return f"Test_3 is not applicable. tau <= |pi - 0.5|. "
    if testConclusion:
        return f"Numbers are random. Test #3 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #3 status : {testConclusion}, pValue: {round(pValue, 5)}"


def runWithinABlockTest_4(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #4 False"
    if bitStringLength < 128:
        raise ValueError("Длина битовой строки должна быть не менее 128 бит.")
    elif bitStringLength < 6272 and bitStringLength >= 128:
        blockSize = 8
        K = 3
        N = 16
        v_count = 6
    elif bitStringLength < 750000 and bitStringLength >= 6272:
        blockSize = 128
        K = 5
        N = 49
        v_count = 7
    else:
        blockSize = 100000
        K = 6
        N = 75
        v_count = 8

    numberOfBlocks = bitStringLength // blockSize
    if numberOfBlocks == 0:
        return "Ошибка: длина битовой строки слишком мала для выбранного размера блока."

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

    max_runs = np.array([longest_run_of_ones(bitString[i * blockSize: (i + 1) * blockSize]) for i in range(N)])

    v = np.zeros(v_count, dtype=int)
    for run in max_runs:
        if run <= 1:
            v[0] += 1
        elif run == 2:
            v[1] += 1
        elif run == 3:
            v[2] += 1
        elif run >= 4 and run < (v_count + 3):  # Здесь 4, 5, 6...
            v[3] += 1
        elif run >= 8:
            v[4] += 1
        elif run >= 9:
            v[5] += 1
        elif run >= 16:
            v[6] += 1

    # Пропорции для блока длиной 8, 128 и 100000 бит
    if blockSize == 8:
        proportionOf1 = [0.2148, 0.3672, 0.2305, 0.1875]
    elif blockSize == 128:
        proportionOf1 = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    elif blockSize == 100000:
        proportionOf1 = [0.00882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    chiSquare = np.sum(
        [(v[i] - N * proportionOf1[i]) ** 2 / (N * proportionOf1[i]) if N * proportionOf1[i] > 0 else 0
         for i in range(len(proportionOf1))]
    )
    pValue = sp.gammaincc(K / 2, chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #4 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #4 status : {testConclusion}, pValue: {round(pValue, 5)}"


def binaryMatrixRankTest_5(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 32768:
            raise ValueError("Длина битовой строки должна быть не менее 32768 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #5 False"
    M = 32
    Q = 32
    numberOfBlocks = bitStringLength // (M * Q)

    prob_full_rank = 0.2888
    prob_one_less_rank = 0.5776
    prob_two_less_rank = 0.1336
    expected_full_rank = numberOfBlocks * prob_full_rank
    expected_one_less_rank = numberOfBlocks * prob_one_less_rank
    expected_two_less_rank = numberOfBlocks * prob_two_less_rank

    # Функция для вычисления ранга через SVD (быстрее, чем LU-разложение для больших матриц)
    def compute_rank(matrix):
        u, s, vh = np.linalg.svd(matrix)
        rank = np.sum(s > 1e-10)
        return rank

    # Функция для обработки одного блока данных
    def process_block(i):
        start = i * M * Q
        end = start + M * Q
        bit_block = bitString[start:end]
        matrix = np.array([int(bit) for bit in bit_block]).reshape(M, Q)
        matrix_rank = compute_rank(matrix)
        return matrix_rank

    # Параллельная обработка блоков
    full_rank_count = 0
    one_less_rank_count = 0
    two_less_rank_count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_block, i) for i in range(numberOfBlocks)]
        for future in concurrent.futures.as_completed(futures):
            matrix_rank = future.result()
            if matrix_rank == min(M, Q):
                full_rank_count += 1
            elif matrix_rank == min(M, Q) - 1:
                one_less_rank_count += 1
            else:
                two_less_rank_count += 1

    # Расчет статистики и p-значения
    chiSquare_stat = ((full_rank_count - expected_full_rank) ** 2) / expected_full_rank
    chiSquare_stat += ((one_less_rank_count - expected_one_less_rank) ** 2) / expected_one_less_rank
    chiSquare_stat += ((two_less_rank_count - expected_two_less_rank) ** 2) / expected_two_less_rank

    # Вычисление p-значения
    pValue = np.exp(-chiSquare_stat / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #5 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #5 status : {testConclusion}, pValue: {round(pValue, 5)}"


def discreteFourierTransformTest_6(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #6 False"

        # Преобразование битов в числа (-1 и 1) с помощью numpy
    numbers = np.array([1 if bit == '1' else -1 for bit in bitString])

    # Разделяем задачу на блоки, если строка большая
    num_blocks = 4  # количество блоков для параллельной обработки
    block_size = bitStringLength // num_blocks
    blocks = [numbers[i * block_size: (i + 1) * block_size] for i in range(num_blocks)]

    def process_block(block):
        # Преобразование Фурье для блока
        s = fft(block)
        modulus = np.abs(s[:len(block) // 2])
        threshold = np.sqrt(len(block) * np.log(1 / 0.05))
        count_below_threshold = np.sum(modulus < threshold)
        expected_count = 0.95 * (len(block) // 2)
        dStat = (count_below_threshold - expected_count) / np.sqrt(len(block) * 0.95 * 0.05 / 4)
        pValue = erfc(np.abs(dStat) / np.sqrt(2))
        return pValue

    # Используем ThreadPoolExecutor для параллельной обработки блоков
    with concurrent.futures.ThreadPoolExecutor() as executor:
        p_values = list(executor.map(process_block, blocks))

    # Рассчитываем итоговый результат (среднее p-значение)
    average_p_value = np.mean(p_values)
    testConclusion = (average_p_value >= 0.01)

    if testConclusion:
        return f"Numbers are random. Test #6 status : {testConclusion}, pValue: {round(average_p_value, 5)}"
    else:
        return f"Numbers are not random. Test #6 status : {testConclusion}, pValue: {round(average_p_value, 5)}"


def nonOverlappingTemplateMachineTest_7(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #7 False"
    template = "1100"
    numberOfBlocks = 10
    blockSize = bitStringLength // numberOfBlocks
    templateLength = len(template)

    # Функция для подсчета вхождений шаблона в блок
    def count_occurrences(block):
        count = 0
        j = 0
        while j <= blockSize - templateLength:
            if block[j:j + templateLength] == template:
                count += 1
                j += templateLength  # перемещаемся на длину шаблона
            else:
                j += 1  # иначе просто сдвигаем на 1
        return count

    # Разбиваем строку на блоки
    blocks = [bitString[i * blockSize:(i + 1) * blockSize] for i in range(numberOfBlocks)]

    # Используем ThreadPoolExecutor для параллельной обработки блоков
    with concurrent.futures.ThreadPoolExecutor() as executor:
        occurrences = list(executor.map(count_occurrences, blocks))

    # Математические вычисления для статистики
    M = blockSize
    m = templateLength
    mu = (M - m + 1) / (2 ** m)
    sigma_squared = M * ((1 / 2 ** m) - (2 * m - 1) / 2 ** (2 * m))

    chiSquare = sum(((W_i - mu) ** 2) / sigma_squared for W_i in occurrences)
    pValue = sp.gammaincc(numberOfBlocks / 2, chiSquare / 2)

    testConclusion = (pValue >= 0.01)

    if testConclusion:
        return f"Numbers are random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"
    else:
        return f"Numbers are not random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"


def overlappingTemplateMachineTest_8(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #8 False"
    template = "1100"
    numberOfBlocks = 10
    blockSize = bitStringLength // numberOfBlocks
    templateLength = len(template)
    K = 5  # количество степеней свободы
    lambda_val = (blockSize - templateLength + 1) / 2 ** templateLength
    pi = [np.exp(-lambda_val) * lambda_val ** i / factorial(i) for i in range(K)]
    pi.append(1 - sum(pi))  # добавляем оставшуюся вероятность

    # Функция подсчета вхождений шаблона в блок
    def count_overlapping_template(block, template):
        count = 0
        for i in range(len(block) - len(template) + 1):
            if block[i:i + len(template)] == template:
                count += 1
        return count

    # Разбиение на блоки и параллельная обработка
    blocks = [bitString[i * blockSize:(i + 1) * blockSize] for i in range(numberOfBlocks)]

    # Параллельная обработка подсчета вхождений шаблона в блоках
    with ThreadPoolExecutor() as executor:
        observed_counts = list(executor.map(lambda block: count_overlapping_template(block, template), blocks))

    # Подсчет частоты вхождений
    F = np.bincount(observed_counts, minlength=K + 1)

    # Ожидаемые значения
    expected_counts = [numberOfBlocks * p for p in pi]

    # Вычисление статистики хи-квадрат с проверкой на ноль
    chiSquare = 0
    for i in range(K + 1):
        if expected_counts[i] > 0:  # Проверяем, что ожидаемое количество больше нуля
            chiSquare += (F[i] - expected_counts[i]) ** 2 / expected_counts[i]
        else:
            # Если ожидаемое количество равно нулю, добавляем небольшой штраф для избегания деления на ноль
            chiSquare += (F[i] - expected_counts[i]) ** 2 / (
                        expected_counts[i] + 1e-10)  # можно заменить на небольшое значение

    # Вычисление p-value с помощью функции gammaincc
    pValue = sp.gammaincc(K / 2, chiSquare / 2)

    # Определение заключения теста
    testConclusion = (pValue >= 0.01)

    if testConclusion:
        return f"Numbers are random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"
    else:
        return f"Numbers are not random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"


def universalStatisticalTest_9(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 387840:
            raise ValueError("Длина битовой строки должна быть не менее 387840 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #9 False"
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
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #9 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #9 status : {testConclusion}, pValue: {round(pValue, 5)}"


def linearComplexityTest_10(bitString, bitStringLength):
    M = 500
    try:
        if len(bitString) < 500:
            raise ValueError("Длина битовой строки должна быть не менее 387840 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #10 False"
    logging.debug("Начало выполнения теста на линейную сложность.")
    N = bitStringLength // M
    expected_complexity = M / 2 + (9 + (-1) ** M) / 36

    def berlekampMasseyAlgorithm(block):
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
        logging.debug("Алгоритм Берлекемпа-Масси завершен для блока.")
        return L

    logging.debug("Разделение битовой строки на блоки.")
    blocks = np.array_split(np.array([int(bit) for bit in bitString]), N)

    # Параллельная обработка блоков
    logging.debug(f"Количество блоков: {len(blocks)}. Начинаем параллельную обработку.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        complexities = list(executor.map(lambda block: berlekampMasseyAlgorithm([int(bit) for bit in block]), blocks))
    logging.debug("Обработка всех блоков завершена.")

    T = [(complexity - expected_complexity) for complexity in complexities]
    logging.debug("Подсчет хи-квадрат.")
    chiSquare, pValue = chi2(complexities, expected_complexity)
    logging.debug(f"Статистика хи-квадрат: {chiSquare}, p-значение: {pValue}")
    testConclusion = (pValue >= 0.01)

    if testConclusion:
        return f"Numbers are random. Test #10 status : {testConclusion}, pValue: {round(pValue, 5)}"
    else:
        return f"Numbers are not random. Test #10 status : {testConclusion}, pValue: {round(pValue, 5)}"


def serialTest_11(bitString: str, bitStringLength: int):
    m = max(2, math.floor(math.log2(bitStringLength)) - 2)
    try:
        if bitStringLength < 2 ** m:
            raise ValueError(f"Длина битовой последовательности должна быть больше {2 ** m} для m = {m}.")
        if len(bitString) != bitStringLength:
            raise ValueError("Длина битовой строки не соответствует указанной длине.")
    except ValueError as e:
        return f"Ошибка: {e}"

    def countPatterns(bitString, m):
        patternCount = {bin(i)[2:].zfill(m): 0 for i in range(2 ** m)}
        for i in range(bitStringLength - m + 1):
            substring = bitString[i:i + m]
            patternCount[substring] += 1
        return patternCount

    # Параллельная обработка подсчета паттернов для V_m, V_m_1, V_m_2
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda m_val: countPatterns(bitString, m_val), [m, m - 1, m - 2]))

    V_m, V_m_1, V_m_2 = results

    N = bitStringLength
    psi_m = (sum(v ** 2 for v in V_m.values()) - N) / N
    psi_m_1 = (sum(v ** 2 for v in V_m_1.values()) - N) / N
    psi_m_2 = (sum(v ** 2 for v in V_m_2.values()) - N) / N

    delta_psi_m = psi_m - psi_m_1
    delta_psi_m_1 = psi_m_1 - psi_m_2

    df_m = 2 ** (m - 2)
    df_m_1 = 2 ** (m - 3)

    pValue_1 = sp.gammainc(df_m / 2, (delta_psi_m ** 2) / 2)
    pValue_2 = sp.gammainc(df_m_1 / 2, (delta_psi_m_1 ** 2) / 2)

    testConclusion = (pValue_1 >= 0.01 and pValue_2 >= 0.01)

    if testConclusion:
        return f"Numbers are random. Test #11 status : {testConclusion}, pValue_1: {round(pValue_1, 5)}, pValue_2:{round(pValue_2, 5)}"
    else:
        return f"Numbers are not random. Test #11 status : {testConclusion}, pValue_1: {round(pValue_1, 5)}, pValue_2:{round(pValue_2, 5)}"


def approximateEntropyTest_12(bitString, bitStringLength):
    try:
        if bitStringLength < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}"
    bits = np.array([int(bit) for bit in bitString], dtype=int)
    blocks_length = min(2, max(3, int(math.floor(math.log(bits.size, 2))) - 6))

    def patternToInt(bit_pattern):
        result = 0
        for bit in bit_pattern:
            result = (result << 1) + bit
        return result

    phi_m = []
    for iteration in range(blocks_length, blocks_length + 2):
        padded_bits = np.concatenate((bits, bits[0:iteration - 1]))
        counts = np.zeros(2 ** iteration, dtype=int)
        for i in range(2 ** iteration):
            count = 0
            for j in range(bits.size):
                if patternToInt(padded_bits[j:j + iteration]) == i:
                    count += 1
            counts[i] = count
        c_i = counts / float(bits.size)
        phi_m.append(np.sum(c_i[c_i > 0.0] * np.log(c_i[c_i > 0.0])))
    chiSquare = 2 * bits.size * (math.log(2) - (phi_m[0] - phi_m[1]))
    pValue = gammaincc(2 ** (blocks_length - 1), (chiSquare / 2.0))
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #12 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #12 status : {testConclusion}, pValue: {round(pValue, 5)}"


def cumulativeSumsTest_13(bitString: str, bitStringLength: int):
    try:
        if bitStringLength < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #13 False"
    bitString = np.array([int(bit) for bit in bitString], dtype=int)
    bitString[bitString == 0] = -1
    cumulativeSum = np.cumsum(bitString)
    cumulativeSumReverse = np.cumsum(bitString[::-1])
    forward_max = max(abs(cumulativeSum))
    backward_max = max(abs(cumulativeSumReverse))

    def computePValue(bitStringLength, max_excursion):
        sum_a = 0.0
        sum_b = 0.0
        start_k = int(math.floor((-bitStringLength / max_excursion + 1.0) / 4.0))
        end_k = int(math.floor((bitStringLength / max_excursion - 1.0) / 4.0))
        for k in range(start_k, end_k + 1):
            c = 0.5 * erfc((4.0 * k + 1.0) * max_excursion / bitStringLength ** 0.5)
            d = 0.5 * erfc((4.0 * k - 1.0) * max_excursion / bitStringLength ** 0.5)
            sum_a += +c - d
        start_k = int(math.floor(((-bitStringLength / max_excursion) - 3.0) / 4.0))
        end_k = int(math.floor((bitStringLength / max_excursion - 1.0) / 4.0))
        for k in range(start_k, end_k + 1):
            c = 0.5 * erfc((4.0 * k + 3.0) * max_excursion / bitStringLength ** 0.5)
            d = 0.5 * erfc((4.0 * k + 1.0) * max_excursion / bitStringLength ** 0.5)
            sum_b += c - d
        return 1.0 - sum_a + sum_b

    pValueForward = computePValue(bitStringLength, forward_max)
    pValueBackward = computePValue(bitStringLength, backward_max)
    testConclusion = (pValueForward >= 0.01 and pValueBackward >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #13 status: {testConclusion}, pValues: Forward = {round(pValueForward, 5)}, Backward = {round(pValueBackward, 5)}"
    else:
        return f"Numbers are not random. Test #13 status: {testConclusion}, pValues: Forward = {round(pValueForward, 5)}, Backward = {round(pValueBackward, 5)}"


def randomExcursionTest_14(bitString: str, bitStringLength: int):
    stateX = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    mappedSequence = np.where(np.array(list(bitString), dtype=int) == 0, -1, 1)
    cumulativeSum = np.cumsum(mappedSequence)
    cumulativeSum = np.insert(cumulativeSum, 0, 0)
    cumulativeSum = np.append(cumulativeSum, 0)
    positionJumps = np.where(cumulativeSum == 0)[0]
    cycleCount = len(positionJumps) - 1

    if cycleCount == 0:
        return None, 0.0, "NO CYCLES"
    stateCounter = {x: [0] * cycleCount for x in stateX}
    for cycle in range(cycleCount):
        for value in cumulativeSum[positionJumps[cycle] + 1:positionJumps[cycle + 1]]:
            if abs(value) in stateCounter:
                stateCounter[abs(value)][cycle] += 1
    pi = np.array([0.5, 0.75, 0.833333333333, 0.916666666667, 0.958333333333,
                   0.983333333333, 0.991666666667, 0.995833333333, 0.997916666667])
    results = {}
    for state in stateX:
        totalVisits = sum(stateCounter[state])
        if totalVisits == 0:
            p_value = 1.0
        else:
            chiSquared = sum(
                [(stateCounter[state][cycle] - cycleCount * pi[min(state - 1, len(pi) - 1)]) ** 2 /
                 (cycleCount * pi[min(state - 1, len(pi) - 1)]) for cycle in range(cycleCount)]
            )
            p_value = gammaincc((cycleCount / 2.0), chiSquared / 2.0)
        results[state] = p_value
    testConclusion = all(p > 0.01 for p in results.values())
    if testConclusion:
        return f"Numbers are random. Test #14 status : {testConclusion}, pValues: {results}"
    if not testConclusion:
        return f"Numbers are not  random. Test #14 status : {testConclusion}, pValues: {results}"


def randomExcursionVariantTest_15(bitString: str, bitStringLength: int):
    try:
        if bitStringLength < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #15 False"
    bitArray = np.array([int(bit) for bit in bitString])
    if bitArray.size == 0:
        return False, np.array([])
    bitArray[bitArray == 0] = -1
    sum_prime: np.ndarray = (np.concatenate((np.array([0]), np.cumsum(bitArray), np.array([0]))).astype(int))
    cycles_size: int = np.count_nonzero(sum_prime[1:] == 0)
    unique, counts = np.unique(sum_prime[abs(sum_prime) < 10], return_counts=True)
    scores = []
    for key, value in zip(unique, counts):
        if key != 0:
            scores.append(abs(value - cycles_size) / math.sqrt(2.0 * cycles_size * ((4.0 * abs(key)) - 2.0)))
    scores = np.array(scores)
    testConclusion = all(score >= 0.01 for score in scores)
    if testConclusion:
        return f"Numbers are random. Test #15 status : {testConclusion}, pValues: {scores}"
    if not testConclusion:
        return f"Numbers are not  random. Test #15 status : {testConclusion}, pValues: {scores}"


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class NISTTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NIST Test GUI")
        self.source_choice = tk.StringVar(value="None")
        self.bitString = None
        self.bitStringLength = None
        self.running_threads = 0
        self.q = queue.Queue()

        self.create_source_selection()

        # Input frame
        self.input_frame = tk.Frame(root)
        self.input_label = tk.Label(self.input_frame, text="Введите путь к файлу:")
        self.input_entry = tk.Entry(self.input_frame, state='disabled')
        self.browse_button = tk.Button(self.input_frame, text="...", command=self.load_file)
        self.input_frame.pack(pady=10)

        # Result display
        self.result_text = tk.Text(root, height=10, width=80, state='disabled')
        self.result_text.pack(pady=10)

        # Test buttons (возвращаем кнопки тестов)
        self.create_test_buttons()

        self.check_queue()  # Проверка очереди на обновления

    def create_source_selection(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        tk.Radiobutton(frame, text="RNG", variable=self.source_choice, value="RNG", command=self.toggle_input).pack(side="left")
        tk.Radiobutton(frame, text="PRNG", variable=self.source_choice, value="PRNG", command=self.toggle_input).pack(side="left")
        tk.Radiobutton(frame, text="Custom", variable=self.source_choice, value="Custom", command=self.toggle_input).pack(side="left")

    def toggle_input(self):
        choice = self.source_choice.get()
        self.input_entry.config(state='normal')
        self.browse_button.config(state='normal')

        if choice in ("RNG", "PRNG"):
            self.input_label.config(text="Введите количество чисел:")
            self.input_label.pack(side="left")
            self.input_entry.pack(side="left")
            self.browse_button.pack_forget()
        elif choice == "Custom":
            self.input_label.config(text="Введите битовую строку или выберите файл:")
            self.input_label.pack(side="left")
            self.input_entry.pack(side="left")
            self.browse_button.pack(side="left")

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filename:
            # Запуск обработки файла в другом потоке, чтобы не блокировать GUI
            threading.Thread(target=self.process_file, args=(filename,), daemon=True).start()

    def process_file(self, filename):
        logging.info(f"Начинаю обработку файла: {filename}")
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()

            dataNumbers = []
            for line in lines:
                line = line.strip()
                if line:
                    dataNumbers.append(line)

            # Обработка данных
            self.handle_data(dataNumbers)

        except Exception as e:
            logging.error(f"Ошибка при обработке файла: {e}")
            messagebox.showerror("Ошибка", f"Не удалось обработать файл: {e}")

    def handle_data(self, dataNumbers):
        logging.info("Начинаю обработку данных в handle_data")

        try:
            # Убираем лишние пробелы из каждой строки данных
            dataNumbers = [line.strip() for line in dataNumbers]

            # Передаем данные как строку, а не список
            self.dataNumbers, self.bitString, self.bitStringLength = getRandomNumbersUser("\n".join(dataNumbers))

            logging.info(f"Обработанные данные: {self.dataNumbers[:10]}... (первые 10 чисел)")

            # Добавляем результаты в интерфейс
            self.append_text_to_result(f"Количество чисел: {len(self.dataNumbers)}\n")
            self.append_text_to_result(f"Длина битовой строки: {self.bitStringLength}\n")

        except Exception as e:
            logging.error(f"Ошибка при обработке данных: {e}")
            messagebox.showerror("Ошибка", f"Произошла ошибка при обработке данных: {e}")

    def append_text_to_result(self, text):
        self.result_text.config(state='normal')
        self.result_text.insert(tk.END, text)
        self.result_text.yview(tk.END)
        self.result_text.config(state='disabled')

    def create_test_buttons(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        test_functions = [
            ("Frequency Test (1)", frequencyTest_1),
            ("Frequency Within a Block Test (2)", frequencyWithinABlockTest_2),
            ("Run Test (3)", runTest_3),
            ("Run Within a Block Test (4)", runWithinABlockTest_4),
            ("Binary Matrix Rank Test (5)", binaryMatrixRankTest_5),
            ("Discrete Fourier Transform Test (6)", discreteFourierTransformTest_6),
            ("Non-overlapping Template Matching Test (7)", nonOverlappingTemplateMachineTest_7),
            ("Overlapping Template Matching Test (8)", overlappingTemplateMachineTest_8),
            ("Universal Statistical Test (9)", universalStatisticalTest_9),
            ("Linear Complexity Test (10)", linearComplexityTest_10),
            ("Serial Test (11)", serialTest_11),
            ("Approximate Entropy Test (12)", approximateEntropyTest_12),
            ("Cumulative Sums Test (13)", cumulativeSumsTest_13),
            ("Random Excursion Test (14)", randomExcursionTest_14),
            ("Random Excursion Variant Test (15)", randomExcursionVariantTest_15)
        ]
        button_font = font.Font(family="Helvetica", size=13, weight="bold")
        max_text_width = max([button_font.measure(name) for name, _ in test_functions]) // 10
        for i, (test_name, test_function) in enumerate(test_functions):
            row, col = divmod(i, 5)
            tk.Button(button_frame, text=test_name, width=max_text_width,
                      command=lambda f=test_function: self.execute_test(f)).grid(row=row, column=col, padx=5, pady=5)
        tk.Button(button_frame, text="All", command=self.run_all_tests, width=max_text_width).grid(row=4, column=1,
                                                                                                   columnspan=1,
                                                                                                   pady=10)
        tk.Button(button_frame, text="Restart", command=self.reset, width=max_text_width).grid(row=4, column=3, pady=10)

    def run_all_tests(self):
        if self.bitString is not None and self.bitStringLength is not None:
            for test in [
                frequencyTest_1, frequencyWithinABlockTest_2, runTest_3,
                runWithinABlockTest_4, binaryMatrixRankTest_5, discreteFourierTransformTest_6,
                nonOverlappingTemplateMachineTest_7, overlappingTemplateMachineTest_8,
                universalStatisticalTest_9, linearComplexityTest_10, serialTest_11,
                approximateEntropyTest_12, cumulativeSumsTest_13, randomExcursionTest_14,
                randomExcursionVariantTest_15
            ]:
                self.execute_test(test)
        else:
            messagebox.showerror("Ошибка", "Пожалуйста, введите или выберите данные для анализа.")

    def execute_test(self, test_function):
        if self.bitString is not None and self.bitStringLength is not None:
            result = test_function(self.bitString, self.bitStringLength)
            self.append_text_to_result(f"Result: {result}\n")
            self.result_text.yview(tk.END)
        else:
            messagebox.showerror("Ошибка", "Недопустимые данные для выполнения теста.")

    def reset(self):
        self.source_choice.set("None")
        self.input_entry.delete(0, tk.END)
        self.input_entry.config(state='disabled')
        self.bitString = None
        self.bitStringLength = None
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state='disabled')

    def check_queue(self):
        """Проверка очереди каждую секунду и обновление интерфейса, если нужно"""
        try:
            while not self.q.empty():
                task, *args = self.q.get_nowait()  # Извлекаем данные из очереди
                if task == "update_input":
                    self.input_entry.delete(0, tk.END)  # Очищаем старые данные
                    self.input_entry.insert(tk.END, args[0])  # Вставляем новые данные
                elif task == "update_result":
                    processed_data, bitString, bitStringLength = args
                    self.append_text_to_result(f"Количество чисел: {len(processed_data)}\n")
                    self.append_text_to_result(f"Битовая строка: {bitString}\n")
                    self.append_text_to_result(f"Длина битовой строки: {bitStringLength}\n")
        finally:
            # Проверка очереди снова через 1000 миллисекунд (1 секунду)
            self.root.after(1000, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = NISTTestGUI(root)
    root.mainloop()
