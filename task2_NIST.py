import requests
import scipy.special as sp
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import chi2
from scipy.fftpack import fft
from scipy.special import erfc
import math


def getRandomNumbers(randomNumbersQRNG: int):
    try:
        url = f"https://www.random.org/integers/?num={randomNumbersQRNG}&min=0&max=65535&col=1&base=10&format=plain&rnd=new"
        response = requests.get(url)
        if response.status_code == 200:
            randomNumbers = list(map(int, response.text.strip().split()))
            bitString = ''.join(format(num, '016b') for num in randomNumbers)
            bitStringLength = len(bitString)
            return randomNumbers, bitString, bitStringLength
        else:
            print("Ошибка при получении данных:", data.get('error', 'Неизвестная ошибка'))
            return None, None, None
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


def getRandomNumbersUser(dataNumbers: str):
    try:
        dataNumbers = [int(num.strip()) for num in dataNumbers.split(',')]
        bitString = ''.join(format(int(num), '016b') for num in dataNumbers)
        return dataNumbers, bitString, len(bitString)
    except ValueError:
        pass
    if all(char in '01' for char in dataNumbers):
        blockSize = 16
        bitString = dataNumbers
        dataNumbers = [int(bitString[i:i + blockSize], 2) for i in range(0, len(bitString), blockSize)]
        return dataNumbers, bitString, len(bitString)
    else:
        raise ValueError("Входные данные не являются ни последовательностью чисел, ни битовой строкой.")


def frequencyTest_1(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #1 False"
    numbers = [-1 if bit == '0' else 1 for bit in bitString]
    nth_PatrialSum = sum(numbers)
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
    pi_values = []
    for i in range(numberOfBlocks):
        block = bitString[i * blockSize:(i + 1) * blockSize]
        pi_i = block.count('1') / blockSize
        pi_values.append(pi_i)
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

    max_runs = []
    for i in range(N):
        block = bitString[i * blockSize:(i + 1) * blockSize]
        max_run = longest_run_of_ones(block)
        max_runs.append(max_run)

    v = [0] * v_count
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

    chiSquare = sum(
        [(v[i] - N * proportionOf1[i]) ** 2 / (N * proportionOf1[i]) if N * proportionOf1[i] > 0 else 0 for
         i in range(len(proportionOf1))]
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
    for i in range(numberOfBlocks):
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
    expected_full_rank = numberOfBlocks * prob_full_rank
    expected_one_less_rank = numberOfBlocks * prob_one_less_rank
    expected_two_less_rank = numberOfBlocks * prob_two_less_rank
    chiSquare_stat = ((full_rank_count - expected_full_rank) ** 2) / expected_full_rank
    chiSquare_stat += ((one_less_rank_count - expected_one_less_rank) ** 2) / expected_one_less_rank
    chiSquare_stat += ((two_less_rank_count - expected_two_less_rank) ** 2) / expected_two_less_rank
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
    numbers = np.array([1 if bit == '1' else -1 for bit in bitString])
    s = fft(numbers)
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
    occurrences = []
    for i in range(numberOfBlocks):
        block = bitString[i * blockSize:(i + 1) * blockSize]
        count = 0
        j = 0
        while j <= blockSize - templateLength:
            if block[j:j + templateLength] == template:
                count += 1
                j += templateLength
            else:
                j += 1
    occurrences.append(count)
    M = blockSize
    m = templateLength
    mu = (M - m + 1) / (2 ** m)
    sigma_squared = M * ((1 / 2 ** m) - (2 * m - 1) / 2 ** (2 * m))
    chiSquare = sum(((W_i - mu) ** 2) / sigma_squared for W_i in occurrences)
    pValue = sp.gammaincc(numberOfBlocks / 2, chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #7 status : {testConclusion}, pValue: {round(pValue, 5)}"


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
    pi = [np.exp(-lambda_val) * lambda_val ** i / np.math.factorial(i) for i in range(K)]
    pi.append(1 - sum(pi))

    def count_overlapping_template(block, template):
        count = 0
        for i in range(len(block) - len(template) + 1):
            if block[i:i + len(template)] == template:
                count += 1
        return count

    blocks = [bitString[i * blockSize:(i + 1) * blockSize] for i in range(numberOfBlocks)]
    observed_counts = [count_overlapping_template(block, template) for block in blocks]
    F = np.bincount(observed_counts, minlength=K + 1)
    expected_counts = [numberOfBlocks * p for p in pi]
    chiSquare = sum((F[i] - expected_counts[i]) ** 2 / expected_counts[i] for i in range(K + 1))
    pValue = sp.gammaincc(K / 2, chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #8 status : {testConclusion}, pValue: {round(pValue, 5)}"


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
        return L

    blocks = [bitString[i * M:(i + 1) * M] for i in range(N)]
    complexities = [berlekampMasseyAlgorithm([int(bit) for bit in block]) for block in blocks]
    T = [(complexity - expected_complexity) for complexity in complexities]
    chiSquare = sum([(t ** 2) / (M / 2) for t in T])
    pValue = sp.gammaincc(N / 2, chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #10 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #10 status : {testConclusion}, pValue: {round(pValue, 5)}"


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

    V_m = countPatterns(bitString, m)
    V_m_1 = countPatterns(bitString, m - 1)
    V_m_2 = countPatterns(bitString, m - 2)
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
    if not testConclusion:
        return f"Numbers are not  random. Test #11 status : {testConclusion}, pValue_1: {round(pValue_1, 5)}, pValue_2:{round(pValue_2, 5)}"


def approximateEntropyTest_12(bitString, bitStringLength):
    try:
        if bitStringLength < 100:
            raise ValueError("Длина битовой строки должна быть не менее 100 бит.")
    except ValueError as e:
        return f"Ошибка: {e}"
    if bitStringLength < 500:
        m = 2
    elif bitStringLength < 2000:
        m = 3
    else:
        m = 4

    def calculate_frequency(pattern_length):
        counts = {}
        for i in range(bitStringLength):
            pattern = bitString[i:i + pattern_length]
            if len(pattern) < pattern_length:
                pattern += bitString[:pattern_length - len(pattern)]
            counts[pattern] = counts.get(pattern, 0) + 1
        total_patterns = bitStringLength
        for key in counts:
            counts[key] /= total_patterns  # нормируем частоты
        return counts

    P_m = calculate_frequency(m)
    P_m1 = calculate_frequency(m + 1)

    def calculate_entropy(patternCounts):
        return sum([p * math.log(p, 2) for p in patternCounts.values() if p > 0])

    entropy_m = calculate_entropy(P_m)
    entropy_m1 = calculate_entropy(P_m1)
    approx_entropy = entropy_m - entropy_m1
    chiSquare = 2 * bitStringLength * (math.log(2) - approx_entropy)
    if chiSquare < 0:
        chiSquare = 0
    pValue = sp.gammaincc(2 ** (m - 1), chiSquare / 2)
    testConclusion = (pValue >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #12 status : {testConclusion}, pValue: {round(pValue, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #12 status : {testConclusion}, pValue: {round(pValue, 5)}"


def cumulativeSumsTest_13(bitString: str, bitStringLength: int):
    try:
        if len(bitString) < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #13 False"
    convertedBits = [1 if bit == '1' else -1 for bit in bitString]
    cumulativeSum = [0] * bitStringLength
    cumulativeSum[0] = convertedBits[0]
    for i in range(1, bitStringLength):
        cumulativeSum[i] = cumulativeSum[i - 1] + convertedBits[i]

    def calcPValue(cumulativeSums):
        maxDeviation = max([abs(x) for x in cumulativeSums])
        pValue = 1.0
        for k in np.arange((-bitStringLength // 2 - 3) / 4, (bitStringLength // 2 - 1) / 4, step=1):
            term1 = erfc((4 * k + 1) * maxDeviation / math.sqrt(2 * bitStringLength))
            term2 = erfc((4 * k - 1) * maxDeviation / math.sqrt(2 * bitStringLength))
            pValue -= (term1 - term2)
        return pValue

    pValueForward = calcPValue(cumulativeSum)
    pValueBackward = calcPValue(cumulativeSum[::-1])
    testConclusion = (pValueForward >= 0.01 and pValueBackward >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #13 status : {testConclusion}, pValue: {round(pValueForward, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #13 status : {testConclusion}, pValue: {round(pValueForward, 5)}"


def randomExcursionTest_14(bitString: str, bitStringLength: int):
    try:
        if bitStringLength < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #14 False"
    transformedBits = [1 if bit == '1' else -1 for bit in bitString]
    cumulativeSums = [0]
    for bit in transformedBits:
        cumulativeSums.append(cumulativeSums[-1] + bit)
    cycles = []
    currentCycle = []
    for i, value in enumerate(cumulativeSums):
        currentCycle.append(value)
        if value == 0 and len(currentCycle) > 1:
            cycles.append(currentCycle)
            currentCycle = []
    if len(cycles) == 0:
        raise ValueError("Последовательность не содержит циклов.")
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    stateVisits = {state: 0 for state in states}
    for cycle in cycles:
        for state in states:
            stateVisits[state] += cycle.count(state)
    pValues = {}
    chiSquare = {}
    total_cycles = len(cycles)
    for state in states:
        observedVisits = stateVisits[state]
        expectedVisits = total_cycles * (1.0 / (2 * abs(state))) if state != 0 else 0
        variance = total_cycles * (1 - 1.0 / (2 * abs(state))) if state != 0 else 0
        if variance > 0:
            chiSquareValue = (observedVisits - expectedVisits) ** 2 / variance
            pValue = round(sp.gammaincc(0.5, chiSquareValue / 2.0), 5)
            chiSquare[state] = chiSquareValue
        else:
            chiSquare[state] = 0
            pValue = 0.0
        pValues[state] = pValue
    testConclusion = all(p > 0.01 for p in pValues.values())
    if testConclusion:
        return f"Numbers are random. Test #14 status : {testConclusion}, pValues: {pValues}"
    if not testConclusion:
        return f"Numbers are not  random. Test #14 status : {testConclusion}, pValues: {pValues}"


def randomExcursionVariantTest_15(bitString: str, bitStringLength: int):
    try:
        if bitStringLength < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #15 False"
    transformedBits = [1 if bit == '1' else -1 for bit in bitString]
    cumulativeSums = [0]
    for bit in transformedBits:
        cumulativeSums.append(cumulativeSums[-1] + bit)
    cycles = []
    currentCycle = []
    for i, value in enumerate(cumulativeSums):
        currentCycle.append(value)
        if value == 0 and len(currentCycle) > 1:
            cycles.append(currentCycle)
            currentCycle = []
    if len(cycles) == 0:
        raise ValueError("Последовательность не содержит циклов.")
    states = list(range(-9, 10))
    states.remove(0)
    stateVisits = {state: 0 for state in states}
    for cycle in cycles:
        for state in states:
            stateVisits[state] += cycle.count(state)
    pValues = {}
    chiSquare = {}
    total_cycles = len(cycles)
    for state in states:
        observedVisits = stateVisits[state]
        expectedVisits = total_cycles * (1.0 / (2 * abs(state))) if state != 0 else 0
        variance = total_cycles * (1 - 1.0 / (2 * abs(state))) if state != 0 else 0
        if variance > 0:
            chiSquareValue = (observedVisits - expectedVisits) ** 2 / variance
            pValue = round(sp.erfc(math.sqrt(chiSquareValue / 2.0)), 5)
            chiSquare[state] = chiSquareValue
        else:
            chiSquare[state] = 0
            pValue = 0.0
        pValues[state] = pValue
    testConclusion = all(p > 0.01 for p in pValues.values())
    if testConclusion:
        return f"Numbers are random. Test #15 status : {testConclusion}, pValues: {pValues}"
    if not testConclusion:
        return f"Numbers are not  random. Test #15 status : {testConclusion}, pValues: {pValues}"


print("Введите номер источника случайных чисел. 1 - QRNG, 2 - PRNG 3 - Пользовательские данные")
source_choise = int(input())
if source_choise == 1:
    randomNumbersQRNG = int(input('Введите количество чисел: '))
    randomNumbers, bitString, bitStringLength = getRandomNumbers(randomNumbersQRNG)
elif source_choise == 2:
    randomNumbersLocal = int(input('Введите количество чисел: '))
    randomNumbers, bitString, bitStringLength = getRandomNumbersLocal(randomNumbersLocal)
elif source_choise == 3:
    dataNumbers = input('Введите последовательность 16-и битных чисел через запятую или битовую строку: ')
    randomNumbers, bitString, bitStringLength = getRandomNumbersUser(dataNumbers)
else:
    print("Некорректный номер источника случайных чисел")

print(f'Список полученных от сервера чисел {randomNumbers}')
print(f'Битовое представления последовательности чисел {bitString}')
print(f'Длина строки {bitStringLength}')
if bitString is not None and bitStringLength is not None:
    print(frequencyTest_1(bitString, bitStringLength))
else:
    print("Error: Could not perform frequency test due to invalid random number data")

print(frequencyWithinABlockTest_2(bitString, bitStringLength))

print(runTest_3(bitString, bitStringLength))

print(runWithinABlockTest_4(bitString, bitStringLength))

print(binaryMatrixRankTest_5(bitString, bitStringLength))

print(discreteFourierTransformTest_6(bitString, bitStringLength))

print(nonOverlappingTemplateMachineTest_7(bitString, bitStringLength))

print(overlappingTemplateMachineTest_8(bitString, bitStringLength))

print(universalStatisticalTest_9(bitString, bitStringLength))

print(linearComplexityTest_10(bitString, bitStringLength))

print(serialTest_11(bitString, bitStringLength))

print(approximateEntropyTest_12(bitString, bitStringLength))

print(cumulativeSumsTest_13(bitString, bitStringLength))

print(randomExcursionTest_14(bitString, bitStringLength))

print(randomExcursionVariantTest_15(bitString, bitStringLength))
