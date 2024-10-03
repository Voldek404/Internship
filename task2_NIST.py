import requests
import scipy.special as sp
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import chi2
from scipy.fftpack import fft
from scipy.special import erfc
from scipy.special import gammaincc
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
        if len(bitString) < 1000:
            raise ValueError("Длина битовой строки должна быть не менее 1000 бит.")
    except ValueError as e:
        return f"Ошибка: {e}, Test #13 False"
    bitString = np.array([int(bit) for bit in bitString], dtype=int)
    bitString_copy = bitString.copy()
    bitString[bitString == 0] = -1
    forward_sum, backward_sum = 0, 0
    forward_max, backward_max = 0, 0
    for i in range(bitString_copy.size):
        forward_sum += bitString_copy[i]
        backward_sum += bitString_copy[bitString_copy.size - 1 - i]
        forward_max = max(abs(forward_sum), forward_max)
        backward_max = max(abs(backward_sum), backward_max)

    def computePValue(bitStringLength, max_excursion):
        sum_a = 0.0
        start_k = int(math.floor((((-bitStringLength / max_excursion) + 1.0) / 4.0)))
        end_k = int(math.floor((((bitStringLength / max_excursion) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c = 0.5 * erfc(-(((4.0 * k) + 1.0) * max_excursion) / math.sqrt(bitStringLength) * math.sqrt(0.5))
        d = 0.5 * erfc(-(((4.0 * k) - 1.0) * max_excursion) / math.sqrt(bitStringLength) * math.sqrt(0.5))
        sum_a += c - d

        sum_b = 0.0
        start_k = int(math.floor((((-bitStringLength / max_excursion) - 3.0) / 4.0)))
        end_k = int(math.floor((((bitStringLength / max_excursion) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c = 0.5 * erfc(-(((4.0 * k) + 3.0) * max_excursion) / math.sqrt(bitStringLength) * math.sqrt(0.5))
        d = 0.5 * erfc(-(((4.0 * k) + 1.0) * max_excursion) / math.sqrt(bitStringLength) * math.sqrt(0.5))
        sum_b += c - d

        return 1.0 - (sum_a + sum_b)

    pValueForward = computePValue(bitString_copy.size, forward_max)
    pValueBackWard = computePValue(bitString.size, backward_max)
    testConclusion = (pValueForward >= 0.01 and pValueBackward >= 0.01)
    if testConclusion:
        return f"Numbers are random. Test #13 status : {testConclusion}, pValue: {round(pValueForward, 5)}"
    if not testConclusion:
        return f"Numbers are not  random. Test #13 status : {testConclusion}, pValue: {round(pValueBackWard, 5)}"


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
print(frequencyTest_1(bitString, bitStringLength))

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
