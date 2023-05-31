# coding=utf-8
import numpy as np


def PERIOD(T, F0, COEFF, TMID):
    DT = (T - TMID) * 1440.

    FREQ = (11 * (DT ** 10) * COEFF[11] + 10 * (DT ** 9) * COEFF[10] + 9 * (DT ** 8) * COEFF[9] + 8 * (DT ** 7) * COEFF[
        8] + 7 * (DT ** 6) * COEFF[7] + 6 * (DT ** 5) * COEFF[6] + 5 * (DT ** 4) * COEFF[5] + 4 * (DT ** 3) * COEFF[
                4] + 3 * (DT ** 2) * COEFF[3] + 2 * DT * COEFF[2] + COEFF[1]) * (1 / 60.) + F0
    return 1 / FREQ


def findP(file, MJD0, deltaT_seg):
    """
    This is a function that given the T (in seconds) since the start of the observation (MJD0)
    finds the instantenous period P (in seconds) for that instant using the polynomial coefficient
    in the .polycos file (file).

    Args:
        file: (string) name of the .polycos file
        MJD0: (float) MJD at the start of the observation
        T_seg: (float) time in seconds since the start of the observation

    Returns:
        P: (float) instantaneous period in seconds at T
    """
    T_MJD = MJD0 + deltaT_seg * 1. / (24. * 60. * 60.)  # Hallamos el MJD correspondiente al T dado
    lines = sum(1 for line in open(file))  # número de líneas

    n = 0  # número de líneas leídas del archivo .polycos. Comenzamos por el primer bloque.
    not_found = True  # variable booleana que nos dice si ya se encontró el bloque del .polycos correspondiente al T
    while not_found:
        TMID = np.genfromtxt(file, comments="none", dtype=float, skip_header=n, max_rows=1, usecols=(3))
        span_min = np.genfromtxt(file, comments="none", dtype=float, skip_header=n + 1, max_rows=1, usecols=(3))
        span_mjd = span_min * (1. / (60. * 24.))

        T_inicial = TMID - span_mjd / 2.
        T_final = TMID + span_mjd / 2.

        if T_MJD >= T_inicial and T_MJD < T_final:  # nos fijamos si el MJD está dentro de este bloque.

            F0 = np.genfromtxt(file, comments="none", dtype=float, skip_header=n + 1, max_rows=1, usecols=(1))
            matrix = np.genfromtxt(file, usecols=range(3), dtype=str, skip_header=n + 2, max_rows=4)
            # print(TMID, span_min, span_mjd, F0, matrix)

            matrix_sub = []
            for m in matrix:
                # print(m)
                m = str(m).replace('\'', '').replace(']', '').replace('[', '').replace('\n', '').replace('D',
                                                                                                         'E').split(' ')
                for m_sub in m:
                    matrix_sub.append(float(m_sub))
            matrix_sub = np.array(matrix_sub)
            COEFF = matrix_sub
            P_seg = PERIOD(T_MJD, F0, COEFF, TMID)  # período en segundos
            not_found = False
        else:
            n += 6  # si no se encontró el MJD en el bloque del .polycos en cuestión, se avanza al siguiente bloque
            if n >= lines:
                print("ERROR: el T indicado no se encuentra dentro del .polycos", T_MJD)
                return

    return P_seg
