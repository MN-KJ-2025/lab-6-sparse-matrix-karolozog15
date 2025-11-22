# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray or sp.sparse.csc_array) -> bool or None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    
    if isinstance(A, sp.sparse._csc.csc_array):
        A = A.toarray()

    if len(A.shape)!=2 or np.size(A,0)!=np.size(A,1):
        return None
    
    diag = np.abs(np.diag(A))
    row_sums = np.sum(np.abs(A), axis=1) - diag
    return np.all(diag > row_sums)

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float or None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """


    if not isinstance(A,np.ndarray) or not isinstance(x,np.ndarray) or not isinstance(b,np.ndarray):
        return None
    
    if A.ndim !=2 or b.ndim !=1 or x.ndim !=1:
        return None
    if np.size(A,1) != np.size(x,0) or np.size(b,0) != np.size(A,0):
        return None
    r=b-A.dot(x)
    norm=np.linalg.norm(r)
    return norm
