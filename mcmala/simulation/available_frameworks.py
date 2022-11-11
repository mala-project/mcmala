is_mala_available = True
try:
    import mala
except ModuleNotFoundError:
    is_mala_available = False
is_qepy_available = True
try:
    from qepy.calculator import QEpyCalculator
except ModuleNotFoundError:
    is_qepy_available = False
