from __future__ import annotations


def generate_pycbc_waveform(
    mass1: float,
    mass2: float,
    approximant: str = "IMRPhenomD",
    delta_t: float = 1.0 / 4096.0,
    f_lower: float = 20.0,
):
    """Generate a time-domain compact-binary waveform with PyCBC."""
    try:
        from pycbc.waveform import get_td_waveform
    except ImportError as exc:
        raise ImportError(
            "pycbc is required for waveform generation. Install with `pip install -e .[gw]`."
        ) from exc

    plus, cross = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        delta_t=delta_t,
        f_lower=f_lower,
    )
    return plus, cross


def generate_pycbc_fd_waveform(
    mass1: float,
    mass2: float,
    approximant: str = "IMRPhenomD",
    delta_f: float = 1.0 / 16.0,
    f_lower: float = 20.0,
    f_final: float = 512.0,
):
    """Generate a frequency-domain compact-binary waveform with PyCBC."""
    try:
        from pycbc.waveform import get_fd_waveform
    except ImportError as exc:
        raise ImportError(
            "pycbc is required for waveform generation. Install with `pip install -e .[gw]`."
        ) from exc

    plus, _ = get_fd_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        delta_f=delta_f,
        f_lower=f_lower,
        f_final=f_final,
    )
    return plus
