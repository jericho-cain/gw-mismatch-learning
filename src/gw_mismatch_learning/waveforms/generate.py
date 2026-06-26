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
