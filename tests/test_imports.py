def test_import_package() -> None:
    import gw_mismatch_learning

    assert gw_mismatch_learning.__version__


def test_lazy_gw_imports() -> None:
    from gw_mismatch_learning.waveforms.banks import sample_binary_mass_bank

    bank = sample_binary_mass_bank(4, (20.0, 40.0), (10.0, 30.0), seed=1)
    assert bank.mass_1.shape == (4,)
