from src.data.data_loader import DataLoader


def test_data_loader():
    """Tests the DataLoader."""
    data_loader = DataLoader(
        data_dir="data/shakesphere", sequence_length=256, batch_size=32
    )
    x, y = next(iter(data_loader))
    assert x.shape == (32, 256)
    assert y.shape == (32, 256)
