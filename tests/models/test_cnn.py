import torch

from fabtrace.models.cnn import WaferCNN, count_parameters


def test_input_output_shape():
    model = WaferCNN().eval()
    with torch.no_grad():
        out = model(torch.zeros(4, 3, 64, 64))
    assert out.shape == (4, 9)


def test_model_is_small():
    assert count_parameters(WaferCNN()) < 200_000