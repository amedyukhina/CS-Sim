from cs_sim.synth.lines import generate_img_with_lines


def test_lines(line_params):
    size = line_params.pop('size')
    img = generate_img_with_lines(imgshape=[size] * 3, **line_params)
    assert img.max() == line_params['maxval']
    assert img.min() == 0
    assert img.shape == (size, size, size)
