from cs_sim.synth.filaments import generate_img_with_filaments


def test_filaments(curve_params):
    size = curve_params['size']
    img = generate_img_with_filaments(imgshape=[size] * 3,
                                      **{key: curve_params[key] for key in curve_params.keys()
                                         if key != 'size'})
    assert img.max() == curve_params['maxval']
    assert img.min() == 0
    assert img.shape == (size, size, size)
