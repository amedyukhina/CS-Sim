import argparse
import json

from cs_sim.batch.batch_synth import batch_generate_img_with_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str, help='json file with parameters', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='Output directory', required=True)
    parser.add_argument('-n', '--n-img', type=int, default=10, help='Number of images to generate')
    parser.add_argument('-j', '--n-jobs', type=int, default=8)
    parser.add_argument('-b', '--fn-base', type=str, default='line_img')
    parser.add_argument('-e', '--ext', type=str, default='.tif')
    args = parser.parse_args()
    parameter_file = args.parameter_file

    with open(parameter_file) as f:
        params = json.load(f)

    print('\nThe following are the parameters that will be used:')
    print(params)
    print('\n')

    batch_generate_img_with_lines(n_img=args.n_img, dir_out=args.output_dir,
                                  fn_base=args.fn_base, fn_ext=args.ext, n_jobs=args.n_jobs,
                                  **params)
