import argparse
import json

from cs_sim.batch.batch_corrupt import batch_corrupt_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str,
                        help='json file with corruption steps and parameters', required=True)
    parser.add_argument('-i', '--input-dir', type=str, help='Input directory', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='Output directory', required=True)
    parser.add_argument('-j', '--n-jobs', type=int, default=8)
    args = parser.parse_args()
    parameter_file = args.parameter_file

    with open(parameter_file) as f:
        corr_steps = json.load(f)

    print('\nThe following are the corruptions steps that will be applied:')
    print(corr_steps)
    print('\n')

    batch_corrupt_image(dir_in=args.input_dir, dir_out=args.output_dir,
                        n_jobs=args.n_jobs, corruption_steps=corr_steps)
