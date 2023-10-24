import argparse
from utils import create_data_lists

def main(args):
    create_data_lists(train_folders=args.train_folders,
                      test_folders=args.test_folders,
                      min_size=args.min_size,
                      output_folder=args.output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data lists for training and testing.")
    parser.add_argument('--train_folders', nargs='+', help="List of training folders")
    parser.add_argument('--test_folders', nargs='+', help="List of testing folders")
    parser.add_argument('--min_size', type=int, default=100, help="Minimum width and height of images")
    parser.add_argument('--output_folder', default="./datasets" ,help="Output folder")
    
    args = parser.parse_args()
    main(args)
