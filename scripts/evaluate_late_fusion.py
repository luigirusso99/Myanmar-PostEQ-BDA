import argparse
from bdd.config import load_config
from bdd.evaluate import evaluate_cross_validation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_cross_validation(load_config(args.config))

if __name__ == "__main__":
    main()
