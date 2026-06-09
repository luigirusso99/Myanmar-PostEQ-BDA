import argparse
from bdd.config import load_config
from bdd.patches import create_dataset_patches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    create_dataset_patches(cfg)

if __name__ == "__main__":
    main()
