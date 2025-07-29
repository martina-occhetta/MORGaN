import yaml
import os
import argparse
from pathlib import Path
import logging
from typing import Dict, Any


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("experiments.log")],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise


def run_experiment(config: Dict[str, Any], dataset_name: str):
    """Run a single experiment with the given configuration and dataset."""
    try:
        # Base command with experiment type and dataset
        base_cmd = f"python main_transductive.py --experiment_type {config['experiment_type']} --dataset {dataset_name}"

        # Add all parameters from config (excluding experiment_type and datasets)
        for param, value in config.items():
            if param not in ["experiment_type", "datasets", "base_ppi"]:
                if isinstance(value, bool):
                    # Handle boolean values
                    if value:
                        base_cmd += f" --{param}"
                else:
                    base_cmd += f" --{param} {value}"

        logging.info(f"\nRunning experiment for dataset: {dataset_name}")
        logging.info(f"Command: {base_cmd}")

        # Execute the command
        exit_code = os.system(base_cmd)
        if exit_code != 0:
            logging.error(
                f"Experiment failed for dataset {dataset_name} with exit code {exit_code}"
            )
        else:
            logging.info(
                f"Successfully completed experiment for dataset {dataset_name}"
            )

    except Exception as e:
        logging.error(f"Error running experiment for dataset {dataset_name}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments from YAML configuration"
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    try:
        # Load configuration
        config = load_config(args.config_path)

        # Create experiment directory if it doesn't exist
        experiment_dir = os.path.join(
            "data", "paper", "graphs", config["experiment_type"]
        )
        os.makedirs(experiment_dir, exist_ok=True)

        # Run experiments for each dataset
        if "datasets" in config:
            for dataset in config["datasets"]:
                run_experiment(config, dataset["name"])
        else:
            # For configurations without explicit datasets list
            # Use the experiment_type as the dataset name
            run_experiment(config, config["experiment_type"])

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
