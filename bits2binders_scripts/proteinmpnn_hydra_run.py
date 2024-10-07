from dataclasses import dataclass
import os
import subprocess
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

env_python_path = "/home/nelly/miniforge3/envs/mlfold/bin/python"

@dataclass
class Paths:
    folder_with_pdbs: str = "./input_pdbs"
    output_dir: str = "./output_pdbs"
    path_to_model_weights: str = ""
    chain_list: str = "A"


@dataclass
class ModelParams:
    model_name: str = "v_48_020"
    use_soluble_model: bool = False
    ca_only: bool = False


@dataclass
class RunParams:
    sampling_temps: List[float] = (0.1, 0.2)
    backbone_noises: List[float] = (0.01, 0.02)
    seeds: List[int] = (37, 42)
    num_seq_per_target: int = 2
    batch_size: int = 1
    suppress_print: bool = False
    save_score: bool = False
    max_length: int = 200000


@dataclass
class ProteinMPNNConfig:
    paths: Paths = Paths()
    model: ModelParams = ModelParams()
    run: RunParams = RunParams()


cs = ConfigStore.instance()
cs.store(name="config", node=ProteinMPNNConfig)


def setup_directories(paths: Paths) -> tuple[str, str]:
    """Create output directory and return paths for parsed and assigned chains."""
    os.makedirs(paths.output_dir, exist_ok=True)
    
    path_for_parsed_chains = os.path.join(paths.output_dir, "parsed_pdbs.jsonl")
    path_for_assigned_chains = os.path.join(paths.output_dir, "assigned_pdbs.jsonl")
    
    return path_for_parsed_chains, path_for_assigned_chains


def run_preprocessing(
    folder_with_pdbs: str,
    path_for_parsed_chains: str,
    path_for_assigned_chains: str,
    chain_list: str
) -> None:
    """Run the prerequisite parsing scripts."""
    # Parse multiple chains
    subprocess.run([
        env_python_path, "../helper_scripts/parse_multiple_chains.py",
        "--input_path", folder_with_pdbs,
        "--output_path", path_for_parsed_chains
    ], check=True)

    # Assign fixed chains
    subprocess.run([
        env_python_path, "../helper_scripts/assign_fixed_chains.py",
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_assigned_chains,
        "--chain_list", chain_list
    ], check=True)


def build_mpnn_command(
    parsed_chains_path: str,
    assigned_chains_path: str,
    cfg: ProteinMPNNConfig,
    sampling_temp: float,
    backbone_noise: float,
    seed: int
) -> List[str]:
    """Construct the ProteinMPNN command with given parameters."""
    cmd = [
        env_python_path, "../protein_mpnn_run.py",
        "--jsonl_path", parsed_chains_path,
        "--chain_id_jsonl", assigned_chains_path,
        "--out_folder", cfg.paths.output_dir,
        "--num_seq_per_target", str(cfg.run.num_seq_per_target),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size", str(cfg.run.batch_size),
        "--backbone_noise", str(backbone_noise),
        "--model_name", cfg.model.model_name,
        "--max_length", str(cfg.run.max_length)
    ]
    
    if cfg.paths.path_to_model_weights:
        cmd.extend(["--path_to_model_weights", cfg.paths.path_to_model_weights])
    
    if cfg.model.use_soluble_model:
        cmd.append("--use_soluble_model")
    
    if cfg.model.ca_only:
        cmd.append("--ca_only")
        
    if cfg.run.suppress_print:
        cmd.extend(["--suppress_print", "1"])
        
    if cfg.run.save_score:
        cmd.extend(["--save_score", "1"])
    
    return cmd


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ProteinMPNNConfig) -> None:
    """Main function to run ProteinMPNN with parameter sweeping."""
    print("Running with config:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup directories and get paths
    parsed_chains_path, assigned_chains_path = setup_directories(cfg.paths)
    
    # Run preprocessing steps
    run_preprocessing(
        cfg.paths.folder_with_pdbs,
        parsed_chains_path,
        assigned_chains_path,
        cfg.paths.chain_list
    )
    
    # Run ProteinMPNN for each parameter combination
    for sampling_temp in cfg.run.sampling_temps:
        for backbone_noise in cfg.run.backbone_noises:
            for seed in cfg.run.seeds:
                print(f"\nRunning with:")
                print(f"  sampling_temp={sampling_temp}")
                print(f"  backbone_noise={backbone_noise}")
                print(f"  seed={seed}")
                
                cmd = build_mpnn_command(
                    parsed_chains_path,
                    assigned_chains_path,
                    cfg,
                    sampling_temp,
                    backbone_noise,
                    seed
                )
                
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()