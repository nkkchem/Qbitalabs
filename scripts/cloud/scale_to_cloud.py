#!/usr/bin/env python3
"""
QBitaLabs Cloud Scaling Script
Automatically scales training to cloud when local resources are insufficient

Trigger Conditions:
1. Dataset > 10GB
2. Model parameters > 1B
3. Training time estimate > 24 hours
4. Memory requirement > 100GB
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Resource requirements for a training job."""
    dataset_size_gb: float
    model_parameters: int
    estimated_training_hours: float
    memory_required_gb: float
    gpu_required: bool
    gpu_type: Optional[str] = None
    num_gpus: int = 1


@dataclass
class CloudConfig:
    """Cloud provider configuration."""
    provider: str  # aws, gcp, azure
    region: str
    instance_type: str
    spot_instance: bool
    max_price_per_hour: float
    storage_type: str
    storage_size_gb: int


class ResourceAnalyzer:
    """Analyzes resource requirements for training jobs."""

    # Scaling thresholds
    DATASET_THRESHOLD_GB = 10
    PARAMETER_THRESHOLD = 1_000_000_000  # 1B
    TIME_THRESHOLD_HOURS = 24
    MEMORY_THRESHOLD_GB = 100

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        self.data_dir = data_dir
        self.config = config

    def analyze_dataset(self) -> float:
        """Calculate total dataset size in GB."""
        total_bytes = 0

        if self.data_dir.exists():
            for file in self.data_dir.rglob("*"):
                if file.is_file():
                    total_bytes += file.stat().st_size

        return total_bytes / (1024 ** 3)

    def estimate_parameters(self) -> int:
        """Estimate model parameters from config."""
        models = self.config.get("models", {})
        total_params = 0

        for model_name, model_config in models.items():
            hidden_dim = model_config.get("hidden_dim", 256)
            num_layers = model_config.get("num_layers", 4)

            # Rough estimation
            params_per_layer = hidden_dim * hidden_dim * 4  # typical transformer layer
            total_params += params_per_layer * num_layers

        return total_params

    def estimate_training_time(self, dataset_size_gb: float, num_params: int) -> float:
        """Estimate training time in hours."""
        # Base time per epoch (hours)
        base_time = 0.1

        # Scale by dataset
        dataset_factor = max(1, dataset_size_gb / 1)  # 1GB baseline

        # Scale by parameters
        param_factor = max(1, num_params / 10_000_000)  # 10M baseline

        # Epochs
        epochs = self.config.get("training", {}).get("max_epochs", 100)

        return base_time * dataset_factor * param_factor * epochs

    def estimate_memory(self, num_params: int, batch_size: int) -> float:
        """Estimate memory requirement in GB."""
        # Rough estimation: 4 bytes per param (float32)
        # Plus gradients, optimizer states, activations
        param_memory = num_params * 4 / (1024 ** 3)

        # Multiply for gradients + optimizer states (Adam needs 3x)
        optimizer_memory = param_memory * 3

        # Activation memory depends on batch size
        activation_memory = batch_size * param_memory * 0.5

        return param_memory + optimizer_memory + activation_memory

    def get_requirements(self) -> ResourceRequirements:
        """Get full resource requirements."""
        dataset_size = self.analyze_dataset()
        num_params = self.estimate_parameters()
        batch_size = self.config.get("training", {}).get("batch_size", 64)

        return ResourceRequirements(
            dataset_size_gb=dataset_size,
            model_parameters=num_params,
            estimated_training_hours=self.estimate_training_time(dataset_size, num_params),
            memory_required_gb=self.estimate_memory(num_params, batch_size),
            gpu_required=num_params > 10_000_000,
            gpu_type="A100" if num_params > 100_000_000 else "T4",
            num_gpus=1 if num_params < 500_000_000 else 4
        )

    def should_scale_to_cloud(self, requirements: ResourceRequirements) -> bool:
        """Determine if cloud scaling is needed."""
        reasons = []

        if requirements.dataset_size_gb > self.DATASET_THRESHOLD_GB:
            reasons.append(f"Dataset size {requirements.dataset_size_gb:.1f}GB > {self.DATASET_THRESHOLD_GB}GB")

        if requirements.model_parameters > self.PARAMETER_THRESHOLD:
            reasons.append(f"Parameters {requirements.model_parameters:,} > {self.PARAMETER_THRESHOLD:,}")

        if requirements.estimated_training_hours > self.TIME_THRESHOLD_HOURS:
            reasons.append(f"Training time {requirements.estimated_training_hours:.1f}h > {self.TIME_THRESHOLD_HOURS}h")

        if requirements.memory_required_gb > self.MEMORY_THRESHOLD_GB:
            reasons.append(f"Memory {requirements.memory_required_gb:.1f}GB > {self.MEMORY_THRESHOLD_GB}GB")

        if reasons:
            logger.info("Cloud scaling triggered:")
            for reason in reasons:
                logger.info(f"  - {reason}")
            return True

        return False


class CloudProvisioner:
    """Provisions cloud resources for training."""

    # Instance recommendations by workload
    INSTANCE_RECOMMENDATIONS = {
        "aws": {
            "small": {"type": "g4dn.xlarge", "gpus": 1, "price": 0.526},
            "medium": {"type": "g4dn.12xlarge", "gpus": 4, "price": 3.912},
            "large": {"type": "p4d.24xlarge", "gpus": 8, "price": 32.77},
        },
        "gcp": {
            "small": {"type": "n1-standard-8-t4", "gpus": 1, "price": 0.50},
            "medium": {"type": "n1-standard-32-v100x4", "gpus": 4, "price": 4.00},
            "large": {"type": "a2-highgpu-8g", "gpus": 8, "price": 28.00},
        },
        "azure": {
            "small": {"type": "Standard_NC6", "gpus": 1, "price": 0.90},
            "medium": {"type": "Standard_NC24", "gpus": 4, "price": 3.60},
            "large": {"type": "Standard_ND96asr_v4", "gpus": 8, "price": 27.20},
        }
    }

    def __init__(self, provider: str = "aws"):
        self.provider = provider
        self.credentials_valid = False

    def select_instance(self, requirements: ResourceRequirements) -> CloudConfig:
        """Select appropriate instance type based on requirements."""
        # Determine workload size
        if requirements.model_parameters > 500_000_000:
            size = "large"
        elif requirements.model_parameters > 50_000_000:
            size = "medium"
        else:
            size = "small"

        instance = self.INSTANCE_RECOMMENDATIONS[self.provider][size]

        return CloudConfig(
            provider=self.provider,
            region="us-west-2" if self.provider == "aws" else "us-west1",
            instance_type=instance["type"],
            spot_instance=True,  # Use spot for cost savings
            max_price_per_hour=instance["price"] * 0.7,  # 70% of on-demand
            storage_type="ssd",
            storage_size_gb=int(requirements.dataset_size_gb * 2) + 100  # 2x data + 100GB
        )

    def estimate_cost(self, config: CloudConfig, hours: float) -> float:
        """Estimate total cost for training."""
        hourly_rate = config.max_price_per_hour if config.spot_instance else \
            self.INSTANCE_RECOMMENDATIONS[config.provider]["small"]["price"]

        compute_cost = hourly_rate * hours
        storage_cost = config.storage_size_gb * 0.10 / 30 * (hours / 24)  # $0.10/GB/month

        return compute_cost + storage_cost

    def generate_terraform(self, config: CloudConfig) -> str:
        """Generate Terraform configuration."""
        if config.provider == "aws":
            return self._generate_aws_terraform(config)
        elif config.provider == "gcp":
            return self._generate_gcp_terraform(config)
        else:
            return self._generate_azure_terraform(config)

    def _generate_aws_terraform(self, config: CloudConfig) -> str:
        """Generate AWS Terraform config."""
        return f'''# QBitaLabs Training Infrastructure - AWS
# Generated: {datetime.now().isoformat()}

provider "aws" {{
  region = "{config.region}"
}}

# Spot Instance Request
resource "aws_spot_instance_request" "training" {{
  ami                    = "ami-0123456789"  # Deep Learning AMI
  instance_type          = "{config.instance_type}"
  spot_price             = "{config.max_price_per_hour}"
  wait_for_fulfillment   = true

  root_block_device {{
    volume_size = {config.storage_size_gb}
    volume_type = "gp3"
  }}

  tags = {{
    Name        = "qbitalabs-training"
    Environment = "development"
    Project     = "mvp"
  }}
}}

# S3 Bucket for Data and Models
resource "aws_s3_bucket" "training_data" {{
  bucket = "qbitalabs-training-data"
}}

# Output
output "instance_ip" {{
  value = aws_spot_instance_request.training.public_ip
}}
'''

    def _generate_gcp_terraform(self, config: CloudConfig) -> str:
        """Generate GCP Terraform config."""
        return f'''# QBitaLabs Training Infrastructure - GCP
# Generated: {datetime.now().isoformat()}

provider "google" {{
  project = "qbitalabs-project"
  region  = "{config.region}"
}}

# GPU Instance
resource "google_compute_instance" "training" {{
  name         = "qbitalabs-training"
  machine_type = "{config.instance_type}"
  zone         = "{config.region}-a"

  boot_disk {{
    initialize_params {{
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = {config.storage_size_gb}
    }}
  }}

  guest_accelerator {{
    type  = "nvidia-tesla-t4"
    count = 1
  }}

  scheduling {{
    preemptible       = {str(config.spot_instance).lower()}
    automatic_restart = false
  }}

  network_interface {{
    network = "default"
    access_config {{}}
  }}
}}
'''

    def _generate_azure_terraform(self, config: CloudConfig) -> str:
        """Generate Azure Terraform config."""
        return f'''# QBitaLabs Training Infrastructure - Azure
# Generated: {datetime.now().isoformat()}

provider "azurerm" {{
  features {{}}
}}

resource "azurerm_resource_group" "training" {{
  name     = "qbitalabs-training"
  location = "{config.region}"
}}

resource "azurerm_linux_virtual_machine" "training" {{
  name                = "qbitalabs-training-vm"
  resource_group_name = azurerm_resource_group.training.name
  location            = azurerm_resource_group.training.location
  size                = "{config.instance_type}"

  priority = {'"Spot"' if config.spot_instance else '"Regular"'}
  eviction_policy = "Deallocate"
  max_bid_price = {config.max_price_per_hour}

  os_disk {{
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = {config.storage_size_gb}
  }}

  source_image_reference {{
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-hpc"
    sku       = "2004"
    version   = "latest"
  }}
}}
'''

    def generate_docker_compose(self, requirements: ResourceRequirements) -> str:
        """Generate Docker Compose for local/cloud deployment."""
        return f'''# QBitaLabs Training Docker Compose
# Generated: {datetime.now().isoformat()}

version: '3.8'

services:
  training:
    image: qbitalabs/training:latest
    build:
      context: .
      dockerfile: Dockerfile.training
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    volumes:
      - ./mvp/data:/app/data
      - ./mvp/models:/app/models
      - ./mvp/logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {requirements.num_gpus}
              capabilities: [gpu]
        limits:
          memory: {int(requirements.memory_required_gb)}G
    command: python scripts/train/train_mvp.py --config configs/training/cloud.yaml

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mvp/mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./mvp/logs:/logs
    command: tensorboard --logdir=/logs --host=0.0.0.0
'''


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions."""

    def __init__(self, analyzer: ResourceAnalyzer, provisioner: CloudProvisioner):
        self.analyzer = analyzer
        self.provisioner = provisioner

    def decide(self) -> Dict[str, Any]:
        """Make scaling decision."""
        requirements = self.analyzer.get_requirements()
        should_scale = self.analyzer.should_scale_to_cloud(requirements)

        decision = {
            "timestamp": datetime.now().isoformat(),
            "requirements": {
                "dataset_size_gb": requirements.dataset_size_gb,
                "model_parameters": requirements.model_parameters,
                "estimated_training_hours": requirements.estimated_training_hours,
                "memory_required_gb": requirements.memory_required_gb,
            },
            "scale_to_cloud": should_scale,
            "recommendation": "local" if not should_scale else "cloud"
        }

        if should_scale:
            cloud_config = self.provisioner.select_instance(requirements)
            estimated_cost = self.provisioner.estimate_cost(
                cloud_config,
                requirements.estimated_training_hours
            )

            decision["cloud_config"] = {
                "provider": cloud_config.provider,
                "instance_type": cloud_config.instance_type,
                "region": cloud_config.region,
                "spot_instance": cloud_config.spot_instance,
                "storage_size_gb": cloud_config.storage_size_gb
            }
            decision["estimated_cost_usd"] = estimated_cost

        return decision


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QBitaLabs Cloud Scaling")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/m4_mac.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./mvp/data",
        help="Data directory"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="aws",
        choices=["aws", "gcp", "azure"],
        help="Cloud provider"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deployment/cloud",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if scaling is needed, don't generate files"
    )

    args = parser.parse_args()

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Initialize components
    data_dir = Path(args.data_dir)
    analyzer = ResourceAnalyzer(data_dir, config)
    provisioner = CloudProvisioner(args.provider)
    engine = ScalingDecisionEngine(analyzer, provisioner)

    # Make decision
    decision = engine.decide()

    # Print decision
    print("\n" + "="*60)
    print("SCALING DECISION")
    print("="*60)
    print(f"Dataset Size: {decision['requirements']['dataset_size_gb']:.2f} GB")
    print(f"Model Parameters: {decision['requirements']['model_parameters']:,}")
    print(f"Estimated Time: {decision['requirements']['estimated_training_hours']:.1f} hours")
    print(f"Memory Required: {decision['requirements']['memory_required_gb']:.1f} GB")
    print()
    print(f"Recommendation: {decision['recommendation'].upper()}")

    if decision["scale_to_cloud"]:
        print()
        print("Cloud Configuration:")
        print(f"  Provider: {decision['cloud_config']['provider']}")
        print(f"  Instance: {decision['cloud_config']['instance_type']}")
        print(f"  Region: {decision['cloud_config']['region']}")
        print(f"  Spot Instance: {decision['cloud_config']['spot_instance']}")
        print(f"  Estimated Cost: ${decision['estimated_cost_usd']:.2f}")

    if not args.check_only and decision["scale_to_cloud"]:
        # Generate infrastructure files
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        requirements = analyzer.get_requirements()
        cloud_config = provisioner.select_instance(requirements)

        # Generate Terraform
        terraform = provisioner.generate_terraform(cloud_config)
        terraform_path = output_dir / f"main.tf"
        with open(terraform_path, "w") as f:
            f.write(terraform)
        print(f"\nGenerated: {terraform_path}")

        # Generate Docker Compose
        compose = provisioner.generate_docker_compose(requirements)
        compose_path = output_dir / "docker-compose.yml"
        with open(compose_path, "w") as f:
            f.write(compose)
        print(f"Generated: {compose_path}")

        # Save decision
        decision_path = output_dir / "scaling_decision.json"
        with open(decision_path, "w") as f:
            json.dump(decision, f, indent=2)
        print(f"Generated: {decision_path}")

    print("="*60)

    return 0 if not decision["scale_to_cloud"] else 1


if __name__ == "__main__":
    sys.exit(main())
