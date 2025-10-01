from __future__ import annotations

import base64
from pathlib import Path

from aws_cdk import (
    CfnOutput,
    CfnParameter,
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct


class InfraStack(Stack):
    """Provision infrastructure to host the step1.py text-generation script."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        key_pair_param = CfnParameter(
            self,
            "KeyPairName",
            type="String",
            description="Name of an existing EC2 key pair for SSH access.",
        )

        vpc = ec2.Vpc(
            self,
            "AijourneyVpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    map_public_ip_on_launch=True,
                )
            ],
        )

        security_group = ec2.SecurityGroup(
            self,
            "AijourneyInstanceSg",
            vpc=vpc,
            description="Security group for the Mistral host.",
            allow_all_outbound=True,
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(22),
            "Allow SSH from anywhere (restrict this in production).",
        )

        instance_role = iam.Role(
            self,
            "AijourneyInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Allows the instance to connect to SSM for management.",
        )
        instance_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        repo_root = Path(__file__).resolve().parents[2]
        step1_path = repo_root / "step1.py"
        if not step1_path.exists():
            raise FileNotFoundError(f"Unable to locate {step1_path} for user data upload.")

        script_b64 = base64.b64encode(step1_path.read_text().encode("utf-8")).decode(
            "utf-8"
        )

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "dnf update -y",
            "dnf install -y git python3.11 python3.11-venv",
            "runuser -l ec2-user -c 'python3.11 -m venv ~/aijourney-venv'",
            "runuser -l ec2-user -c 'source ~/aijourney-venv/bin/activate && pip install --upgrade pip'",
            "runuser -l ec2-user -c 'source ~/aijourney-venv/bin/activate && pip install torch transformers accelerate optimum safetensors'",
            "install -d -o ec2-user -g ec2-user /home/ec2-user/aijourney",
            f"printf '%s' '{script_b64}' | base64 -d > /home/ec2-user/aijourney/step1.py",
            "chown ec2-user:ec2-user /home/ec2-user/aijourney/step1.py",
            "runuser -l ec2-user -c 'echo "source ~/aijourney-venv/bin/activate" >> ~/.bashrc'",
        )

        instance = ec2.Instance(
            self,
            "MistralHost",
            instance_type=ec2.InstanceType("m7i-flex.large"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=security_group,
            role=instance_role,
            key_name=key_pair_param.value_as_string,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=200,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        encrypted=True,
                    ),
                )
            ],
        )

        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "PublicDns", value=instance.instance_public_dns_name)
        CfnOutput(self, "VpcId", value=vpc.vpc_id)
