# Aijourney CDK Stack

This CDK app provisions an `m7i-flex.large` Amazon Linux 2023 EC2 instance, installs the Python dependencies required by `step1.py`, and copies the script to `/home/ec2-user/aijourney/step1.py` so you can log in and run it immediately.

## Prerequisites
- AWS CDK v2 CLI installed locally (`cdk --version`)
- AWS account credentials configured for the target account/region
- An existing EC2 key pair in that region (needed for SSH)
- AWS account bootstrapped for CDK v2 (`cdk bootstrap`) if not done already

## Setup
```bash
cd infra
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Deploy
Replace `YOUR_KEYPAIR_NAME` with the name of your existing EC2 key pair and optionally export the desired region/account through environment variables (for example `AWS_PROFILE`, `AWS_REGION`).

```bash
cdk deploy \
  --parameters KeyPairName=YOUR_KEYPAIR_NAME
```

The stack outputs the instance ID and public DNS name when deployment completes. SSH is enabled from any IPv4 address by default; tighten the security group if needed before running `cdk deploy`.

## Connect and run
```bash
ssh -i /path/to/YOUR_KEYPAIR.pem ec2-user@<public-dns-from-output>
source ~/aijourney-venv/bin/activate
cd ~/aijourney
python step1.py
```

The virtual environment is pre-populated with `torch`, `transformers`, `accelerate`, `optimum`, and `safetensors`.

## Cleanup
```bash
cdk destroy
```

## Notes
- The included `step1.py` loads a GPTQ-quantized model, which still expects an NVIDIA GPU. The provided `m7i-flex.large` instance is CPU-only; adjust the model or select a GPU instance type before running if you need GPU acceleration.
- Session Manager access is enabled through the attached IAM role (`AmazonSSMManagedInstanceCore`), so you can alternatively connect via `aws ssm start-session --target <instance-id>`.
- The root EBS volume is sized to 200 GiB to leave headroom for model downloads. Adjust `volume_size` in `infra/infra/infra_stack.py` if you prefer a different size.
