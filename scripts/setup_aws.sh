# scripts/setup_aws.sh
#!/bin/bash

# Exit on error
set -e

# Default values
INSTANCE_TYPE="g4dn.xlarge"
REGION="us-west-2"
SPOT_INSTANCE="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --instance-type)
            INSTANCE_TYPE="$2"
            shift
            shift
            ;;
        --region)
            REGION="$2"
            shift
            shift
            ;;
        --no-spot)
            SPOT_INSTANCE="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Setting up AWS environment..."

# Check AWS CLI installation
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
        sudo installer -pkg AWSCLIV2.pkg -target /
        rm AWSCLIV2.pkg
    else
        # Linux installation
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
fi

# Configure AWS credentials if not already configured
if [ ! -f ~/.aws/credentials ]; then
    echo "Configuring AWS credentials..."
    aws configure
fi

# Create security group
SECURITY_GROUP_NAME="shallowflow-sg"
aws ec2 create-security-group \
    --group-name $SECURITY_GROUP_NAME \
    --description "Security group for ShallowFlow" \
    --region $REGION || true

# Add SSH access
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP_NAME \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION || true

# Launch instance
if [ "$SPOT_INSTANCE" = "true" ]; then
    echo "Launching spot instance..."
    aws ec2 request-spot-instances \
        --instance-count 1 \
        --type one-time \
        --launch-specification "{
            \"ImageId\": \"ami-0c55b159cbfafe1f0\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"SecurityGroups\": [\"$SECURITY_GROUP_NAME\"]
        }" \
        --region $REGION
else
    echo "Launching on-demand instance..."
    aws ec2 run-instances \
        --image-id ami-0c55b159cbfafe1f0 \
        --instance-type $INSTANCE_TYPE \
        --security-groups $SECURITY_GROUP_NAME \
        --region $REGION
fi

echo "AWS setup complete!"