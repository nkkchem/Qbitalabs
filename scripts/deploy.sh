#!/bin/bash
# QBitaLabs Deployment Script
# Usage: ./scripts/deploy.sh [staging|production] [--build] [--push]

set -e

echo "========================================"
echo "  QBitaLabs Deployment"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
ENVIRONMENT="staging"
BUILD=false
PUSH=false
VERSION=$(cat pyproject.toml | grep 'version = ' | head -1 | cut -d'"' -f2)
IMAGE_NAME="qbitalabs/qbita-fabric"

# Parse arguments
for arg in "$@"; do
    case $arg in
        staging|production)
            ENVIRONMENT=$arg
            ;;
        --build|-b)
            BUILD=true
            ;;
        --push|-p)
            PUSH=true
            ;;
        --help|-h)
            echo "Usage: ./deploy.sh [ENVIRONMENT] [OPTIONS]"
            echo ""
            echo "Environments:"
            echo "  staging      Deploy to staging (default)"
            echo "  production   Deploy to production"
            echo ""
            echo "Options:"
            echo "  --build, -b  Build Docker image"
            echo "  --push, -p   Push image to registry"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
    esac
done

echo -e "\n${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}Version: ${VERSION}${NC}"

# Pre-deployment checks
echo -e "\n${YELLOW}Running pre-deployment checks...${NC}"

# Check if tests pass
echo -e "  Running tests..."
pytest tests/unit/ -q --tb=no
if [ $? -ne 0 ]; then
    echo -e "${RED}  Tests failed! Aborting deployment.${NC}"
    exit 1
fi
echo -e "${GREEN}  Tests passed ✓${NC}"

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}  Warning: Uncommitted changes detected${NC}"
    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "${RED}  Cannot deploy to production with uncommitted changes!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}  Git status clean ✓${NC}"

# Build Docker image
if [ "$BUILD" = true ]; then
    echo -e "\n${YELLOW}Building Docker image...${NC}"

    # Set tag based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        TAG="${IMAGE_NAME}:${VERSION}"
        LATEST_TAG="${IMAGE_NAME}:latest"
    else
        TAG="${IMAGE_NAME}:${VERSION}-${ENVIRONMENT}"
        LATEST_TAG="${IMAGE_NAME}:${ENVIRONMENT}"
    fi

    docker build -t $TAG -t $LATEST_TAG .

    echo -e "${GREEN}Docker image built: $TAG ✓${NC}"
fi

# Push to registry
if [ "$PUSH" = true ]; then
    echo -e "\n${YELLOW}Pushing to container registry...${NC}"

    if [ "$ENVIRONMENT" = "production" ]; then
        docker push "${IMAGE_NAME}:${VERSION}"
        docker push "${IMAGE_NAME}:latest"
    else
        docker push "${IMAGE_NAME}:${VERSION}-${ENVIRONMENT}"
        docker push "${IMAGE_NAME}:${ENVIRONMENT}"
    fi

    echo -e "${GREEN}Image pushed to registry ✓${NC}"
fi

# Deploy based on environment
echo -e "\n${YELLOW}Deploying to ${ENVIRONMENT}...${NC}"

case $ENVIRONMENT in
    staging)
        echo "  Using docker-compose for staging..."
        docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
        ;;
    production)
        echo "  Deploying to Kubernetes..."
        kubectl apply -f kubernetes/
        kubectl rollout status deployment/qbitalabs-api
        ;;
esac

# Health check
echo -e "\n${YELLOW}Running health check...${NC}"
sleep 5

if [ "$ENVIRONMENT" = "staging" ]; then
    HEALTH_URL="http://localhost:8000/health"
else
    HEALTH_URL="https://api.qbitalabs.com/health"
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}Health check passed ✓${NC}"
else
    echo -e "${RED}Health check failed! HTTP code: $HTTP_CODE${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================"
echo "  Deployment Complete!"
echo "========================================"
echo -e "${NC}"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo ""
echo "API URL:"
if [ "$ENVIRONMENT" = "staging" ]; then
    echo "  http://localhost:8000"
else
    echo "  https://api.qbitalabs.com"
fi
