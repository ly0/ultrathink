#!/bin/bash
# Build the ultrathink frontend for production

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/../ultrathink/web"

echo "Building ultrathink frontend..."
echo ""

cd "$WEB_DIR"

# Check if yarn is installed
if ! command -v yarn &> /dev/null; then
    echo "Error: yarn is not installed"
    echo "Install with: npm install -g yarn"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
yarn install

# Build for production (static export)
echo "Building for production..."
yarn build

echo ""
echo "Frontend built successfully!"
echo "Output directory: $WEB_DIR/out"
echo ""
echo "You can now run: ultrathink serve"
