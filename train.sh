#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")"

# Parse command line arguments
RESUME_CHECKPOINT=""
LANG_PAIR="en-ja"
DRY_RUN=""
EPOCHS=""
LIMIT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume)
      RESUME_CHECKPOINT="$2"
      shift # past argument
      shift # past value
      ;;
    --lang-pair)
      LANG_PAIR="$2"
      shift # past argument
      shift # past value
      ;;
    --epochs)
      EPOCHS="--epochs $2"
      shift # past argument
      shift # past value
      ;;
    --limit)
      LIMIT="--limit $2"
      shift # past argument
      shift # past value
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift # past argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for the current session if not already there
    if ! command -v uv &> /dev/null; then
        echo "Adding uv to PATH..."
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "Failed to install uv. Please install it manually:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    echo "uv installed successfully."
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if ! uv pip show datasets &> /dev/null; then
    echo "Installing dependencies with uv..."
    uv pip install -r requirements.txt
fi

# Run the training script
echo "Starting training for ${LANG_PAIR} translation..."
if [ -n "$DRY_RUN" ]; then
    echo "Dry run mode enabled (no actual training)"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
    python transformer/train.py --resume "$RESUME_CHECKPOINT" --lang-pair "$LANG_PAIR" $EPOCHS $LIMIT $DRY_RUN
else
    echo "Starting new training run"
    python transformer/train.py --lang-pair "$LANG_PAIR" $EPOCHS $LIMIT $DRY_RUN
fi

echo "Training complete!"