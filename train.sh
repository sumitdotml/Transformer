#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")"

# Parse command line arguments
RESUME_CHECKPOINT=""
LANG_PAIR="en-ja"
DRY_RUN=""

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

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if ! pip show datasets > /dev/null; then
    echo "Installing dependencies..."
    pip install torch transformers datasets matplotlib tqdm numpy seaborn
fi

# Run the training script
echo "Starting training for ${LANG_PAIR} translation..."
if [ -n "$DRY_RUN" ]; then
    echo "Dry run mode enabled (no actual training)"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
    python transformer/train.py --resume "$RESUME_CHECKPOINT" --lang-pair "$LANG_PAIR" $DRY_RUN
else
    echo "Starting new training run"
    python transformer/train.py --lang-pair "$LANG_PAIR" $DRY_RUN
fi

echo "Training complete!"