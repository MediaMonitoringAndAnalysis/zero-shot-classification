# Zero-Shot Classification

A two-stage zero-shot classifier that uses a smaller model for initial filtering and a larger LLM for more precise classification.

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/zero-shot-classification.git
```

## Usage

```python
from zero_shot_classification import MultiStepZeroShotClassifier

# Initialize the classifier
classifier = MultiStepZeroShotClassifier(
    second_pass_model="gpt-4o-mini",
    second_pass_pipeline="OpenAI",
    second_pass_api_key="your-openai-api-key",  # Optional
    first_pass_model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # Optional, this is the default
    first_pass_threshold=0.25,  # Optional, this is the default
    batch_size=8,  # Optional, this is the default
    device="cpu"  # Optional, defaults to best available device
)

# Define your text and tags
texts = [
    "The humanitarian crisis in Gaza has reached unprecedented levels...",
    "Access to basic necessities such as food, water, and medical supplies remains severely limited..."
]

tags = [
    "Humanitarian Crisis",
    "Displacement",
    "Conflict",
    "Aid Organizations",
    "Cooperation"
]

# Run classification
results = classifier(texts, tags)
```

## Features

- Two-stage classification pipeline
- First pass uses a smaller multilingual model for efficient filtering
- Second pass uses a larger LLM for precise classification
- Batch processing support
- Configurable models and thresholds

## Package structure

```text
zero-shot-classification/
├── setup.py
├── README.md
├── LICENSE
├── .gitignore
└── zero_shot_classification/
    ├── __init__.py
    └── multistep_zero_shot_classification.py
```

## Requirements

- Python >= 3.8
- See setup.py for package dependencies

## License

Affero GPL License v3.0