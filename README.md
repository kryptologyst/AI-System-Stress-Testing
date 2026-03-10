# AI System Stress Testing

**DISCLAIMER**: This project is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run stress tests**:
```bash
python scripts/run_stress_tests.py --config configs/default.yaml
```

3. **Launch interactive demo**:
```bash
streamlit run demo/app.py
```

## Features

- **Adversarial Attack Testing**: FGSM, PGD attacks with robustness curves
- **Uncertainty Quantification**: Monte Carlo Dropout, Temperature Scaling
- **Out-of-Distribution Detection**: Energy-based, Max Softmax, Entropy methods
- **Model Calibration**: Reliability diagrams, ECE, Brier score
- **Interactive Demo**: Streamlit interface for real-time testing

## Project Structure

```
├── src/                    # Core source code
│   ├── models/            # Model implementations
│   ├── attacks/           # Adversarial attack methods
│   ├── uncertainty/       # Uncertainty quantification
│   ├── ood/              # Out-of-distribution detection
│   ├── data/             # Data loading and preprocessing
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── demo/                 # Streamlit demo application
└── assets/               # Generated plots and results
```

## Configuration

Edit `configs/default.yaml` to customize:
- Dataset parameters (samples, features, classes)
- Model architecture and training settings
- Attack methods and epsilon values
- Uncertainty quantification methods
- OOD detection methods

## Example Usage

```python
from src.stress_tester import StressTester
import yaml

# Load configuration
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize stress tester
tester = StressTester(config)

# Run comprehensive stress tests
results = tester.run_stress_tests()
```

## Results

The framework generates:
- **Robustness curves** showing accuracy vs attack strength
- **Reliability diagrams** for calibration assessment
- **OOD detection plots** comparing ID vs OOD score distributions
- **Summary report** with key metrics and recommendations

## Limitations

- Results are based on synthetic data and may not generalize
- Adversarial examples are artificially generated
- Uncertainty estimates are approximations
- Stress testing is not a substitute for comprehensive validation

## Contributing

This project follows reproducibility guidelines with deterministic seeding and structured logging.

## License

For educational and research purposes only.# AI-System-Stress-Testing
