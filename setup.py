from setuptools import setup, find_packages

setup(
    name="personalized-query-rewriter",
    version="0.1.0",
    description="Personalized Query Rewriter for Chase UEP: FT-Transformer → E2P → Gemma3-1B LoRA",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov"],
        "full": ["bitsandbytes>=0.43.0", "accelerate>=0.30.0", "wandb>=0.16.0"],
    },
)
