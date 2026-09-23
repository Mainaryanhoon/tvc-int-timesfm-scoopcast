import numpy as np

print("1. Loading TimesFM 3.0 Evaluator...")
try:
    from timesfm3 import TimesFM3Evaluator, ModelConfig
except ImportError as e:
    print("❌ Error: Could not import timesfm3.")
    print("Make sure you have pulled the latest TimesFM repository from Google Research and installed it in your .venv.")
    raise e

print("\n2. Initializing TimesFM 3.0 (google/timesfm-3.0-pytorch) on CPU...")
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=8,
    device="cpu"  # Using CPU for testing stability
)
forecaster = TimesFM3Evaluator(config)
print("✅ TimesFM 3.0 initialized successfully!")

print("\n3. Testing Native Multivariate Forecasting with Covariates...")
context_len = 128
horizon = 24

# Mocking 3 joint target variates across past context
# Shape: (3 variates, 128 context)
target = np.random.randn(3, context_len).astype(np.float32)

# Mocking 1 past-only covariate channel across past context
# Shape: (1 channel, 128 context)
past_only_cov = np.random.randn(1, context_len).astype(np.float32)

# Mocking 2 past-and-future covariate channels across context + horizon
# Shape: (2 channels, 152 length)
past_future_cov = np.random.randn(2, context_len + horizon).astype(np.float32)

print(f"   Inputs generated: Target {target.shape}, Past Cov {past_only_cov.shape}, Past+Future Cov {past_future_cov.shape}")

# Generate joint forecast across all 3 target variates
outputs = list(
    forecaster.predict_batch(
        contexts=[target],
        horizon=horizon,
        past_only_covariates=[past_only_cov],
        past_future_covariates=[past_future_cov],
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)

print("\n✅ Forecast generated successfully in a single forward pass!")
print("Multivariate forecast shape:", outputs[0].forecast.shape)   # Expected: (3, 24)
print("Multivariate quantiles shape:", outputs[0].quantiles.shape) # Expected: (3, 24, 9)

if outputs[0].forecast.shape == (3, 24):
    print("\n🚀 SUCCESS: TimesFM-3 Machine Validation Passed! You are ready for Phase 2.")
else:
    print("\n❌ FAILED: Unexpected tensor shapes returned.")