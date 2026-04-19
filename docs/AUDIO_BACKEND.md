# Audio Backend & Silence Handling

## macOS Audio Backend Issue

### Problem
On macOS, loading WAV files with `torchaudio.load()` requires the system to have `libsndfile` installed. Without it, you'll get:

```
RuntimeError: Couldn't find appropriate backend to handle uri ... and format None.
```

This can occur at two points:
1. **During data preparation** - when generating silence clips from background noise
2. **During training** - when loading actual speech command samples

### Solution

#### Option 1: Install libsndfile (Recommended)

```bash
brew install libsndfile
```

Or use the provided setup script:
```bash
bash scripts/setup_audio_backend.sh
```

#### Option 2: Graceful Degradation (Automatic, Built-In)

If `libsndfile` is not installed, the system automatically:

1. **During data preparation**:
   - Logs warnings about failed noise file loads
   - Generates silence as **zero tensors** instead
   - Continues successfully

2. **During training**:
   - Sets `num_workers=0` automatically on macOS
   - Catches backend errors and replaces problematic samples with **white noise** fallbacks
   - Logs warnings about skipped samples
   - Continues training seamlessly

This means training never fails - you can proceed immediately and get results, though with less realistic audio.

## How Silence is Generated

The `_make_silence_clips()` function in `src/data/dataset.py`:

1. **Tries to load real background noise** from the dataset's `_background_noise_/` folder
2. **On error**: Catches `RuntimeError`, logs a warning, skips that file
3. **If all files fail**: Generates silence as **zero tensors** with proper dimensions

## How Speech Samples are Loaded

The `__getitem__` method in `src/data/dataset.py`:

1. **Tries to load the WAV file normally** via `torchaudio`
2. **On error**: Catches `RuntimeError`, logs a warning
3. **Generates fallback audio**: White noise with low amplitude (0.1x scaling)
4. **Returns fallback with correct label**: Training continues normally

## Automatic Compatibility Fixes

The training scripts (`scripts/train.py`, `scripts/evaluate.py`) automatically:

- **Detect macOS** and set `num_workers=0` to avoid multiprocessing backend issues
- **Detect MPS device** and disable `pin_memory` (not supported on MPS)
- **Log warnings** about compatibility adjustments

## Silence Class Statistics

The silence class is generated as a fraction of the training set:

- **Training split**: 10% of training samples (configurable via `SILENCE_FRACTION` in `src/utils/constants.py`)
- **Validation split**: Fixed 260 samples (configurable via `N_SILENCE_FIXED`)
- **Test split**: Fixed 260 samples

## Impact on Results

| Scenario | Silence Quality | Training Impact |
|----------|-----------------|-----------------|
| libsndfile installed | Real background noise | Optimal model performance |
| Data prep only, libsndfile missing | Zero tensors | Slightly reduced recall on silence, but continues |
| Training, libsndfile missing, num_workers>0 | Real or white noise fallback | Runtime errors prevented; minor accuracy impact |
| Training, libsndfile missing, num_workers=0 (auto) | Real or white noise fallback | **No errors, training succeeds** (automatically set on macOS) |

## Best Practices

1. **Best**: Install `libsndfile` for optimal results
   ```bash
   brew install libsndfile
   ```

2. **Good**: Use the system as-is on macOS (automatic fallbacks included)
   - No action needed
   - Training will auto-adjust (`num_workers=0`, error handling in place)
   - Minor accuracy impact but fully functional

3. **Avoid**: Manually setting `num_workers>0` on macOS without `libsndfile`
   - Will cause runtime errors in DataLoader worker processes
   - System can't inject error handlers into worker processes
   - (Automatically fixed by the training scripts, so avoid manually overriding)

