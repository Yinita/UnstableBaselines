# Win Rate Tracking System Guide

## Overview

The win rate tracking system has been successfully integrated into the UnstableBaselines framework. This system tracks the learner model's win rate against itself and specific historical opponents during training and evaluation phases.

## Features

### Core Functionality
- **Overall Win Rate Tracking**: Tracks win rate across all games
- **Opponent-Specific Win Rate**: Tracks win rate against specific models defined in `RECORD_MODELS`
- **Phase Separation**: Separate tracking for training and evaluation phases
- **Real-time Logging**: Metrics are logged to wandb and local logs
- **Statistical Aggregation**: Automatic calculation of win rate statistics

### Integration Points
- **Tracker System**: Extended `trackers.py` with win rate methods
- **Collector Integration**: Modified `collector.py` to pass opponent information
- **Runtime Support**: Updated `runtime.py` and `mixed_play_builder.py` for initialization
- **Comprehensive Testing**: Created test suites to verify functionality

## Configuration

### Environment Variables

Set the `RECORD_MODELS` environment variable to specify which opponents to track:

```python
import os
os.environ["RECORD_MODELS"] = "openai-gpt-4o,openai-gpt-4o-mini,claude-3"
```

### Example Usage

```python
# In your training script
import os
from mixed_play_builder import build_mixed_play

# Configure win rate tracking
os.environ["RECORD_MODELS"] = "openai-gpt-4o,openai-gpt-4o-mini,openai-gpt-5"

# Build and run training
run = build_mixed_play(
    model_name="Qwen/Qwen3-8B",
    train_envs=train_envs,
    eval_envs=eval_envs,
    algorithm="ppo",
    # ... other parameters
)

# Start training with win rate tracking
run.start(learning_steps=1000)
```

## Metrics Structure

### Training Phase Metrics
- `core/train/win_rate_overall`: Overall win rate during training
- `core/train/win_rate_vs_{model_name}`: Win rate against specific models
- `core/train/step`: Training step counter

### Evaluation Phase Metrics
- `core/eval/win_rate_overall`: Overall win rate during evaluation
- `core/eval/win_rate_vs_{model_name}`: Win rate against specific models
- `core/eval/step`: Evaluation step counter

## API Reference

### Tracker Methods

#### `_track_win_rate(phase, is_win, opponent_info=None)`
Records a single game result for win rate statistics.

**Parameters:**
- `phase`: "train" or "eval"
- `is_win`: Boolean indicating if the game was won
- `opponent_info`: Dictionary with opponent information (e.g., `{"name": "gpt-4o"}`)

#### `get_win_rate_stats(phase)`
Retrieves current win rate statistics for a given phase.

**Returns:**
- Dictionary with win rate statistics:
  ```python
  {
      "overall": 0.65,
      "vs_gpt-4o": 0.70,
      "vs_claude-3": 0.60
  }
  ```

#### `log_win_rate_summary(phase)`
Logs a summary of win rate statistics to the console.

### Integration Points

#### Collector Integration
The collector automatically extracts opponent information from game data and passes it to the tracker:

```python
# In collector._post_train()
opponent_info = None
if game_information.names:
    opponent_names = [name for pid, name in game_information.names.items() if pid != traj.pid]
    if opponent_names:
        opponent_info = {"name": opponent_names[0]}
self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id, opponent_info=opponent_info)
```

## Monitoring and Visualization

### Wandb Integration
All win rate metrics are automatically logged to wandb under the `core/` namespace:
- Training metrics: `core/train/win_rate_*`
- Evaluation metrics: `core/eval/win_rate_*`

### Console Logging
Win rate summaries are logged to the console during initialization and can be called manually:

```python
tracker.log_win_rate_summary("train")
tracker.log_win_rate_summary("eval")
```

## Testing

The system includes comprehensive tests to verify functionality:

```bash
# Run win rate tracking tests
python -m pytest test/test_win_rate_simple.py -v
```

## Implementation Details

### Data Structure
Win rates are stored as lists of binary outcomes (1 for win, 0 for loss) in the tracker's `_data` dictionary:

```python
_data = {
    "core/train/win_rate_overall": [1, 0, 1, 1, 0],  # 3 wins, 2 losses
    "core/train/win_rate_vs_gpt-4o": [1, 0, 1],      # 2 wins, 1 loss
}
```

### Aggregation
Win rates are calculated as the mean of binary outcomes during metric aggregation:

```python
win_rate = sum(outcomes) / len(outcomes)
```

### Opponent Filtering
Only opponents specified in `RECORD_MODELS` are tracked individually. Unknown opponents contribute to overall statistics but don't get individual tracking.

## Best Practices

1. **Environment Setup**: Always set `RECORD_MODELS` before initializing the training system
2. **Model Naming**: Use consistent naming for opponent models across training runs
3. **Monitoring**: Regularly check win rate trends in wandb dashboards
4. **Testing**: Run tests after making changes to ensure system integrity

## Troubleshooting

### Common Issues

1. **Missing Win Rate Metrics**: Ensure `RECORD_MODELS` is set before initialization
2. **Opponent Not Tracked**: Check that opponent name matches exactly with `RECORD_MODELS`
3. **Import Errors**: Verify all dependencies are properly imported

### Debug Commands

```python
# Check current win rate statistics
stats = ray.get(tracker.get_win_rate_stats.remote("train"))
print(stats)

# Log win rate summary
ray.get(tracker.log_win_rate_summary.remote("train"))
```

## Future Enhancements

Potential improvements to consider:
- Win rate trend visualization
- Confidence intervals for win rates
- Historical win rate comparison
- Opponent strength estimation
- Dynamic opponent selection based on win rates
