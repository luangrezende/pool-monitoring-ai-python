# Pool Monitoring System

Pool monitoring system with object detection using YOLOv8.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
detect_pool.py           # Main application
modules/
  config_manager.py      # Configuration management
  pool_boundary.py       # Pool area definition
  point_selector.py      # Interactive point selection
  object_detector.py     # YOLOv8 detection
config.json              # Application configuration
pool_config.json         # Pool coordinates (auto-generated)
```

## Usage

1. Set video path in `config.json`
2. Run: `python detect_pool.py`
3. First run - select pool boundary points:
   - Click to add points
   - Drag to move
   - Right-click to remove
   - Enter to save (minimum 3 points)
4. Next runs - press 'c' to reconfigure or any key to use saved config

Keys during monitoring:
- `q` - Quit
- `d` - Toggle detection

## Configuration

All settings in `config.json`. No default values - all parameters are required.

### Key parameters

```json
{
  "video": {
    "path": "pool_video.mp4"
  },
  "object_detection": {
    "enabled": true,
    "model_size": "n",
    "confidence_threshold": 0.15,
    "person_confidence_threshold": 0.5,
    "alert_delay_seconds": 2.0,
    "grace_period_seconds": 5.0,
    "skip_frames": 1,
    "process_size": 640
  }
}
```

**Timing:**
- `alert_delay_seconds` - Seconds in zone before alert
- `grace_period_seconds` - Alert persistence after last detection

**Detection:**
- `confidence_threshold` - Minimum confidence for general detection
- `person_confidence_threshold` - Confidence to classify as person (higher = fewer false positives)
- `skip_frames` - Process every N frames (higher = faster)
- `process_size` - Processed frame width (smaller = faster)

## How it Works

Detects objects and people using YOLOv8, checks if detection center is inside pool boundary, triggers alerts based on zone occupation time.

- Detection enters zone → timer starts
- Stays for `alert_delay_seconds` → alert triggers  
- Object leaves zone → alert persists for `grace_period_seconds`

Each tracked object maintains consistent classification (person vs object) throughout its lifetime to prevent alternation.