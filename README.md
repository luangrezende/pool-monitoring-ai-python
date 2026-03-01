# life-guard-ai-python

Pool boundary detection system using OpenCV.

## Features

- **Pool Boundary Detection**: Define and visualize pool area with custom coordinates
- **Modular Architecture**: Clean, maintainable code structure

## Project Structure

```
├── detect_pool.py          # Main application entry point
├── modules/                # Application modules
│   ├── __init__.py         # Module initialization
│   └── pool_boundary.py    # Pool boundary management
└── requirements.txt        # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## How to use

1. Place your pool video (MP4 format) in the project folder
2. Edit the `detect_pool.py` file:
   - Change `video_path` to your video filename
   - Adjust the 4 pool coordinates in `pool_coordinates`:
     - Top-left corner [x, y]
     - Top-right corner [x, y]
     - Bottom-right corner [x, y]
     - Bottom-left corner [x, y]
3. Run the script:

```bash
python detect_pool.py
```

4. Press 'q' to close the video

## Modules

### pool_boundary.py
Manages pool area definition and rendering:
- `PoolBoundary`: Class to handle pool coordinates
- `draw()`: Draws pool boundary on video frames
- `is_point_inside()`: Check if a point is inside pool area
- `get_mask()`: Generate binary mask of pool area

## Example Coordinates

```python
pool_coordinates = [
    [480, 350],   # Top-left corner
    [1440, 560],  # Top-right corner
    [1250, 980],  # Bottom-right corner
    [60, 450]     # Bottom-left corner
]
```