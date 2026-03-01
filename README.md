# life-guard-ai-python

Simple pool detection system in video using OpenCV.

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

## Coordinate example

```python
pool_coordinates = np.array([
    [100, 100],  # Top-left corner
    [500, 120],  # Top-right corner
    [520, 400],  # Bottom-right corner
    [80, 380]    # Bottom-left corner
], dtype=np.int32)
```