# Auto-boost Algorithms

## Version 1.0: brightness-based

> Gets the average brightness of a scene and lowers CQ/CRF/Q the darker the scene is, in a _zones.txt_ file to feed Av1an.
Only requirements are Vapoursynth and vstools.

__Usage:__
```
python auto-boost_1.0.py "{animu.mkv}" "{scenes.json}" {base CQ/CRF/Q} "{encoder: aom/svt-av1/rav1e (optional)}"
```

__Example:__
```
python auto-boost_1.0.py "path/to/nice_boat.mkv" "path/to/scenes.json" 30
```
