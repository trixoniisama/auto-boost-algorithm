# Auto-boost Algorithm

## Version 1.0: brightness-based

> Gets the average brightness of a scene and lowers CQ/CRF/Q the darker the scene is, in a _zones.txt_ file to feed Av1an.
The requirements are Vapoursynth, vstools and LSMASHSource.

__Usage:__
```
python auto-boost_1.0.py "{animu.mkv}" "{scenes.json}" {base CQ/CRF/Q} "{encoder: aom/svt-av1/rav1e (optional)}"
```

__Example:__
```
python auto-boost_1.0.py "path/to/nice_boat.mkv" "path/to/scenes.json" 30
```

__Advantages:__
- Fast
- No bs
- Solves one long-lasting issue of AV1 encoders: low bitrate allocation in dark scenes

__Known limitations:__
- Not every dark scene is made equal, brightness is not a great enough metric to determine whether CRF should be decreased or not
- CRF is boosted to the max during credits
- Script now entirely irrelevant with SVT-AV1-PSY's new frame-luma-bias feature

_Inspiration was drawn from the original Av1an (python) boosting code_

## Version 2.0: SSIMULACRA2-based

> Does a fast encode of the provided file, calculates SSIMULACRA2 scores of each chunks and adjusts CRF per-scene to be closer to the average total score, in a _zones.txt_ file to feed Av1an.
The requirements are Vapoursynth, vstools, LSMASHSource, fmtconv, Av1an and vapoursynth-ssimulacra2.

__Usage:__
```
python auto-boost_2.0.py "{animu.mkv}" {base CQ/CRF/Q}
```

__Example:__
```
python auto-boost_2.0.py "path/to/nice_boat.mkv" 30
```

__Advantages:__
- Lower quality deviation of individual scenes in regards to the entire stream
- Better allocates bitrate in more complex scenes and compensates by giving less bitrate to scenes presenting some headroom for further compression

__Known limitations:__
- Slow process
- No bitrate cap in place so the size of complex scenes can go out of hand
- The SSIMULACRA2 metric is not ideal, plus the score alone is not representative enough of if a CRF adjustement is relevant in the context of that scene (AI will save)

_Borrowed some code from Sav1or's SSIMULACRA2 script_

## Version 3.0: SSIMULACRA2-based + per-scene grain synthesis strength determination

...and a few other improvements.

_Soon:tm:_
