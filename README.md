# Auto-boost Algorithm

## Auto-Boost-Next:

_Soon:tm:_

## Version 2.5: SSIMULACRA2&XPSNR-based

The requirements are Vapoursynth, LSMASHSource, Av1an, vszip and a few python library libraries. Optionally: ffmpeg built with XPSNR support, turbo-metrics.

Refined 2.0 with the following additions & changes:
- Proper argument parsing, more control over the script
- Separated the fast encode, metric calculation and zone creation into three independant, callable stages
- Replaced the deprecated `vapoursynth-ssimulacra2` by `vszip`
- Added `turbo-metrics` for GPU-accelerated metrics measurement (Nvidia only)
- Added XPSNR metric and a few zones calculation methods
- Taking advantage of SVT-AV1-PSY quarter-step CRF feature for more granular control
- Possibility to use a more aggressive boosting curve
- And a few other smaller changes...

_Many thanks to R1chterScale, Yiss and Kosaka for iterating on auto-boost and making these amazing contributions!_

## Version 2.0: SSIMULACRA2-based

> Does a fast encode of the provided file, calculates SSIMULACRA2 scores of each chunks and adjusts CRF per-scene to be closer to the average total score, in a _zones.txt_ file to feed Av1an.
The requirements are Vapoursynth, LSMASHSource, fmtconv, Av1an and vapoursynth-ssimulacra2.

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