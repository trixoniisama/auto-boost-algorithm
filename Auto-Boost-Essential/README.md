# Auto-Boost-Essential

Auto-Boost-Essential is the latest iteration of the Auto-Boost formula, streamlined and refined for greater convenience, speeds and boosting gains!

This encoding script is intended to be paired with my [SVT-AV1-Essential](https://github.com/nekotrix/SVT-AV1-Essential) encoder fork.  
SVT-AV1-Essential sports *excellent* quality consistency, but Auto-Boost-Essential offers *exceptional* consistency!

**Here is how it works:** the script runs a first encoder fass-pass, finds scenes based on the introduced keyframes, calculate metrics scores, automatically adjusts the CRF of scenes in order to increase quality consistency and then runs a final-pass with these adjustements.  

The quality metric at play this time again is SSIMULACRA2.

Auto-Boost-Essential can be considered a helper script, as all you need to do is provide an input video file and it will manage everything for you:
```bash
python Auto-Boost-Essential.py "my_video_file.mp4"
```

Results:
|                    Metrics                    |                    Speed                    |
|-----------------------------------------------|---------------------------------------------|
| ![Metrics](https://i.kek.sh/2Ulmd7e7zIJ.webp) | ![Speed](https://i.kek.sh/0fehFRVGuhT.webp) |
| ![Metrics](https://i.kek.sh/WckNMr7IzRa.webp) | ![Speed](https://i.kek.sh/NHYJEEeJrhB.webp) |  

*Speed may vary depending on your hardware configuration and source resolution.*

The above results are not even best-case scenarios. The selected samples are very complex. More gains are expected on your average clip, granted it contains more than one scene at a minimum!

The script is also capable of resuming unfinished encodes, and can also be run with boosting disabled!