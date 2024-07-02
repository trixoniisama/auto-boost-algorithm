import statistics
from math import ceil
import json
import sys
from vapoursynth import core
from vstools import clip_async_render

if "--help" in sys.argv[1:]:
    print('Usage:\npython auto-boost_1.0.py "{animu.mkv}" "{scenes.json}" {base CQ/CRF/Q} "{encoder: aom/svt-av1/rav1e (optional)}"\n\nExample:\npython "auto-boost_1.0.py" "path/to/nice_boat.mkv" "path/to/scenes.json" 30')
    exit(0)
else:
    pass

og_cq = int(sys.argv[3]) # CQ to start from
br = 10 # maximum CQ change from original
bl = og_cq-br # hard cap how low CQ can be set by boost

try:
    ENCODER = sys.argv[4] # select encoder between aom, svt-av1 and rav1e :allhailav1:
except:
    ENCODER = "svt-av1"

def get_ranges(scenes):
     ranges = []
     ranges.insert(0,0)
     with open(scenes, "r") as file:
        content = json.load(file)
        for i in range(len(content['scenes'])):
            ranges.append(content['scenes'][i]['end_frame'])
        return ranges

def get_brightness(video, start, end):
        brightness = []
        ref = video[:].std.PlaneStats(plane=0)

        render = clip_async_render(
             ref, outfile=None, progress=f'Getting frame props... from {start} to {end}',
             callback=lambda _, f: f.props.copy()
        )
        props = [prop['PlaneStatsAverage'] for prop in render]

        for prop in props:
                    brightness.append(prop)

        brig_geom = round(statistics.geometric_mean([x+0.01 for x in brightness]), 2) #x+1
        #print(brig_geom)

        return brig_geom

def boost(br_geom):
        global br, bl
        global og_cq

        if br_geom < 0.5: # too dark, cq needs to change
            new_cq = og_cq - ceil((0.5 - br_geom) / 0.5 * br)

            if new_cq < bl: # Cap on boosting
                new_cq = bl

            return new_cq

iter = 0
def zones_txt(beginning_frame, end_frame, cq, zones_loc):
    global iter
    iter += 1

    with open(zones_loc, "w" if iter == 1 else "a") as file:
        if ENCODER == "aom":
            file.write(f"{beginning_frame} {end_frame} aom --cq-level={cq}\n")
        elif ENCODER == "svt-av1":
            file.write(f"{beginning_frame} {end_frame} svt-av1 --crf {cq}\n")
        elif ENCODER == "rav1e":
            file.write(f"{beginning_frame} {end_frame} rav1e --quantizer {cq}\n")
        else:
            print("Incompatible encoder given.")
            exit(-1)

def zones_main(chunk, start, end, zones_loc):
        global og_cq

        br = get_brightness(chunk, start, end) # brightness range is [0,1]

        cq = boost(br)

        if og_cq != cq and cq != None:
            print(f'Enc:  [{start}:{end}]\n'
                    f'Avg brightness: {br}\n'
                    f'Adjusted CQ: {cq}\n\n')
            zones_txt(start, end, cq, f"{zones_loc}zones.txt")
        elif cq == None:
                print(f"cq = None (brightness > 0.5, actually: {br})")

scenes_loc = sys.argv[2] # scene file is expected to be named 'scenes.json'
ranges = get_ranges(scenes_loc)
src = core.lsmas.LWLibavSource(source=sys.argv[1], cache=0)
for i in range(len(ranges)-1):
    #print(f"[{ranges[i]}:{ranges[i+1]}]")
    zones_main(src[ranges[i]:ranges[i+1]], ranges[i], ranges[i+1], scenes_loc[:-11])
