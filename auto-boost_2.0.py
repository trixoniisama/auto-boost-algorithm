import statistics
from math import ceil
import json
import sys
import subprocess
import vapoursynth as vs
core = vs.core

if "--help" in sys.argv[1:]:
    print('Usage:\npython auto-boost_2.0.py "{animu.mkv}" {base CQ/CRF/Q}"\n\nExample:\npython "auto-boost_2.0.py" "path/to/nice_boat.mkv" 30')
    exit(0)
else:
    pass

og_cq = int(sys.argv[2]) # CQ to start from
br = 10 # maximum CQ change from original

def get_ranges(scenes):
     ranges = []
     ranges.insert(0,0)
     with open(scenes, "r") as file:
        content = json.load(file)
        for i in range(len(content['scenes'])):
            ranges.append(content['scenes'][i]['end_frame'])
        return ranges

iter = 0
def zones_txt(beginning_frame, end_frame, cq, zones_loc):
    global iter
    iter += 1

    with open(zones_loc, "w" if iter == 1 else "a") as file:
        file.write(f"{beginning_frame} {end_frame} svt-av1 --crf {cq}\n")

def calculate_standard_deviation(score_list: list[int]):
    filtered_score_list = [score for score in score_list if score >= 0]
    sorted_score_list = sorted(filtered_score_list)
    average = sum(filtered_score_list)/len(filtered_score_list)
    return (average, sorted_score_list[len(filtered_score_list)//20])

fast_av1an_command = f'av1an -i "{sys.argv[1]}" --temp "{sys.argv[1][:-4]}/temp/" -y \
                    --verbose --keep --split-method av-scenechange -m lsmash \
                    --min-scene-len 12 -c mkvmerge --sc-downscale-height 480 \
                    --set-thread-affinity 2 -e svt-av1 --force -v \" \
                    --preset 9 --crf {og_cq} --rc 0 --film-grain 0 --lp 2 \
                    --scm 0 --keyint 0 --fast-decode 1 --color-primaries 1 \
                    --transfer-characteristics 1 --matrix-coefficients 1 \" \
                    --pix-format yuv420p10le -x 240  -w {WORKERS} \
                    -o "{sys.argv[1][:-4]}_fastpass.mkv"'

p = subprocess.Popen(fast_av1an_command, shell=True)
exit_code = p.wait()

if exit_code != 0:
    print("Av1an encountered an error, exiting.")
    exit(-2)

scenes_loc = f"{sys.argv[1][:-4]}/temp/scenes.json"
ranges = get_ranges(scenes_loc)

src = core.lsmas.LWLibavSource(source=sys.argv[1], cache=0)
enc = core.lsmas.LWLibavSource(source=f"{sys.argv[1][:-4]}_fastpass.mkv", cache=0)

print(f"source: {len(src)} frames")
print(f"encode: {len(enc)} frames")

source_clip = src.resize.Bicubic(format=vs.RGBS, matrix_in_s='709').fmtc.transfer(transs="srgb", transd="linear", bits=32)
encoded_clip = enc.resize.Bicubic(format=vs.RGBS, matrix_in_s='709').fmtc.transfer(transs="srgb", transd="linear", bits=32)

percentile_5_total = []
total_ssim_scores: list[int] = []

skip = 10 # amount of skipped frames

for i in range(len(ranges)-1):
    cut_source_clip = source_clip[ranges[i]:ranges[i+1]].std.SelectEvery(cycle=skip, offsets=0)
    cut_encoded_clip = encoded_clip[ranges[i]:ranges[i+1]].std.SelectEvery(cycle=skip, offsets=0)
    result = cut_source_clip.ssimulacra2.SSIMULACRA2(cut_encoded_clip)
    chunk_ssim_scores: list[int] = []

    for index, frame in enumerate(result.frames()):
        score = frame.props['_SSIMULACRA2']
        # print(f'Frame {index}/{result.num_frames}: {score}')
        chunk_ssim_scores.append(score)
        total_ssim_scores.append(score)

    (average, percentile_5) = calculate_standard_deviation(chunk_ssim_scores)
    percentile_5_total.append(percentile_5)

(average, percentile_5) = calculate_standard_deviation(total_ssim_scores)
print(f'Median score:  {average}\n\n')

for i in range(len(ranges)-1):

    new_cq = og_cq - ceil((1.0 - (percentile_5_total[i]/average)) / 0.5 * 10) # trust me bro

    if new_cq < og_cq-br: # set lowest allowed cq
        new_cq = og_cq-br

    if new_cq > og_cq+br: # set highest allowed cq
        new_cq = og_cq+br

    print(f'Enc:  [{ranges[i]}:{ranges[i+1]}]\n'
            f'Chunk 5th percentile: {percentile_5_total[i]}\n'
            f'Adjusted CRF: {new_cq}\n\n')
    zones_txt(ranges[i], ranges[i+1], new_cq, f"{scenes_loc[:-11]}zones.txt")

# yes, this is messier than the 1.0 code, laziness won over me, deal with it
