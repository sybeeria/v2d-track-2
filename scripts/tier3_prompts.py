"""Prepare SAM2 prompts and inspection PNGs from exo video pixels only.

Annotations use coordinates in a 512x384 preview. No trajectory/scene metadata is
opened. IDs follow the upstream contract: human=0, manipulated object=1.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


def read_frame(video: Path, frame: int, *, preview=False) -> Image.Image:
    select = f'select=eq(n\\,{frame})' + (',scale=512:384' if preview else '')
    data = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video),
                                   '-vf', select, '-frames:v', '1', '-f', 'image2pipe',
                                   '-vcodec', 'png', '-threads', '1', '-'])
    return Image.open(io.BytesIO(data)).convert('RGB')


def make_prompts(annotation, width, height, count):
    prompts = []
    for object_id, role in enumerate(('human', 'object')):
        item = annotation[role]
        frame = item.get('frame', 0)
        box = item['box']
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 512 and 0 <= box[1] < box[3] <= 384):
            raise ValueError('invalid 512x384 preview box')
        if not isinstance(frame, int) or not 0 <= frame < count:
            raise ValueError('prompt frame outside video')
        scaled = [int(round(v * (width / 512 if j % 2 == 0 else height / 384))) for j, v in enumerate(box)]
        points = [[float(x * width / 512), float(y * height / 384)] for x,y in item.get('points', [])]
        labels = item.get('point_labels', [1] * len(points))
        if len(labels) != len(points) or any(v not in (0, 1) for v in labels):
            raise ValueError('invalid point labels')
        if any(not (0 <= x < width and 0 <= y < height) for x,y in points):
            raise ValueError('point outside video')
        prompts.append({'frame_index': frame, 'object_id': object_id, 'role': role,
                        'points': [{'x': x, 'y': y} for x,y in points], 'point_labels': labels,
                        'box': None if points else dict(zip(('x0','y0','x1','y1'), scaled))})
    return {'prompts': prompts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=list(range(30)))
    p.add_argument('--annotations', type=Path)
    a = p.parse_args()
    annotations = json.loads(a.annotations.read_text()) if a.annotations else None
    a.out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for index in a.episodes:
        video = a.videos / f'episode_{index:06d}.mp4'
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))['streams'][0]
        count, width, height = int(info['nb_frames']), int(info['width']), int(info['height'])
        frames = [0, count // 2, count - 1]
        sheet = Image.new('RGB', (1536,384))
        for j, frame in enumerate(frames):
            im = read_frame(video, frame, preview=True)
            d = ImageDraw.Draw(im); d.rectangle((0,0,160,22), fill='black')
            d.text((5,5), f'ep{index:02d} frame {frame}', fill='white')
            sheet.paste(im, (j*512,0))
        sheet.save(a.out / f'ep{index:02d}_samples.jpg')
        record = {'episode': index, 'video': str(video.resolve()), 'frames': count,
                  'width': width, 'height': height, 'sample_frames': frames,
                  'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest()}
        if annotations:
            annotation = annotations[str(index)]
            prompts = make_prompts(annotation, width, height, count)
            (a.out / f'ep{index:02d}.json').write_text(json.dumps(prompts, indent=2)+'\n')
            prompt_frames = sorted({v['frame_index'] for v in prompts['prompts']})
            overlay = Image.new('RGB', (width * len(prompt_frames), height))
            for j, frame in enumerate(prompt_frames):
                im = read_frame(video, frame)
                d = ImageDraw.Draw(im)
                for prompt in prompts['prompts']:
                    if prompt['frame_index'] != frame:
                        continue
                    b = prompt['box']
                    xy = [b[k] for k in ('x0','y0','x1','y1')] if b else [int(round(v * (width/512 if k%2==0 else height/384))) for k,v in enumerate(annotation[prompt['role']]['box'])]
                    color = '#00ff8a' if prompt['object_id'] == 0 else '#ff4b41'
                    d.rectangle(xy, outline=color, width=5)
                    d.text((xy[0], max(0,xy[1]-16)), f"{prompt['role']} {prompt['object_id']}", fill=color)
                    for point, label in zip(prompt['points'],prompt['point_labels']):
                        x,y = point['x'],point['y']
                        d.ellipse((x-6,y-6,x+6,y+6), fill=color if label else 'black')
                d.text((20,20),f'ep{index:02d} frame {frame} | VIDEO-ONLY PROMPT; NOT A SEGMENTATION',fill='white')
                overlay.paste(im,(j*width,0))
            overlay.save(a.out / f'ep{index:02d}_overlay.png')
            record.update(target=annotation['target'], flags=annotation.get('flags',[]),
                          evidence='inferred target/boxes from video pixels; SAM2 propagation untested')
        inventory.append(record)
    (a.out / 'video_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'episodes':len(inventory),'frames':sum(x['frames'] for x in inventory)}))


if __name__ == '__main__':
    main()
