#
# Animation Script v6.0
# Inspired by Deforum Notebook
# Must have ffmpeg installed in path.
# Poor img2img implentation, will trash images that aren't moving.
#
# See https://github.com/Animator-Anon/Animator

import os
import time
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images, sd_models
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state
import random
import subprocess
import numpy as np
import pandas as pd
import json
import cv2
import glob
import shutil
import piexif
import piexif.helper

from PIL import Image, ImageFilter, ImageDraw, ImageFont


def zoom_at2(img, rot, x, y, zoom):
    w, h = img.size

    # Zoom image
    img2 = img.resize((int(w * zoom), int(h * zoom)), Image.Resampling.LANCZOS)

    # Create background image
    padding = 2
    resimg = addnoise(img.copy(), 0.75).resize((w + padding * 2, h + padding * 2), Image.Resampling.LANCZOS). \
        filter(ImageFilter.GaussianBlur(5)). \
        crop((padding, padding, w + padding, h + padding))

    resimg.paste(img2.rotate(rot), (int((w - img2.size[0]) / 2 + x), int((h - img2.size[1]) / 2 + y)))

    return resimg


def get_pnginfo(filepath):

    image = Image.open(filepath, "r")
    worked = False

    if image is None:
        return worked, '', 'Error: No image supplied'

    items = image.info
    geninfo = ''

    if "exif" in image.info:
        exif = piexif.load(image.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        items['exif comment'] = exif_comment
        geninfo = exif_comment

        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
                      'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)

    geninfo = items.get('parameters', geninfo)

    info = ''
    for key, text in items.items():
        info += f"{str(key).strip()}:{str(text).strip()}".strip()+"\n"

    if len(info) == 0:
        info = "Error: Nothing found in the image."
    else:
        worked = True

    return worked, geninfo, info


def pasteprop(img, props, propfolder):
    img2 = img.convert('RGBA')

    for propname in props:
        # prop_name | prop filename | x pos | y pos | scale | rotation
        propfilename = os.path.join(propfolder.strip(), str(props[propname][1]).strip())
        x = int(props[propname][2])
        y = int(props[propname][3])
        scale = float(props[propname][4])
        rotation = float(props[propname][5])

        if not os.path.exists(propfilename):
            print("Prop: Cannot locate file: " + propfilename)
            return img

        prop = Image.open(propfilename)
        w2, h2 = prop.size
        prop2 = prop.resize((int(w2 * scale), int(h2 * scale)), Image.Resampling.LANCZOS).rotate(rotation, expand=True)
        w3, h3 = prop2.size

        tmplayer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        tmplayer.paste(prop2, (int(x - w3 / 2), int(y - h3 / 2)))
        img2 = Image.alpha_composite(img2, tmplayer)

    return img2.convert("RGB")


def rendertext(img, textblocks):
    pad = 1  # Rounding and edge padding of the bubble background.
    d1 = ImageDraw.Draw(img)
    font_size = 20
    for textname in textblocks:
        # textblock_name | text_prompt | x | y | w | h | back_color | white_color | font_filename
        textprompt = str(textblocks[textname][1]).strip().replace('\\n', '\n')
        x = int(textblocks[textname][2])
        y = int(textblocks[textname][3])
        w = int(textblocks[textname][4])
        h = int(textblocks[textname][5])
        # Try convert text to a tuple (255,255,255) or just leave as text "white"
        try:
            backcolor = eval(textblocks[textname][6].strip())
        except:
            backcolor = textblocks[textname][6].strip()
        try:
            forecolor = eval(textblocks[textname][7].strip())
        except:
            forecolor = textblocks[textname][7].strip()
        font_name = str(textblocks[textname][8]).strip().lower()
        # Auto size the text.
        for fs in range(70):
            myfont = ImageFont.truetype(font_name, fs)
            txtsize = d1.multiline_textbbox((0, 0), textprompt, font=myfont, align='center')
            if txtsize[2] - txtsize[0] > (w - pad * 2) or txtsize[3] - txtsize[1] > (h - pad * 2):
                font_size = fs - 1
                break

        myfont = ImageFont.truetype(font_name, font_size)
        # print(f"size:{font_size} loc:{x}, {y} size:{w}, {h}")

        txtsize = d1.multiline_textbbox((0, 0), textprompt, font=myfont, align='center')
        # print(f"txtsize:{txtsize}")

        d1.rounded_rectangle((x, y, x + w, y + h), radius=pad, fill=backcolor)
        d1.multiline_text((x + pad, y + pad + (h - txtsize[3]) / 2), textprompt, fill=forecolor, font=myfont,
                          align='center')

    return img


def addnoise(img, percent):
    # Draw coloured circles randomly over the image. Lame, but for testing.
    # print("Noise function")
    w2, h2 = img.size
    draw = ImageDraw.Draw(img)
    for i in range(int(50 * float(percent))):
        x2 = random.randint(0, w2)
        y2 = random.randint(0, h2)
        s2 = random.randint(0, int(50 * float(percent)))
        pos = (x2, y2, x2 + s2, y2 + s2)
        draw.ellipse(pos, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     outline=(0, 0, 0))
    return img


def morph(img1, img2, count):
    """
    count=4
    img1:0
            0.2 (1/5)
            0.4 (2/5)
            0.6 (3/5)
            0.8 (4/5)
    img2:1
    """
    arr1 = np.array(img1)
    diff = (np.array(img2).astype('int16') - arr1.astype('int16'))
    img_list = []
    for x in range(1, count + 1):
        img_list.append(Image.fromarray((arr1 + diff * (x / (count + 1))).astype('uint8'), 'RGB'))

    return img_list


def make_gif(filepath, filename, fps, create_vid, create_bat):
    # Create filenames
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.gif"
    # Build cmd for bat output, local file refs only
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        out_filename
    ]
    # create bat file
    if create_bat:
        with open(os.path.join(filepath, "makegif.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])#, "\r\n", "pause"])
    # Fix paths for normal output
    cmd[5] = os.path.join(filepath, in_filename)
    cmd[6] = os.path.join(filepath, out_filename)
    # create output if requested
    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_webm(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.webm"

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-crf', str(50),
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makewebm.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])#, "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[10] = os.path.join(filepath, out_filename)

    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_mp4(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.mp4"

    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makemp4.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])#, "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[16] = os.path.join(filepath, out_filename)

    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class Script(scripts.Script):

    def title(self):
        return "Animator v6"

    def show(self, is_img2img: bool):
        return True

    def ui(self, is_img2img: bool):

        if is_img2img:
            i1 = gr.HTML("<p>Running in img2img mode:<br><br>Render these video formats:</p>")
        else:
            i1 = gr.HTML("<p>Running in txt2img mode:<br><br>Render these video formats:</p>")
        with gr.Row():
            vid_gif = gr.Checkbox(label="GIF", value=False)
            vid_mp4 = gr.Checkbox(label="MP4", value=False)
            vid_webm = gr.Checkbox(label="WEBM", value=True)

        i2 = gr.HTML("<p style=\"margin-bottom:0.75em\">Animation Parameters</p>")
        with gr.Row():
            total_time = gr.Textbox(label="Total Animation Length (s)", lines=1, value="10.0")
            fps = gr.Textbox(label="Framerate", lines=1, value="15")
            smoothing = gr.Slider(label="Smoothing_Frames", minimum=0, maximum=32, step=1, value=0)
        with gr.Row():
            add_noise = gr.Checkbox(label="Add_Noise", value=False)
            noise_strength = gr.Slider(label="Noise Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.10)

        i3 = gr.HTML("<p style=\"margin-bottom:0.75em\">Initial Parameters</p>")
        with gr.Row():
            with gr.Column():
                denoising_strength = gr.Slider(label="Denoising Strength", minimum=0.0,
                                               maximum=1.0, step=0.01, value=0.40)
                seed_march = gr.Checkbox(label="Seed_March", value=False)
            with gr.Column():
                zoom_factor = gr.Textbox(label="Zoom Factor (scale/s)", lines=1, value="1.0")
                x_shift = gr.Textbox(label="X Pixel Shift (pixels/s)", lines=1, value="0")
                y_shift = gr.Textbox(label="Y Pixel Shift (pixels/s)", lines=1, value="0")
                rotation = gr.Textbox(label="Rotation (deg/s)", lines=1, value="0")

        i4 = gr.HTML("<p style=\"margin-bottom:0.75em\">Prompt Template, applied to each keyframe below</p>")
        tmpl_pos = gr.Textbox(label="Positive Prompts", lines=1, value="")
        tmpl_neg = gr.Textbox(label="Negative Prompts", lines=1, value="")

        i5 = gr.HTML("<p style=\"margin-bottom:0.75em\">Props, Stamps</p>")
        propfolder = gr.Textbox(label="Poper_Folder:", lines=1, value="")

        i6 = gr.HTML(
            "<p>Supported Keyframes:<br>"
            "time_s | source | video, images, img2img | path<br>"
            "time_s | prompt | positive_prompts | negative_prompts<br>"
            "time_s | template | positive_prompts | negative_prompts<br>"
            "time_s | prompt_from_png | file_path<br>"
            "time_s | transform | zoom | x_shift | y_shift | rotation<br>"
            "time_s | seed | new_seed_int<br>"
            "time_s | noise | added_noise_strength<br>"            
            "time_s | denoise | denoise_value<br>"
            "time_s | set_text | textblock_name | text_prompt | x | y | w | h | fore_color | back_color | font_name<br>"
            "time_s | clear_text | textblock_name<br>"
            "time_s | prop | prop_name | prop_filename | x pos | y pos | scale | rotation<br>"
            "time_s | set_stamp | stamp_name | stamp_filename | x pos | y pos | scale | rotation<br>"
            "time_s | clear_stamp | stamp_name<br>"
            "time_s | col_set<br>"
            "time_s | col_clear<br>"
            "time_s | model | " + ", ".join(
                sorted([x.model_name for x in sd_models.checkpoints_list.values()])) + "</p>")

        chkimg2img = gr.Checkbox(label="img2img_mode", value=is_img2img, visible=False)

        key_frames = gr.Textbox(label="Keyframes:", lines=5, value="")
        return [i1, i2, i3, i4, i5, i6, total_time, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos, tmpl_neg,
                key_frames, denoising_strength, x_shift, y_shift, rotation, propfolder, seed_march, smoothing,
                add_noise, noise_strength, chkimg2img]

    def run(self, p, i1, i2, i3, i4, i5, i6, total_time, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos,
            tmpl_neg, key_frames, denoising_strength, x_shift, y_shift, rotation, propfolder, seed_march, smoothing,
            add_noise, noise_strength, is_img2img):

        print(os.getcwd())

        # Fix variable types, i.e. text boxes giving strings.
        total_time = float(total_time)
        fps = float(fps)
        zoom_factor = float(zoom_factor)
        x_shift = float(x_shift)
        y_shift = float(y_shift)
        rotation = float(rotation)
        apply_colour_corrections = True

        frame_count = int(fps * total_time)

        # Theoretical frame rate, may lead to diff video length.
        final_fps = fps + fps * smoothing

        # Frame source
        source = 'img2img'
        source_path = ''
        source_cap = None

        output_filename = time.strftime('%Y%m%d%H%M%S')
        output_path = os.path.join(p.outpath_samples, output_filename)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        # Pandas key-framing, hopefully use this to interpolate a bunch of values.
        variables = {'pos1': np.nan,
                     'neg1': np.nan,
                     'pos2': np.nan,
                     'neg2': np.nan,
                     'prompt': np.nan,
                     'denoise': np.nan,
                     'noise': np.nan,
                     'x_shift': np.nan,
                     'y_shift': np.nan,
                     'zoom': np.nan,
                     'rotation': np.nan}

        df = pd.DataFrame(variables, index=range(frame_count + 1))
        # Preload the dataframe with initial values.
        df.loc[0, ['denoise', 'x_shift', 'y_shift', 'zoom', 'rotation', 'noise']] = [denoising_strength,
                                                                            x_shift / fps,
                                                                            y_shift / fps,
                                                                            zoom_factor ** (1.0 / fps),
                                                                            rotation / fps,
                                                                            noise_strength]

        keyframes = {}
        my_prompts = []
        my_seeds = []

        # Prime seed
        processing.fix_seed(p)

        for key_frame in key_frames.splitlines():
            key_frame_parts = key_frame.split("|")
            if len(key_frame_parts) < 2:
                continue
            tmp_frame_no = int(float(key_frame_parts[0]) * fps)
            tmp_command = key_frame_parts[1].lower().strip()

            if tmp_frame_no not in keyframes:
                keyframes[tmp_frame_no] = []
            keyframes[tmp_frame_no].append(key_frame_parts[1:])

            if tmp_command == "transform" and len(key_frame_parts) == 6 and is_img2img:
                # Time (s) | transform  | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Rotation (deg/s)
                df.loc[tmp_frame_no, ['x_shift', 'y_shift', 'zoom', 'rotation']] = [float(key_frame_parts[3]) / fps,
                                                                                    float(key_frame_parts[4]) / fps,
                                                                                    float(key_frame_parts[2]) ** (
                                                                                            1.0 / fps),
                                                                                    float(key_frame_parts[5]) / fps]
            elif tmp_command == "denoise" and len(key_frame_parts) == 3 and is_img2img:
                # Time (s) | denoise | denoise
                df.loc[tmp_frame_no, ['denoise']] = [float(key_frame_parts[2])]
            elif tmp_command == "noise" and len(key_frame_parts) == 3 and is_img2img:
                # Time (s) | noise | noise_strength
                df.loc[tmp_frame_no, ['noise']] = [float(key_frame_parts[2])]
            elif tmp_command == "seed" and len(key_frame_parts) == 3:
                # Time (s) | seed | seed
                my_seeds.append((tmp_frame_no, int(key_frame_parts[2])))
            elif tmp_command == "prompt" and len(key_frame_parts) == 4:
                # Time (s) | prompt | Positive Prompts | Negative Prompts
                my_prompts.append((tmp_frame_no, key_frame_parts[2].strip().strip(",").strip(),
                                   key_frame_parts[3].strip().strip(",").strip()))
            elif tmp_command == "template" and len(key_frame_parts) == 4:
                # Time (s) | template | Positive Prompts | Negative Prompts
                tmpl_pos = key_frame_parts[2].strip().strip(",").strip()
                tmpl_neg = key_frame_parts[3].strip().strip(",").strip()
            elif tmp_command == "prompt_from_png" and len(key_frame_parts) == 3:
                # Time (s) | prompt_from_png | file name
                foundinfo, geninfo, info = get_pnginfo(key_frame_parts[2].strip().strip(",").strip())
                if foundinfo:
                    if "\nNegative prompt:" in geninfo:
                        tmp_posprompt = geninfo[:geninfo.find("\nNegative prompt:")]
                        tmp_negprompt = geninfo[geninfo.find("\nNegative prompt:")+18:geninfo.rfind("\nSteps:")]
                    else:
                        tmp_posprompt = geninfo[:geninfo.find("\nSteps:")]
                        tmp_negprompt = ''
                    tmp_params = geninfo[geninfo.rfind("\nSteps:")+1:]
                    tmp_seed = int(tmp_params[tmp_params.find('Seed: ') + 6: tmp_params.find(",", tmp_params.find('Seed: ') + 6)])
                    my_prompts.append((tmp_frame_no, tmp_posprompt, tmp_negprompt))
                    my_seeds.append((tmp_frame_no, tmp_seed))
                    # print(f"Pos:[{tmp_posprompt}] Neg:[{tmp_negprompt}] Seed:[{tmp_seed}]")
            elif tmp_command == "source" and len(key_frame_parts) > 2 and is_img2img:
                # time_s | source | source_name | path
                tmp_source_name = key_frame_parts[2].lower().strip()
                tmp_source_path = key_frame_parts[3].lower().strip()
                if tmp_source_name == 'video':
                    if os.path.exists(tmp_source_path):
                        try:
                            source_cap = cv2.VideoCapture(tmp_source_path)
                            source = tmp_source_name
                            source_path = tmp_source_path
                        except Exception as ex:
                            print(f"Failed to load video: {ex}")
                    else:
                        print(f"Could not locate video: {tmp_source_path}")
                elif tmp_source_name == 'images':
                    source_cap = glob.glob(tmp_source_path)
                    if len(source_cap) > 0:
                        source = tmp_source_name
                        print(f'Found {len(source_cap)} images in {tmp_source_path}')
                    else:
                        print(f'No images found, reverting back to img2img: {tmp_source_path}')

        # Sort list of prompts, and then populate the dataframe in a alternating fashion.
        # need to do this to ensure the prompts flow onto each other correctly.
        my_prompts = sorted(my_prompts)

        # Special case if no prompts supplied.
        if len(my_prompts) == 0:
            df.loc[0, ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = ["", "", "", "", 1.0]
        for x in range(len(my_prompts) - 1):
            df.loc[my_prompts[x][0], ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = [my_prompts[x][1],
                                                                                    my_prompts[x][2],
                                                                                    my_prompts[x + 1][1],
                                                                                    my_prompts[x + 1][2],
                                                                                    1]
            if x > 0:
                df.loc[my_prompts[x][0] - 1, 'prompt'] = 0
        df.at[df.index[-1], 'prompt'] = 0

        if len(my_seeds) > 0:
            # Seed commands given.
            my_seeds = sorted(my_seeds)
            if seed_march or not is_img2img:
                # Try to interpolate from seed -> sub-seed, by increasing sub-seed strength
                for x in range(len(my_seeds) - 1):
                    df.loc[my_seeds[x][0], ['seed_start', 'seed_end', 'seed_str']] = [str(my_seeds[x][1]),
                                                                                      str(my_seeds[x + 1][1]),
                                                                                      0]
                    if x == len(my_seeds) - 2:
                        df.at[df.index[-1], 'seed_str'] = 1
                    if x > 0:
                        df.loc[my_seeds[x][0] - 1, 'seed_str'] = 1  # Ensure all values tend to one in the list
                df.loc[:, ['seed_start', 'seed_end']] = df.loc[:, ['seed_start', 'seed_end']].ffill()
            else:
                # Just interpolate from one seed value to the next. experimental. Set sub-seed to None to disable.
                for x in range(len(my_seeds)):
                    df.at[df.index[my_seeds[x][0]], 'seed_start'] = my_seeds[x][1]
                df['seed_end'] = None
                df['seed_str'] = 0
        else:
            # No seeds given, load in initial value, series fill. Set sub-seed to None to disable.
            df.at[df.index[0], 'seed_start'] = int(p.seed)
            df.at[df.index[-1], 'seed_start'] = int(p.seed) + frame_count
            df['seed_end'] = None
            df['seed_str'] = 0

        # Interpolate columns individually depending on how many data points.
        for name, values in df.items():
            if name in ['prompt', 'seed_str']:
                df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')
            elif values.count() > 3:
                df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both', method="polynomial", order=2)
                df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')  # catch last null values.
            else:
                df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')

        # df = df.interpolate(limit_direction='both')
        df.loc[:, ['pos1', 'neg1', 'pos2', 'neg2']] = df.loc[:, ['pos1', 'neg1', 'pos2', 'neg2']].ffill()
        # print(df)

        # Check if templates are filled in. If not, try grab prompts at top (i.e. image sent from png info)
        if len(tmpl_pos.strip()) == 0:
            tmpl_pos = p.prompt if len(p.prompt.strip()) > 0 else ''
        if len(tmpl_neg.strip()) == 0:
            tmpl_neg = p.negative_prompt if len(p.negative_prompt.strip()) > 0 else ''

        df['pos_prompt'] = str(tmpl_pos) + ", " + df['pos1'].map(str) + ":" + df['prompt'].map(str) + ' AND ' + \
                            str(tmpl_pos) + ', ' + df['pos2'].map(str) + ":" + (1.0 - df['prompt']).map(str)
        df['neg_prompt'] = str(tmpl_neg) + ", " + df['neg1'].map(str) + ":" + df['prompt'].map(str) + ' AND ' + \
                            str(tmpl_neg) + ', ' + df['neg2'].map(str) + ":" + (1.0 - df['prompt']).map(str)

        csv_filename = os.path.join(output_path, f"{str(output_filename)}_frames.csv")
        df.to_csv(csv_filename)

        # Clean up prompt templates
        tmpl_pos = str(tmpl_pos).strip()
        tmpl_neg = str(tmpl_neg).strip()

        # Save extra parameters for the UI
        p.extra_generation_params = {
            "Create GIF": vid_gif,
            "Create MP4": vid_mp4,
            "Create WEBM": vid_webm,
            "Total Time (s)": total_time,
            "FPS": fps,
            "Seed March": seed_march,
            "Smoothing Frames": smoothing,
            "Initial De-noising Strength": denoising_strength,
            "Initial Zoom Factor": zoom_factor,
            "Initial X Pixel Shift": x_shift,
            "Initial Y Pixel Shift": y_shift,
            "Rotation": rotation,
            "Prop Folder": propfolder,
            "Prompt Template Positive": tmpl_pos,
            "Prompt Template Negative": tmpl_neg,
            "Keyframe Data": key_frames,
        }

        # save settings, just dump out the extra_generation dict
        settings_filename = os.path.join(output_path, f"{str(output_filename)}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            json.dump(dict(p.extra_generation_params), f, ensure_ascii=False, indent=4)

        # Check prompts. If no prompt given, but templates exist, set them.
        if len(p.prompt.strip(",").strip()) == 0:
            p.prompt = tmpl_pos
        if len(p.negative_prompt.strip(",").strip()) == 0:
            p.negative_prompt = tmpl_neg

        # Post Processing object dicts
        text_blocks = {}
        props = {}
        stamps = {}

        p.batch_size = 1
        p.n_iter = 1

        # output_images, info = None, None
        initial_seed = None
        initial_info = None

        # grids = []
        all_images = []
        last_keyframe_image = None

        # Make bat files before we start rendering video, so we could run them manually to preview output.
        make_gif(output_path, output_filename, final_fps, False, True)
        make_mp4(output_path, output_filename, final_fps, False, True)
        make_webm(output_path, output_filename, final_fps, False, True)

        state.job_count = frame_count

        if is_img2img:
            initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        x_shift_cumulative = 0
        y_shift_cumulative = 0

        last_frame = None
        frame_save = 0

        # Iterate through range of frames
        for frame_no in range(frame_count):

            if state.interrupted:
                # Interrupt button pressed in WebUI
                break

            # Check if keyframes exists for this frame
            # print("process keyframes")
            if frame_no in keyframes:
                # Keyframes exist for this frame.
                print(f"\r\nKeyframe at {frame_no}: {keyframes[frame_no]}\r\n")

                for keyframe in keyframes[frame_no]:
                    keyframe_command = keyframe[0].lower().strip()
                    # Check the command, should be first item.
                    if keyframe_command == "seed" and len(keyframe) == 3:
                        # Time (s) | seed | seed
                        p.seed = int(keyframe[1])
                        processing.fix_seed(p)
                    elif keyframe_command == "subseed" and len(keyframe) == 3:
                        # Time (s) | subseed | subseed
                        p.subseed = int(keyframe[1])
                        processing.fix_seed(p)

                    elif keyframe_command == "model" and len(keyframe) == 2:
                        # Time (s) | model    | model name
                        info = sd_models.get_closet_checkpoint_match(keyframe[1].strip() + ".ckpt")
                        if info is None:
                            raise RuntimeError(f"Unknown checkpoint: {keyframe[1]}")
                        sd_models.reload_model_weights(shared.sd_model, info)

                    elif keyframe_command == "col_set" and len(keyframe) == 1 and is_img2img:
                        # Time (s) | col_set
                        apply_colour_corrections = True
                        if frame_no > 0:
                            # Colour correction is set automatically above
                            initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
                    elif keyframe_command == "col_clear" and len(keyframe) == 1 and is_img2img:
                        # Time (s) | col_clear
                        apply_colour_corrections = False

                    elif keyframe_command == "prop" and len(keyframe) == 6 and is_img2img:
                        # Time (s) | prop | prop_filename | x pos | y pos | scale | rotation
                        # bit of a hack, no prop name is supplied, but same function is used to draw.
                        # so the command is passed in place of prop name, which will be ignored anyway.
                        props[len(props)] = keyframe
                    elif keyframe_command == "set_stamp" and len(keyframe) == 7:
                        # Time (s) | set_stamp | stamp_name | stamp_filename | x pos | y pos | scale | rotation
                        stamps[keyframe[1].strip()] = keyframe[1:]
                    elif keyframe_command == "clear_stamp" and len(keyframe) == 2:
                        # Time (s) | clear_stamp | stamp_name
                        if keyframe[1].strip() in stamps:
                            stamps.pop(keyframe[1].strip())

                    elif keyframe_command == "set_text" and len(keyframe) == 10:
                        # time_s | set_text | name | text_prompt | x | y | w | h | fore_color | back_color | font_name
                        text_blocks[keyframe[1].strip()] = keyframe[1:]
                    elif keyframe_command == "clear_text" and len(keyframe) == 2:
                        # Time (s) | clear_text | textblock_name
                        if keyframe[1].strip() in text_blocks:
                            text_blocks.pop(keyframe[1].strip())

            # print("set processing options")
            p.prompt = str(df.loc[frame_no, ['pos_prompt']][0])
            # print(p.prompt)
            p.negative_prompt = str(df.loc[frame_no, ['neg_prompt']][0])
            # print(p.negative_prompt)

            p.seed = int(df.loc[frame_no, ['seed_start']][0])
            p.subseed = None \
                if df.loc[frame_no, ['seed_end']][0] is None else int(df.loc[frame_no, ['seed_end']][0])
            p.subseed_strength = None \
                if df.loc[frame_no, ['seed_str']][0] is None else float(df.loc[frame_no, ['seed_str']][0])
            # print(f"Frame:{frame_no} Seed:{p.seed} Sub:{p.subseed} Str:{p.subseed_strength}")

            p.denoising_strength = df.loc[frame_no, ['denoise']][0]

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            init_img = None
            #
            # Get source frame
            #
            # print("get source frame")
            if source == 'img2img' and is_img2img:
                # Extra processing parameters
                if apply_colour_corrections:
                    p.color_corrections = initial_color_corrections

                # TODO: Make this seed marching a diff img source
                if seed_march:
                    # Feed back last seed image
                    if frame_no == 0:
                        last_keyframe_image = p.init_images[0]
                    elif p.subseed_strength == 0.0:
                        last_keyframe_image = processed.images[0]
                    init_img = last_keyframe_image.copy()
                else:
                    # Feed back image
                    if frame_no == 0:
                        init_img = p.init_images[0]
                    else:
                        init_img = processed.images[0]

            elif source == 'video' and is_img2img:
                source_cap.set(1, frame_no)
                ret, tmp_array = source_cap.read()
                init_img = Image.fromarray(cv2.cvtColor(tmp_array, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')

            elif source == 'images' and is_img2img:
                if frame_no >= len(source_cap):
                    init_img = Image.open(source_cap[-1])
                    print('Out of frames, reverting to last frame!')
                else:
                    init_img = Image.open(source_cap[frame_no])
                if init_img.mode != 'RGB':
                    init_img = init_img.convert('RGB')

            #
            # Pre-process source frame
            #
            # print("pre process frame")
            if init_img is not None:
                # Update transform details
                x_shift_per_frame = df.loc[frame_no, ['x_shift']][0]
                y_shift_per_frame = df.loc[frame_no, ['y_shift']][0]
                rot_per_frame = df.loc[frame_no, ['rotation']][0]
                zoom_factor = df.loc[frame_no, ['zoom']][0]

                # Translate source frame when source is img2img where they have an effect frame to frame.
                x_shift_cumulative = x_shift_cumulative + x_shift_per_frame
                y_shift_cumulative = y_shift_cumulative + y_shift_per_frame

                init_img = zoom_at2(init_img, rot_per_frame, int(x_shift_cumulative), int(y_shift_cumulative),
                                    zoom_factor)

                # Subtract the integer portion we just shifted.
                x_shift_cumulative = x_shift_cumulative - int(x_shift_cumulative)
                y_shift_cumulative = y_shift_cumulative - int(y_shift_cumulative)

                # Props
                if len(props) > 0:
                    init_img = pasteprop(init_img, props, propfolder)
                    props = {}

                # Noise

                if add_noise and is_img2img:
                    # print("Adding Noise!!")
                    init_img = addnoise(init_img, df.loc[frame_no, ['noise']][0])

                # print("processing frame now.")
                state.job = f"Major frame {frame_no} of {frame_count}"
                p.init_images = [init_img]

            # Debug, print out source frame
            #init_img.save(os.path.join(output_path, f"{output_filename}_{frame_save:05}_initial.png"))

            #
            # Process source frame into destination frame
            #

            processed = processing.process_images(p)

            #
            # Post-process destination frame
            #
            # print("post process")
            post_processed_image = processed.images[0].copy()
            if len(stamps) > 0:
                post_processed_image = pasteprop(post_processed_image, stamps, propfolder)
            if len(text_blocks) > 0:
                post_processed_image = rendertext(post_processed_image, text_blocks)

            #
            # Save frame
            #
            # Save every seconds worth of frames to the output set displayed in UI
            # print("save frame")
            if seed_march:
                if frame_no == 0 or p.subseed_strength == 0 or frame_no == frame_count:
                    all_images.append(post_processed_image)
            elif frame_no % int(fps) == 0:
                all_images.append(post_processed_image)

            # Create and save smoothed intermediate frames
            if frame_no > 0 and smoothing > 0:
                # working a frame behind, smooth from last_frame -> post_processed_image
                for idx, img in enumerate(morph(last_frame, post_processed_image, smoothing)):
                    img.save(os.path.join(output_path, f"{output_filename}_{frame_save:05}.png"))
                    print(f"{frame_save:03}: {frame_no:03} > {idx} smooth frame")
                    frame_save += 1

            # save main frames
            post_processed_image.save(os.path.join(output_path, f"{output_filename}_{frame_save:05}.png"))
            # print(f"{frame_save:03}: {frame_no:03} frame")
            frame_save += 1

            last_frame = post_processed_image.copy()

            # I guess this is important, don't really know.
            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info
            # print("end of loop")

        # If not interrupted, make requested movies. Otherwise, the bat files exist.
        make_gif(output_path, output_filename, final_fps, vid_gif & (not state.interrupted), False)
        make_mp4(output_path, output_filename, final_fps, vid_mp4 & (not state.interrupted), False)
        make_webm(output_path, output_filename, final_fps, vid_webm & (not state.interrupted), False)

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
