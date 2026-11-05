import os
import subprocess
import shutil

def has_nvidia_nvenc() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

def render_chapter_to_mp4(chapter: dict, preset: dict, project_dir: str, use_nvenc: bool = None) -> str:
    chap_num = chapter["chapter_number"]
    chap_out = os.path.join(project_dir, f"chapter_{chap_num:02d}.mp4")
    if os.path.exists(chap_out):
        return chap_out

    if use_nvenc is None:
        use_nvenc = has_nvidia_nvenc()

    fps = 24
    z_speed = preset["render"]["zoom_speed"]
    max_z = preset["render"]["max_zoom"]
    post_filter = preset["render"]["ffmpeg_post_filter"]
    v_codec = "h264_nvenc" if use_nvenc else "libx264"

    scene_clips = []
    for scene in chapter["scenes"]:
        sid = scene["scene_id"]
        img_path = os.path.join(project_dir, f"scene_{sid:04d}.png")
        wav_path = os.path.join(project_dir, f"scene_{sid:04d}.wav")
        clip_path = os.path.join(project_dir, f"temp_scene_{sid:04d}.mp4")
        duration = max(1.0, float(scene.get("audio_duration", 5.0)))
        frames = int(duration * fps) + 4

        if scene.get("scene_type") in ["title_card", "split_comparison"] and not preset["channel_style"].startswith("mr_nightmare"):
            vf = f"scale=1920x1080,fps={fps},format=yuv420p"
        else:
            motion = scene.get("camera_motion", "zoom_in")
            if motion == "pan_right":
                zp = f"z='1.12':x='(iw-iw/zoom)*(on/{frames})':y='ih/2-(ih/zoom/2)'"
            elif motion == "pan_left":
                zp = f"z='1.12':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)'"
            elif motion == "zoom_out":
                zp = f"z='if(eq(on,1),{max_z},max(zoom-{z_speed},1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            else:
                zp = f"z='min(zoom+{z_speed},{max_z})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            vf = f"scale=3840x2160,zoompan={zp}:d={frames}:s=1920x1080:fps={fps},{post_filter},format=yuv420p"

        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-i", wav_path,
            "-vf", vf, "-c:v", v_codec,
            "-preset", "p2" if use_nvenc else "veryfast",
            "-t", f"{duration:.2f}", "-c:a", "aac", "-b:a", "192k", "-shortest", clip_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        scene_clips.append(clip_path)

    list_file = os.path.join(project_dir, f"chap_{chap_num:02d}_list.txt")
    with open(list_file, "w") as lf:
        for cp in scene_clips:
            lf.write(f"file '{os.path.abspath(cp)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", chap_out
    ], check=True)

    for cp in scene_clips:
        if os.path.exists(cp):
            os.remove(cp)

    return chap_out
