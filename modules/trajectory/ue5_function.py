import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import sys
import os
import json
import subprocess
import shutil
from scipy.linalg import logm
from math_utils import *

def axes_to_quaternion(x_axis, y_axis, z_axis):
    # Form the rotation matrix from the axes
    R = np.column_stack((x_axis, y_axis, z_axis))
    # Convert to quaternion
    quaternion =R2q(R)
    return quaternion

def cartesian_to_spherical(x, y, z):
    # Calculate azimuth angle (phi)
    azimuth = np.arctan2(y, x)
    
    # Calculate elevation angle (theta)
    elevation = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    
    return azimuth, elevation

def read_gt_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find section headers to locate the data
    sections = {}
    current_section = None
    for i, line in enumerate(lines):
        line = line.strip()
        if '=' in line:
            current_section = line.split('=')[0].strip()  # Get the part before '='
            sections[current_section] = i + 1  # Start of data is next line
            
    # Now extract data based on section positions
    nSamples = int(lines[sections['nSamples']].strip())
    
     # helper: return None if the section isn't in the file
    def read_block(name, width):
        if name not in sections:
            return None
        start = sections[name]
        return np.array(
            [list(map(float, lines[start+i].split())) for i in range(nSamples)]
        ).reshape(nSamples, width)
    
    timestamps = read_block("timestamps", 1).ravel()
    q_GS   = read_block("q_GS",   4)
    q_IG   = read_block("q_IG",   4)
    q_IS   = read_block("q_IS",   4)
    r_SG_G = read_block("r_SG_G", 3)
    r_OG_G = read_block("r_OG_G", 3)

    # Extract sun_az_el if it exists
    if 'sun_az_el' in sections:
        sun_az_el = read_block("sun_az_el", 2)
    else:
        sun_az = read_block("sun_az", 1)
        sun_el = read_block("sun_el", 1)
        sun_az_el = np.hstack([sun_el, sun_az]) if sun_az is not None else None

    data_dict = {
        "nSamples": nSamples,
        "timestamps": timestamps,
        "q_GS": q_GS,
        "r_SG_G": r_SG_G,
        "q_IG": q_IG,
        "q_IS": q_IS,
        "r_OG_G": r_OG_G,
        "sun_az_el": sun_az_el
    }
    return data_dict

def _to_native(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    return obj

def create_json(camera, data_dict, tstep, tend, foutput, earth=False, stars=False):
    # Format earth location data
    earth_loc_data = [{"data": e_loc} for e_loc in data_dict['r_OG_G']]
    
    # Format sun rotation data 
    # IMPORTANT: Note the order - first element is elevation, second is azimuth
    sun_rot_data = []
    for s_rot in data_dict['sun_az_el']:
        # s_rot[0] is elevation, s_rot[1] is azimuth in the gtValues file
        sun_rot_data.append({"data": [[s_rot[0]], [s_rot[1]]]})
    
    # Compute vertical FOV in radians & aspect ratio
    yfov_rad = 2 * np.arctan((camera['resolution'] / 2) / camera['focal_length'])
    aspect   = float(camera['resolution']) / float(camera['resolution'])

    data = {
        "post_process_params": {
            "cam_shutter_speed": 200.0,
            "cam_iso": 100.0,
            "cam_aperture": 4.0,
            "exposure_comp": 0.01,
            "bloom_intensity": 0.1,
            "chromatic_aberration_intensity": 0.0,
            "chromatic_aberration_start_offset": 0.0,
            "vignette_intensity": 0.1,
            "lens_flare_intensity": 0.1,
            "film_grain_intensity": 0.1
        },
        "custom_model_params": {
            # Scale should be set to 1.0 for real-dimensioned models
             "scale": 1.0,
             "custom_model_path": "C:\\Users\\jdflo\\Documents\\UE5-SpaceImageSimulator\\models\\hst.fbx"
            # "scale": 0.00028,
            # "custom_model_path": "C:\\Users\\jdflo\\Documents\\UE5-SpaceImageSimulator\\models\\integral.fbx"
        },
        "timing_params": {
          "tstep": tstep,
          "tend": tend
        },
        "camera_params": { # Unused in simulator, but used to create config.yaml
            "focal_length": camera['focal_length'],
            "fov": np.rad2deg(2*np.arctan((camera['resolution']/2)/camera['focal_length'])),
            "x_resolution": camera['resolution'],
            "y_resolution": camera['resolution']
        },
        "cameras": [
            {
                "type":        "perspective",
                "perspective": {
                    "aspectRatio": aspect,
                    "yfov":        float(yfov_rad)
                }
            }
        ],
        "environment_params": {
            "light_brightness": 128000.0,
            "star_vis": False if stars == "false" or not stars else True,
            "earth_vis": False if earth == "false" or not earth else True,
            "earth_loc": earth_loc_data,
            "sun_rot": sun_rot_data
        },
        "nSamples": data_dict['nSamples'],
        "timestamps": data_dict['timestamps'].tolist(),
        "q_GS": [{"data": quat} for quat in data_dict['q_GS']],
        "r_SG_G": [{"data": r} for r in data_dict['r_SG_G']]
    }
    
    foutput = f'{foutput}'
    with open(foutput, 'w') as json_file:
        data = _to_native(data)
        json.dump(data, json_file, indent=4)


def rename_imgs_in_folder(folder):

    # Iterate over files in the directory
    for filename in os.listdir(folder):
        # Check if the file starts with 'simulator_image_' and ends with '.png'
        if filename.endswith(".png"):
            # Extract the number from the original filename
            number = filename.split('_')[-1].split('.')[0]
            # Create the new filename with leading zeros
            new_filename = f"img_{int(number):05d}.png"

            # Get full paths
            old_file = os.path.join(folder, filename)
            new_file = os.path.join(folder, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)

def create_ffmpeg(gt_output, filename):
    parent_directory = os.path.dirname(gt_output)
    output_video = os.path.join(parent_directory, filename + '.mp4')
    print(f"Creating video {output_video} from images in {gt_output}")
    
    ffmpeg_command = [
        # ffmpeg
        r'C:\ffmpeg\bin\ffmpeg.exe',
        '-framerate', str(30),
        '-i', os.path.join(gt_output, "img_%05d.png"),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    my_env = os.environ.copy()

    try:
        # Set working directory to ensure correct relative paths
        result = subprocess.run(
            ffmpeg_command,
            check=True,
            env=my_env,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        print(f"Video created successfully: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"ffmpeg stderr: {e.stderr}")
        return False

