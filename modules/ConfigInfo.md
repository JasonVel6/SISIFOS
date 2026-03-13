# Config Information


This is a document outlining the different fields present in a config passed to the main SISIFOS renderer (```main.py```) or the trajectory generator (```modules/trajectory/generateTrajectoriesUnified.py```).

Configs are should be kept in the /configs folder for organizaiton. A few example configs are provided:
* ```example_path.json``` an example of passing a trajectory path to SISIFOS (Note: the trajectory passed is not provided in the repo so this will not run unless a valid path is provided)
* ```example_trajectory_generation.json``` Will generate an example trajectory and then render it.
* ```example_trajectory_generation_sweep_test.json``` Will generate 2 trajectories and render both using the ```"sweep_parameters"``` field.
* ```example-sampling.json``` Will generate camera positions by sampling a sphere around the target.

## Background
Config JSON files let us easily specify all the customizable parameters needed for a run of SISIFOS. They also allow us to sweep multiple values of parameters easily to generate multiple datasets with slight differences. The config generation is managed by ```modules/config.py``` which is primarally built using the pydantic python module. More information on pydantic can be found here: https://docs.pydantic.dev/latest/

### Running main.py
main.py should be run with the following command:
```
blender -b -P main.py -- --sweep_config_path <path to your config>
```
The 2 options for passing configs are:
```
--sweep_config_path
--config_path
```
```--config_path``` allows the user to specify a path for a single run config and ```--sweep_config_path``` allows the user to specify the path for a sweep of runs. More information below on the difference between the 2 configs.

## Sweep Config Fields
A sweep config is made up of 2 fields ```"base_config"``` and ```"sweep_parameters"```
### Base Config
Base config provides all the fields necessary for a single run and can be used with the ```--config_path``` argument. It is a nested dictionary.

### Sweep Parameters
Sweep parameters is a dictionary that can be used to set sweeps of any parameter in the ```"base_config"``` section. It is a dictionary that maps the path of the parameter (separated by periods) to a list of its sweep values.
#### An example of sweeping 2 trajectory path modes:
```
"sweep_parameters": {"trajectory.path_mode": ["tumbling", "inertial"]}
```
#### An example of sweeping spacecraft models:
```
"sweep_parameters": {"selected_model": ["RF_Integral", "RF_Hubble"]}
```
#### These can be combined:
```
"sweep_parameters": {
    "selected_model": ["RF_Integral", "RF_Hubble"],
    "trajectory.path_mode": ["tumbling", "inertial"]
}
```
When these parameter sweeps are combined the combinations are generated such that every combination is tried in a grid fashion. The example above would generate 4 different runs. Note: adding a lot of sweeps can increase the number of renders generated exponentially so be careful with how many sweeps you choose.

## Base Config parameters
#### scene_blend_path:
String containing the path to the downloaded scene_blend
```"assets/scene.blend"``` is a good default

#### hdri_path:
string containing the path to the starmap (high resolution image)
```"assets/starmap_2020_16k.exr"``` is good for lower quality renders
```"assets/starmap_2020_64k.exr"``` is good for higher quality renders

#### objects:
Dictionary mapping string object names to their ```ObjectConfig```

#### camera
a nested ```CameraConfig```

#### render:
a nested ```RenderConfig```

#### setup:
a nested ```SetupConfig```

#### trajectory_type:
a string literal describing the type of trajectory to use
Options:
* ```"trajectory_generator"``` will use the ```generateTrajectoriesUnified.py``` script to generate a physically accurate trajectory (uses the ```"trajectory"``` field of the config)
* ```"sampling_trajectory"``` will sample views around the target in a sphere (uses the ```"trajectory_sampling"``` field of the config)
* ```"filepath"``` will look for a trajectory file at ```"trajectory_filepath"```

#### trajectory_sampling:
a nested config of type ```SamplingTrajectoryConfig``` to be used when ```"sampling_trajectory"``` is selected

#### trajectory:
a nested config of type ```TrajectoryConfig``` to be used when the ```"trajectory_generator"``` option is selected

#### trajectory_filepath:
string containing path to the reference trajectory file to be used when the ```"filepath"``` option is selected

#### save_depth:
bool option to save depth images

#### save_normals:
bool option to save normal images

#### save_optical_flow:
bool option to save the optical flow

#### save_segmentation:
bool option to save the segmentation

#### save_obj_poses
bool option to save object poses

#### save_scene_plots
bool option to save trajectory scene graphs

#### scene_plot_max_frames
int or null controlling the maximum number of scene-plot frames generated when `save_scene_plots` is true.
Set to `null` to remove the cap.

#### frame_ids
optional list of explicit frame indices to render.
If this is provided, SISIFOS renders exactly these frames and ignores `from_frame_id`.

#### from_frame_id
optional int for the first frame to render when `frame_ids` is `null`.
This value is inclusive, so `from_frame_id: 10` renders frames 10 through the end of the trajectory.
Default behavior when both `frame_ids` and `from_frame_id` are `null` is to render all frames starting at 0.

#### selected_model
string to determine what model to render
Default: ```"RF_Hubble"```

#### model_rotation_z_deg
float initial z rotation

### ObjectConfig
Basic config to describe objects that are loaded in
#### name
string containing the name of the object to describe
#### blend_path
(optional) string containing the path to the .blend file for this object
Is optional because some objects are contained in the scene or other blend files
#### scale_factor:
a float that describes how much the object should be scaled by
used for scaling the earth and other assets so that their size appears correct given their distances
Default is 1.0 (no scaling)

### Camera Config
#### focal_length:
float describing the focal length of the camera in milimeters (mm)
#### sensor_width:
float containing the sensor width of the camera in milimeters (mm)
#### clip_start:
float to define the near boundary for rendering in blender in meters (m)
#### clip_end:
float to define the far boundary for rendering in blender in meters (m)
#### resolution:
tuple (int, int) containing the resolution in pixels of the camera
#### lens_flare:
float representing the lens flare of the camera
#### exposure_time_s:
float representing the exposure time of the camera in seconds (s)
#### focal_length_px:
property meaning it is not in the config json but can be calculated from the pydantic object
Gives the focal length in pixels instead of mm

### RenderConfig
Basic info for the blender render module
#### engine:
string representing the engine to use in blender
#### samples:
int representing the number of samples blender takes
4 for fas render
32 for high quality render
#### earth_dist_scale_factor:
float to scale the distance of the earth for better rendering
we currently use 1/1000

### SetupConfig
Basic setup for the environment
#### earth_mode:
string literal: on or off
#### stars_mode
string literal: on or off
#### enable_blur
string literal: on or off
#### enable_glare
string literal: on or off
#### t_ref_s
float for rendering motion blur and shutter speed
#### blur_shutter_factor
float for rendering motion blur and shutter speed
#### blur_motion_factor
float for rendering motion blur and shutter speed
#### glare_threshold:
float for calculating glare
#### glare_size:
int for calculating glare

### SamplingTrajectoryConfig
Basic config for sampling views around the target
#### num_frame
int representing the number of views to generate
#### R_RPO
float representing distance from camera to target
#### R_LEO
float representing altitude of LEO orbit
#### sun_az
float representing azimuth angle of the the sun in degrees
#### sun_el
float representing the elevation angle of the sun in degrees

### TrajectoryConfig
A config used to define physical trajectories in ```generateTrajectoriesUnified.py```
#### path_mode:
different trajectory types
options: "inertial", "hill", and "tumbling"
See reldyn methodology paper for more info
#### seed:
optional int to allow for deterministic rollouts for reproducibility
if not provided a pseudorandom seed is autogenerated
#### num_agents:
int to decide the number of agents to generate
#### num_MC:
int for the number of monte carlo trials to run
#### r_AG_G:
list of 3 floats representing the vector from G frame to A frame
#### sigma_Rxy_aps3 and sigma_Rz_aps3:
float representing the noise of star tracker
Default:
    ASTRO APS3 star tracker (Jena-Optronik)
    https://www.jena-optronik.de/products/star-sensors/astro-aps3.html
    Datasheet does not provide axis breakdown; we model
    lower cross-boresight noise and conservative roll degradation.
#### sigma_omega and sigma_accel:
Sensor noise of the IMU
Default:
    IMU Gyroscope ARW noise (Astrix NS IMU, Exail Astrix Series)
    https://www.exail.com/product-range/astrix-series
    Converted to rad/s and scaled for 10 Hz measurements:
#### MEAN_DEFAULT:
default mean of the accel noise
#### GYRO_BIAS_SIGMA_DEGPHR:
float for gyro bias
#### GYRO_BIAS_TAU_S:
float for gyro bias
#### ACCEL_BIAS_SIGMA_UG:
float for accel bias
#### ACCEL_BIAS_TAU_S:
float for accel bias
#### R0_const:
nominal midpoint standoff from target
#### SUN_ALIGN_ENABLE:
bool if we align with the sun 
#### SUN_ALIGN_CONE_DEG:
float in degreees of the cone to align with the sun
#### SUN_ALIGN_JITTER_D:
float in degrees of jitter of sun alignment
#### EARTH_BACKGROUND_ENABLE:
enable earth in background
#### mu_ref:
float for mu constant of earth (m^3/s^2)
default: 3.986004418e14 
#### h_orbit:
float for height of orbit in meters
#### R_earth:
float radius of earth meters
#### IMAGE_MAX_DT_S:
float largest timestep between images seconds
#### tend:
float end time seconds
#### tstep:
timestep in seconds
#### MIN_F2F_PX_MED:
minimum frame to frame pixel movement for analysis
#### inertia of target parameters (box model):
all floats for calculation of the satelite inertia matrix
m: mass (kg)
l: length (m)
w: width (m)
h: (m)
#### a_ref:
calculated field not included in config
#### n_scalar:
calculated field not included in config
#### J:
calculated field for the inertia matrix
#### COV_R_ASTRO_APS3:
calculated cov matrix for star tracker
#### COV_OMEGA_ASTRIX:
calculated cov matrix for star tracker
#### COV_ACCEL_ASTRIX:
calculated cov matrix for star tracker
#### rotMode_Gframe:
calculated field for int representation of rotation mode
