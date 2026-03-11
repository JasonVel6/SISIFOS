"""
config.py

Defines the configuration schema for the trajectory generator and renderer

"""

import time
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Dict, Optional, Tuple, Any, Literal
import json
import numpy as np
import itertools
import copy
import enum
import yaml

spacecraft_default_filepath = "modules/spacecraft_defaults.yaml"
with open(spacecraft_default_filepath, 'r') as f:
    spacecraft_defaults = yaml.safe_load(f)

class ObjectConfig(BaseModel):
    name: str
    blend_path: Optional[str] = None
    scale_factor: float = 1.0


class CameraConfig(BaseModel):
    """Camera properties"""
    # TODO auto-compute from R0_const, target size, and desired fill fraction
    focal_length: float = 400.0  # mm (matches Blender default)
    sensor_width: float = 36.0  # mm (matches Blender default)
    clip_start: float = 0.00001 #m
    clip_end: float = 5000000.0 #m
    resolution: Tuple[int, int] = (480, 480)
    lens_flare: float = 0.01
    exposure_time_s: float = 1.0/60.0 #s

    @property
    def focal_length_px(self) -> float:
        """Pixel focal length derived from mm focal length, sensor width, and resolution."""
        return self.focal_length / self.sensor_width * self.resolution[0]


class RenderConfig(BaseModel):
    """Render settings"""
    engine: str = "CYCLES"
    samples: int = 32
    # We scale the earth and bring it closer to the camera to help rendering
    earth_dist_scale_factor: float = 0.001

class SetupConfig(BaseModel):
    """Environment Setup"""
    # Reference settings
    earth_mode: Literal["on", "off"] = "on"
    stars_mode: Literal["on", "off"] = "on"
    enable_blur: Literal["on", "off"] = "off"
    enable_glare: Literal["on", "off"] = "off"
    t_ref_s: float = 0.01666667 # TODO why is this deffined here and in the camera config
    blur_shutter_factor: float =  0.8
    blur_motion_factor: float =  0.8
    glare_threshold: float = 0.95
    glare_size: int = 6
    generate_video: bool = True
    video_fps: int = 10

class SamplingTrajectoryConfig(BaseModel):
    """Configuration for the sampling-based trajectory generator"""
    num_frames: int = 200
    R_RPO: float = 36.0
    R_LEO: float = 8000000.0
    sun_az: float = 0.0
    sun_el: float = 0.0

class ConstantRotationConfig(BaseModel):
    """Configuration for the constant-rotation-based trajectory generator"""
    R_RPO: float = 36.0
    R_LEO: float = 8000000.0
    sun_az: float = 0.0
    sun_el: float = 0.0
    tstep: float = 0.1
    tend: float = 10.0
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 1.0)

class InertiaConfig(BaseModel):
    """
    Defines the inertia for physics propagation
    Triaxial inertia avoids axisymmetric observability degeneracy
    X - Highest inertial axis
    Y - Intermediate inertial axis
    Z - Lowest inertial axis
    """
    inertia_type: Literal["box", "cylinder", "custom"]
    m: float = 4000.0 #kg
    # following spatial dimensions are in meters
    l: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None
    r:Optional[float] = None

    Jx: Optional[float] = None
    Jy: Optional[float] = None
    Jz: Optional[float] = None

    @property
    def J(self) -> np.ndarray:
        if self.inertia_type == "box":
            if self.l is None or self.w is None or self.h is None:
                raise ValueError("Box inertia requires l, w, and h to be defined")
            J_xx = (1.0/12.0) * self.m * (self.w**2 + self.h**2)  # rotation about x (longest)
            J_yy = (1.0/12.0) * self.m * (self.l**2 + self.h**2)  # rotation about y
            J_zz = (1.0/12.0) * self.m * (self.l**2 + self.w**2)  # rotation about z
            return np.array([[J_xx, 0, 0], [0, J_yy, 0], [0, 0, J_zz]])
        elif self.inertia_type == "cylinder":
            # From https://scienceworld.wolfram.com/physics/MomentofInertiaCylinder.html
            # Simplified cylinder inertia (assuming rotation about central axis)
            if self.r is None or self.h is None:
                raise ValueError("Cylinder inertia requires r and h to be defined")
            J_xx = (1.0/12.0) * self.m * self.h **2 + (1/4) * self.m * self.r**2  # rotation about x (longitudinal)
            J_yy = (1.0/12.0) * self.m * self.h **2 + (1/4) * self.m * self.r**2  # rotation about y
            J_zz = (1.0/2.0) * self.m * self.r **2  # rotation about z (central axis)
            return np.array([[J_xx, 0, 0], [0, J_yy, 0], [0, 0, J_zz]])
        elif self.inertia_type == "custom":
            if self.Jx is None or self.Jy is None or self.Jz is None:
                raise ValueError("Custom inertia requires Jx, Jy, and Jz to be defined")
            return np.array([[self.Jx, 0, 0], [0, self.Jy, 0], [0, 0, self.Jz]])
        else:
            raise ValueError(f"Invalid or not implemented inertia_type: {self.inertia_type}")

def default_inertia_config(data: dict) -> InertiaConfig:
    if not data.get("selected_model"):
        raise ValueError("model_name must be specified if inertia_config is not provided")
    
    if data["selected_model"] not in spacecraft_defaults:
        raise ValueError(f"Couldn't find default for model_name: {data['selected_model']} in spacecraft_defaults.yaml")
    
    model_defaults = spacecraft_defaults[data["selected_model"]]
    # Default to integral inertia if not specified
    default_inertia = model_defaults["inertia"]
    return InertiaConfig(**default_inertia)

class TrajectoryConfig(BaseModel):
    """Trajectory generation settings"""
    selected_model: str
    # Commonly changed parameters
    path_mode: Literal["inertial", "hill", "tumbling"] = "tumbling"
    seed: Optional[int] = None # For reproducibility
    num_agents: int = 1
    num_mc: int = 1

    r_AG_G: List[float] = Field(default_factory=lambda: [0.1, 0.05, 0.15])
    
    # Sensor noise
    # ASTRO APS3 star tracker (Jena-Optronik)
    # https://www.jena-optronik.de/products/star-sensors/astro-aps3.html
    # Datasheet does not provide axis breakdown; we model
    # lower cross-boresight noise and conservative roll degradation.
    sigma_Rxy_aps3: float = 0.8 * (np.pi / 180) / 3600.0  # rad
    sigma_Rz_aps3: float = 7.0 * (np.pi / 180) / 3600.0  # rad

    # IMU Gyroscope ARW noise (Astrix NS IMU, Exail Astrix Series)
    # https://www.exail.com/product-range/astrix-series
    # Converted to rad/s and scaled for 10 Hz measurements:
    sigma_omega: float = np.deg2rad(0.0025/60.0) * np.sqrt(10.0/2.0)
    # Accelerometer noise is less often published in detail for space IMUs;
    # using a nominal value based on typical performance.
    sigma_accel: float = 10.0 * 1e-6 * 9.80665 * np.sqrt(10.0 / 2.0)
    MEAN_DEFAULT: List[float] = [0.0, 0.0, 0.0]

    # Bias models
    # Gyro bias stability < 0.005 deg/h (Astrix NS published spec).
    # Modeled as first-order Gauss-Markov with 1-hour correlation time.
    GYRO_BIAS_SIGMA_DEGPHR: float = 0.005
    GYRO_BIAS_TAU_S: float = 3600.0
    # Accelerometer bias modeled as 10 µg (typical navigation-grade
    # IMU assumption). No explicit public Astrix NS accel bias spec available.
    ACCEL_BIAS_SIGMA_UG: float = 10.0
    ACCEL_BIAS_TAU_S: float = 3600.0

    # Distance to target
    R0_const: float = 30.0
    # Sun alignment
    SUN_ALIGN_ENABLE: bool = True
    SUN_ALIGN_CONE_DEG: float = 12.0
    SUN_ALIGN_JITTER_D: float = 4.0
    # Earth-in-background alignment: Sun -> Camera -> Target -> Earth
    EARTH_BACKGROUND_ENABLE: bool = True
    # ---------- Constants / environment ----------
    mu_ref: float = 3.986004418e14      # Earth mu (m^3/s^2)
    h_orbit: float = 550e3              # circular altitude (m)
    R_earth: float = 6371e3             # Earth radius (m)
    # Time settings
    IMAGE_MAX_DT_S: float = 1.0
    tend: float = 500.0
    tstep: float = 0.5
    MIN_F2F_PX_MED: float = 3.0

    inertia_config: InertiaConfig = Field(default_factory=default_inertia_config)

    @property
    def a_ref(self) -> float:
        return self.R_earth + self.h_orbit
    
    @property
    def n_scalar(self) -> float:
        return np.sqrt(self.mu_ref / self.a_ref**3)
    
    @property
    def COV_R_ASTRO_APS3(self) -> List[List[float]]:
        return [[self.sigma_Rxy_aps3**2, 0, 0], [0, self.sigma_Rxy_aps3**2, 0], [0, 0, self.sigma_Rz_aps3**2]]
    
    @property
    def COV_OMEGA_ASTRIX(self) -> List[List[float]]:
        return [[self.sigma_omega**2, 0, 0], [0, self.sigma_omega**2, 0], [0, 0, self.sigma_omega**2]]
    
    @property
    def COV_ACCEL_ASTRIX(self) -> List[List[float]]:
        return [[self.sigma_accel**2, 0, 0], [0, self.sigma_accel**2, 0], [0, 0, self.sigma_accel**2]]
    
    # TODO this would be much better done with an enum will implement later tho
    # This has a fair amount of overhead as it is referenced in multiple files but is pretty important for code maintainability so I think its worth it
    @property
    def rotMode_Gframe(self) -> str:
        if self.path_mode == "inertial":
            return "1"
        elif self.path_mode == "hill":
            return "2"
        elif self.path_mode == "tumbling":
            return "3"
        else:
            raise ValueError(f"Invalid path_mode: {self.path_mode}")

def default_trajectory_config(data: dict) -> TrajectoryConfig:
    if not data.get("selected_model"):
        raise ValueError("model_name must be specified if trajectory_config is not provided")
    
    return TrajectoryConfig(selected_model=data["selected_model"])

def default_model_rotation(data: dict) -> tuple[float, float, float]:
    if not data.get("selected_model"):
        return (0.0, 0.0, 0.0) # No rotation

    if data["selected_model"] not in spacecraft_defaults:
        return (0.0, 0.0, 0.0) # No rotation
    
    model_defaults = spacecraft_defaults[data["selected_model"]]
    model_rotation = model_defaults.get("model_rotation_euler")
    model_rotation = (model_rotation["x"], model_rotation["y"], model_rotation["z"])
    # Default to identity rotation if not specified
    return model_rotation

class SceneConfig(BaseModel):
    """Total Configuration, model and output"""
    scene_blend_path: str = "assets/scene.blend"
    hdri_path: str = "assets/starmap_2020_16k.exr"
    objects: Dict[str, ObjectConfig] = Field(default_factory=dict)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    setup: SetupConfig = Field(default_factory=SetupConfig)
    # Vision Blender addon settings
    save_depth: bool = True
    save_normals: bool = True
    save_optical_flow: bool = True
    save_segmentation: bool = True
    save_obj_poses: bool = True
    save_scene_plots: bool = True

    selected_model: str = "RF_Hubble"
    
    # Rendering control
    frame_ids: Optional[List[int]] = None  # If None, use all frames
    model_rotation_quat: Optional[float] = None # To align model with propper inertia calculation

    trajectory_type: Literal["trajectory_generator", "sampling_trajectory", "filepath", "const_rotate"] = "trajectory_generator"
    trajectory_sampling: SamplingTrajectoryConfig = Field(default_factory=SamplingTrajectoryConfig)
    trajectory_const_rotate: ConstantRotationConfig = Field(default_factory=ConstantRotationConfig)
    trajectory: TrajectoryConfig = Field(default_factory=default_trajectory_config)
    trajectory_filepath: Optional[str] = ""

    model_rotation_A_model_euler: tuple[float, float, float] = Field(default_factory=default_model_rotation)#XYZ

class SweepConfig(BaseModel):
    """
    Configuration for parameter sweeps
    Define the parameters to sweep over and the base config
    Validates the output configs
    """
    base_config: SceneConfig
    sweep_parameters: Dict[str, List[Any]] = Field(default_factory=dict) # Dict of parameter full path to list of values to sweep over

    @staticmethod
    def _set_nested_attr(obj: Any, param_path: str, value: Any) -> None:
        """Set a nested attribute or dict key using a dot-delimited path."""
        parts = param_path.split(".")
        if not parts:
            raise ValueError("param_path must be non-empty")

        target = obj
        for part in parts[:-1]:
            if isinstance(target, dict):
                if part not in target:
                    raise KeyError(f"Missing key '{part}' in path '{param_path}'")
                target = target[part]
            else:
                if not hasattr(target, part):
                    raise AttributeError(f"Missing attribute '{part}' in path '{param_path}'")
                target = getattr(target, part)

        last = parts[-1]
        if isinstance(target, dict):
            if last not in target:
                raise KeyError(f"Missing key '{last}' in path '{param_path}'")
            target[last] = value
        else:
            if not hasattr(target, last):
                raise AttributeError(f"Missing attribute '{last}' in path '{param_path}'")
            setattr(target, last, value)

    def generate_sweep_configs(self) -> List[SceneConfig]:
        """Generate a list of SceneConfig instances for each combination of sweep parameters"""
        # Generate all combinations of sweep parameters
        if not self.sweep_parameters:
            return [copy.deepcopy(self.base_config)]

        keys, values = zip(*self.sweep_parameters.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        sweep_configs = []
        for combo in combinations:
            config_copy = copy.deepcopy(self.base_config)
            for param_path, value in combo.items():
                self._set_nested_attr(config_copy, param_path, value)
            sweep_configs.append(config_copy)

        return sweep_configs
