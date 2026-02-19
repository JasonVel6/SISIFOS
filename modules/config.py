"""
config.py

Defines the configuration schema for the trajectory generator and renderer

"""

import time
from pydantic import BaseModel, Field, computed_field
from typing import List, Dict, Optional, Tuple, Any, Literal
import json
import numpy as np
import itertools
import copy
import enum

class ObjectConfig(BaseModel):
    name: str
    blend_path: Optional[str] = None
    scale_factor: float = 1.0


class CameraConfig(BaseModel):
    """Camera properties"""
    focal_length: float = 400.0
    clip_start: float = 0.00001
    clip_end: float = 5000000.0
    resolution: Tuple[int, int] = (480, 480)
    lens_flare: float = 0.01 # TODO see if we really need this because it is only used in the trajectory generator
    exposure_time_s: float = 1.0/60.0 # TODO resolve this exposure time being in multiple places and not being used in the renderer, should be unified and used in both places


class RenderConfig(BaseModel):
    """Render settings"""
    engine: str = "CYCLES"
    samples: int = 32
    bg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    motion_blur: float = 0.0
    noise_strength: float = 0.0
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


class SamplingTrajectoryConfig(BaseModel):
    """Configuration for the sampling-based trajectory generator"""
    num_frames: int = 200
    R_RPO: float = 70.0
    R_LEO: float = 10000.0
    sun_az: float = 0.0
    sun_el: float = 0.0


class TrajectoryConfig(BaseModel):
    """Trajectory generation settings"""
    # Commonly changed parameters
    path_mode: Literal["inertial", "hill", "tumbling"] = "tumbling"
    seed: Optional[int] = None # For reproducibility
    num_agents: int = 1
    num_mc: int = 1

    r_AG_G: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Sensor noise (ASTRO APS3 star tracker and Astrix NS IMU at 10 Hz)
    sigma_Rxy_aps3: float = 0.8 * (np.pi / 180) / 3600  # rad (0.8 arcsec)
    sigma_Rz_aps3: float = 7.0 * (np.pi / 180) / 3600   # rad (7.0 arcsec)
    sigma_omega: float = np.deg2rad(0.0025 / 60.0) * np.sqrt(10.0 / 2.0)
    sigma_accel: float = 10.0 * 1e-6 * 9.80665 * np.sqrt(10.0 / 2.0)
    MEAN_DEFAULT: List[float] = [0.0, 0.0, 0.0]
    # Bias models
    GYRO_BIAS_SIGMA_DEGPHR: float = 0.005
    GYRO_BIAS_TAU_S: float = 3600.0
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

    # Inertia (ESA INTEGRAL satellite, box approximation)
    # Dimensions: 2.8 x 3.2 x 5.0 m, mass ~4000 kg
    # Triaxial inertia avoids axisymmetric observability degeneracy
    m: float = 4000.0
    l: float = 5.0  # longest axis (5m), y=3.2m, z=2.8m
    w: float = 3.2
    h: float = 2.8

    # Trajectory generation camera params
    # TODO these should be unified with the CameraConfig for single source of truth
    FOCAL_LENGTH_PX: float = 2500.0
    CAMERA_RESOLUTION: int = 1024
    LENS_FLARE: float = 0.01

    @property
    def a_ref(self) -> float:
        return self.R_earth + self.h_orbit
    
    @property
    def n_scalar(self) -> float:
        return np.sqrt(self.mu_ref / self.a_ref**3)
    
    @property
    def J(self) -> np.ndarray:
        J_xx = (1/12) * self.m * (self.w**2 + self.h**2)  # rotation about x (longest)
        J_yy = (1/12) * self.m * (self.l**2 + self.h**2)  # rotation about y
        J_zz = (1/12) * self.m * (self.l**2 + self.w**2)  # rotation about z
        return np.array([[J_xx, 0, 0], [0, J_yy, 0], [0, 0, J_zz]])
    
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


class SceneConfig(BaseModel):
    """Total Configuration, model and output"""
    scene_blend_path: str
    hdri_path: str
    objects: Dict[str, ObjectConfig] = Field(default_factory=dict)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    setup: SetupConfig = Field(default_factory=SetupConfig)
    trajectory_type: Literal["trajectory_generator", "sampling_trajectory", "filepath"] = "trajectory_generator"
    trajectory_sampling: SamplingTrajectoryConfig = Field(default_factory=SamplingTrajectoryConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    trajectory_filepath: Optional[str] = None
    # Vision Blender addon settings
    save_depth: bool = True
    save_normals: bool = True
    save_optical_flow: bool = True
    save_segmentation: bool = True
    save_obj_poses: bool = True
    
    # Rendering control
    frame_ids: Optional[List[int]] = None  # If None, use all frames
    selected_model: str = "RF_Hubble"
    model_rotation_z_deg: float = 45.0  # Apply initial Z rotation, will be extended to X,Y


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
