from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json

@dataclass
class ObjectConfig:
    name: str
    blend_path: Optional[str] = None      
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    hide_render: bool = False
    extra_scale: float = 1.0              # Potential dditional scaling after loading

@dataclass
class CameraConfig:
    """Camera properties"""
    focal_length: float = 400.0
    clip_start: float = 0.00001
    clip_end: float = 5000000.0
    resolution: tuple = (480, 480)

@dataclass
class RenderConfig:
    """Render settings"""
    engine: str = "CYCLES"
    samples: int = 32
    bg_color: tuple = (0.0, 0.0, 0.0, 1.0)
    motion_blur: float = 0.0
    noise_strength: float = 0.0

@dataclass
class SetupConfig:
    """Trajectory and Environment Setup"""
    num_frames: int = 200
    R_RPO: float = 70.0
    R_LEO: float = 10000.0
    # Sweep definitions (optional)
    sweep_exposure: Optional[Dict[str, float]] = None 
    sweep_sun_az_el: Optional[Dict[str, Dict[str, float]]] = None
     # Reference settings
    t_ref_s: float = 0.01666667
    sun_az_ref: float = 0.0
    sun_el_ref: float = 0.0
    earth_mode: str = "on"
    stars_mode: str = "on"
    enable_blur: str = "off"
    blur_shutter_factor: float =  0.8
    blur_motion_factor: float =  0.8
    enable_glare: str = "OFF"
    glare_threshold: float = 0.95
    glare_size: int = 6

@dataclass
class SceneConfig:
    """Total Configuration, model and output"""
    scene_blend_path: str
    hdri_path: str
    objects: Dict[str, ObjectConfig] = field(default_factory=dict)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    exposure_times_s: List[float] = field(default_factory=lambda: [1/60])
    # Vision Blender addon settings
    save_depth: bool = True
    save_normals: bool = True
    save_optical_flow: bool = True
    save_segmentation: bool = True
    save_obj_poses: bool = True
    
    # Rendering control
    frame_ids: Optional[List[int]] = None  # If None, use all frames
    selected_models: List[str] = field(default_factory=list)  # Empty = render all RF_* models
    model_rotation_z_deg: float = 45.0  # Apply initial Z rotation, will be extended to X,Y
    
    @classmethod
    def from_json(cls, json_path: str) -> "SceneConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'camera' in data:
            data['camera'] = CameraConfig(**data['camera'])
        if 'render' in data:
            data['render'] = RenderConfig(**data['render'])
        if 'setup' in data:
            data['setup'] = SetupConfig(**data['setup'])
        if 'objects' in data:
            data['objects'] = {k: ObjectConfig(**v) for k, v in data['objects'].items()}
        
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file"""
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)
        
        with open(json_path, 'w') as f:
            json.dump(asdict(self, dict_factory=lambda x: {k: serialize(v) for k, v in x}), 
                     f, indent=2)