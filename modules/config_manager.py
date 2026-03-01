import json
import os
from typing import Dict, Any


class ConfigManager:
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file '{self.config_file}' not found. "
                f"Please create it using config.example.json as a template."
            )
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {self.config_file}")
            return config
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file '{self.config_file}': {e.msg}",
                e.doc,
                e.pos
            )
    
    def get(self, *keys):
        value = self.config
        for i, key in enumerate(keys):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                key_path = '.'.join(keys[:i+1])
                raise KeyError(
                    f"Configuration key '{key_path}' not found in {self.config_file}. "
                    f"Please check your configuration file."
                )
        return value
    
    def set(self, *keys, value):
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save()
    
    def save(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def reload(self):
        self.config = self._load_config()
    
    def get_video_path(self):
        return self.get('video', 'path')
    
    def get_detection_enabled(self):
        return self.get('object_detection', 'enabled')
    
    def get_detection_config(self):
        return {
            'model_size': self.get('object_detection', 'model_size'),
            'confidence_threshold': self.get('object_detection', 'confidence_threshold'),
            'person_confidence_threshold': self.get('object_detection', 'person_confidence_threshold'),
            'skip_frames': self.get('object_detection', 'skip_frames'),
            'process_size': self.get('object_detection', 'process_size'),
            'alert_delay_seconds': self.get('object_detection', 'alert_delay_seconds'),
            'grace_period_seconds': self.get('object_detection', 'grace_period_seconds'),
            'min_detection_area_percent': self.get('object_detection', 'min_detection_area_percent'),
            'max_detection_area_percent': self.get('object_detection', 'max_detection_area_percent'),
            'messages': self.get('object_detection', 'messages'),
            'display': self.get('object_detection', 'display')
        }
    
    def get_pool_boundary_config(self):
        return {
            'min_points': self.get('pool_boundary', 'min_points'),
            'max_points': self.get('pool_boundary', 'max_points'),
            'color': tuple(self.get('pool_boundary', 'line_color')),
            'thickness': self.get('pool_boundary', 'line_thickness')
        }
    
    def get_point_selector_config(self):
        return {
            'point_radius': self.get('point_selector', 'point_radius'),
            'hover_radius': self.get('point_selector', 'hover_radius'),
            'window_name': self.get('point_selector', 'window_name')
        }
    
    def get_ui_config(self):
        return {
            'resizable_windows': self.get('ui', 'resizable_windows'),
            'font_scale': self.get('ui', 'font_scale'),
            'font_thickness': self.get('ui', 'font_thickness'),
            'status_indicator_position': self.get('ui', 'status_indicator_position')
        }
    
    def get_hotkeys(self):
        return {
            'quit': self.get('hotkeys', 'quit'),
            'toggle_detection': self.get('hotkeys', 'toggle_detection'),
            'reconfigure_boundary': self.get('hotkeys', 'reconfigure_boundary'),
            'clear_points': self.get('hotkeys', 'clear_points')
        }
    
    def get_performance_config(self):
        return {
            'video_frame_delay_ms': self.get('performance', 'video_frame_delay_ms')
        }
    
    def get_alert_box_config(self):
        return {
            'color_in_pool': tuple(self.get('object_detection', 'alert_box_color_in_pool')),
            'color_outside': tuple(self.get('object_detection', 'alert_box_color_outside')),
            'thickness_in_pool': self.get('object_detection', 'alert_box_thickness_in_pool'),
            'thickness_outside': self.get('object_detection', 'alert_box_thickness_outside')
        }
