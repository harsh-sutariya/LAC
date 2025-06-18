import os
import numpy as np
import carla
import cv2 as cv
import random
import time
from math import radians
from pynput import keyboard
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from ultralytics import YOLO


def get_entry_point():
    return 'YOLOAgent'


class YOLOAgent(AutonomousAgent):
    """
    YOLO-based autonomous agent for lunar navigation with real-time object detection
    """

    def setup(self, path_to_conf_file):
        """Initialize the agent with YOLO model and sensor configurations"""
        
        # Setup keyboard listener for manual control
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Initialize control variables
        self.current_v = 0
        self.current_w = 0
        self.view = 0
        self.capture = False
        self.timer = time.time()
        
        # Data storage arrays
        self.store_imu = np.zeros((0, 6), dtype=float)
        self.store_pose = np.zeros((0, 6), dtype=float)
        self.store_inp = np.zeros((0, 2), dtype=float)

        # Create capture directories for active cameras
        self.capture_folder = "captured_images"
        self._create_capture_folders()

        # Camera configuration
        self.camera_positions = {
            'front_left': carla.SensorPosition.FrontLeft,
            'front_right': carla.SensorPosition.FrontRight,
            'left': carla.SensorPosition.Left,
            'right': carla.SensorPosition.Right,
            'back_left': carla.SensorPosition.BackLeft,
            'back_right': carla.SensorPosition.BackRight,
            'front': carla.SensorPosition.Front,
            'back': carla.SensorPosition.Back,
        }
        self.selected_camera = 'front'

        # Initialize YOLO model
        self.model = YOLO('models/yolo.pt')

        # Class mapping for semantic segmentation
        self.class_mapping = {
            (81, 0, 81): 0,         # Moon terrain/regolith
            (42, 59, 108): 1,       # Rocks
            (142, 0, 0): 2,         # Robot
            (160, 190, 110): 3,     # Lander
            (30, 170, 250): 4,      # Fiducials
            (180, 130, 70): 5       # Earth
        }
        self.color_mapping = {v: k for k, v in self.class_mapping.items()}

    def _create_capture_folders(self):
        """Create necessary directories for image capture"""
        folders = ['FR', 'BR', 'L', 'R']
        
        if not os.path.exists(self.capture_folder):
            os.makedirs(self.capture_folder)
            
        for folder in folders:
            folder_path = os.path.join(self.capture_folder, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def use_fiducials(self):
        """Enable fiducial detection"""
        return True

    def sensors(self):
        """Configure sensor parameters for all camera positions"""
        sensor_config = {
            'camera_active': True, 
            'light_intensity': 1.0, 
            'width': '1224', 
            'height': '1024', 
            'use_semantic': True
        }
        
        sensors = {
            carla.SensorPosition.FrontLeft: sensor_config,
            carla.SensorPosition.FrontRight: sensor_config,
            carla.SensorPosition.Left: sensor_config,
            carla.SensorPosition.Right: sensor_config,
            carla.SensorPosition.BackLeft: sensor_config,
            carla.SensorPosition.BackRight: sensor_config,
            carla.SensorPosition.Front: sensor_config,
            carla.SensorPosition.Back: sensor_config,
        }
        return sensors
    
    def draw_mask_annotations(self, image, masks, classes, title):
        """Draw YOLO detection masks on the image with class labels"""
        img_copy = image.copy()
        img_height, img_width = image.shape[:2]
        
        # Add title at the top of the image
        cv.putText(img_copy, title, (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for mask, cls in zip(masks, classes):
            color = self.color_mapping.get(cls, (255, 255, 255))  # Default to white if class not found
            
            # Resize mask to match image dimensions
            mask = cv.resize(mask.astype(float), (img_width, img_height))
            mask = mask > 0.5  # Convert to boolean

            # Create overlay mask
            overlay = img_copy.copy()
            overlay[mask] = color

            # Blend the overlay with the original image
            alpha = 0.4
            cv.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

            # Add class label
            label = f'Class {cls}'
            y, x = np.where(mask)
            if len(y) > 0 and len(x) > 0:
                cv.putText(img_copy, label, (x[0], y[0] - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy

    def run_step(self, input_data):
        """Main control loop - process sensor data and return vehicle control"""
        
        # Set arm angles on first run
        if self.view == 0:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Get sensor data from active cameras
        sensor_data = {
            'FR': input_data['Grayscale'][carla.SensorPosition.FrontRight],
            'BR': input_data['Grayscale'][carla.SensorPosition.BackRight],
            'L': input_data['Grayscale'][carla.SensorPosition.Left],
            'R': input_data['Grayscale'][carla.SensorPosition.Right]
        }

        # Process front-right camera for YOLO detection
        if sensor_data['FR'] is not None:
            img = sensor_data['FR']
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        
            # Run YOLO inference
            results = self.model(img)

            # Extract predictions
            pred_masks = []
            pred_classes = []
            for r in results:
                masks = r.masks
                if masks is not None:
                    for i, mask in enumerate(masks.data):
                        pred_masks.append(mask.cpu().numpy())
                        pred_classes.append(int(r.boxes.cls[i]))

            # Draw predictions on image
            pred_img = self.draw_mask_annotations(img, pred_masks, pred_classes, "YOLO Predictions")
            
            # Display camera feeds
            cv.imshow(f'{self.selected_camera} camera view', sensor_data['FR'])
            cv.imshow(f'{self.selected_camera} segmented view', pred_img)
            cv.waitKey(1)

        # Return vehicle control
        control = carla.VehicleVelocityControl(self.current_v, self.current_w)
        return control

    def capture_image(self, sensor_data, sensor_position):
        """Capture and save image from specified sensor"""
        filename = os.path.join(self.capture_folder, f"{sensor_position}/{sensor_position}_{self.view:04d}.png")
        cv.imwrite(filename, sensor_data)
        print(f"Captured image: {filename}")

    def finalize(self):
        """Clean up resources and generate final geometric map"""
        cv.destroyAllWindows()

        # Generate random geometric map data (placeholder)
        geometric_map = self.get_geometric_map()
        for i in range(100):
            x = 10 * random.random() - 5
            y = 10 * random.random() - 5
            geometric_map.set_height(x, y, random.random())
            rock_flag = random.random() > 0.5
            geometric_map.set_rock(x, y, rock_flag)

    def on_press(self, key):
        """Handle keyboard press events for manual control"""
        # Movement controls
        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        elif key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        elif key == keyboard.Key.left:
            self.current_w = 0.6
        elif key == keyboard.Key.right:
            self.current_w = -0.6      

        # Camera switching controls
        camera_keys = {
            keyboard.Key.f1: ('front_left', 'Front Left'),
            keyboard.Key.f2: ('front_right', 'Front Right'),
            keyboard.Key.f3: ('left', 'Left'),
            keyboard.Key.f4: ('right', 'Right'),
            keyboard.Key.f5: ('back_left', 'Back Left'),
            keyboard.Key.f6: ('back_right', 'Back Right'),
            keyboard.Key.f7: ('front', 'Front'),
            keyboard.Key.f8: ('back', 'Back'),
        }
        
        if key in camera_keys:
            self.selected_camera, display_name = camera_keys[key]
            print(f"Switched to {display_name} Camera")
        
        # Capture control
        if hasattr(key, 'char') and key.char == 'c':
            self.capture = True

    def on_release(self, key):
        """Handle keyboard release events"""
        # Stop movement when keys are released
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.current_v = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.current_w = 0
        elif hasattr(key, 'char') and key.char == 'c':
            self.capture = False
        elif key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()