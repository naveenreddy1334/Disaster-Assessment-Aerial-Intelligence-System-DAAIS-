import cv2
import numpy as np
import torch
import time
from collections import deque
import math
import warnings

warnings.filterwarnings('ignore')


class DualViewEOIRDetector:
    def __init__(self, model_size='n', conf_thresh=0.5, iou_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.fps_history = deque(maxlen=30)
        self.heat_signatures = {}
        self.last_thermal = None
        self.frame_count = 0
        self.detection_memory = {}
        self.ambient_cache = None
        self.thermal_kernel = cv2.getGaussianKernel(25, 0)
        
        # Raspberry Pi optimizations
        self.rpi_optimizations = True
        self.process_every_n_frames = 3
        self.thermal_update_interval = 5

        # Setup device
        self.device = torch.device('cpu')
        self._setup_device()
        self._load_model(model_size)
        self._init_thermal_profiles()

    def _setup_device(self):
        print("💻 Raspberry Pi CPU mode - optimized")
        torch.set_num_threads(2)

    def _load_model(self, size):
        try:
            from ultralytics import YOLO

            model_name = 'yolov8n'  # Force nano model for RPi
            print(f"🔧 Loading {model_name} (RPi optimized)...")

            self.model = YOLO(f'{model_name}.pt')
            self.model.to(self.device)
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self.model.max_det = 30
            self.model.amp = False

            self.class_names = list(self.model.names.values())
            print(f"✅ Model ready | Classes: {len(self.class_names)}")
            return True

        except ImportError:
            print("❌ Install ultralytics: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Model error: {e}")
            return False

    def _init_thermal_profiles(self):
        self.thermal_profiles = {
            'person': (0.85, 0.10, 1.0),
            'car': (0.80, 0.12, 0.9),
            'dog': (0.82, 0.08, 0.8),
            'cat': (0.80, 0.09, 0.7),
            'motorcycle': (0.78, 0.11, 0.8),
            'bus': (0.77, 0.05, 1.1),
            'truck': (0.76, 0.06, 1.0),
            'bicycle': (0.60, 0.10, 0.7),
            'chair': (0.35, 0.05, 0.6),
            'book': (0.25, 0.03, 0.4),
            'cell phone': (0.30, 0.04, 0.3),
            'bottle': (0.20, 0.03, 0.3),
            '*default*': (0.40, 0.10, 0.6)
        }

    def _get_thermal_profile(self, class_name):
        return self.thermal_profiles.get(
            class_name,
            self.thermal_profiles['*default*']
        )

    def _generate_ambient(self, height, width):
        if self.ambient_cache is None or self.ambient_cache.shape != (height, width):
            cy, cx = height // 2, width // 2
            y, x = np.indices((height, width))
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            ambient = np.maximum(0.2, 0.4 * (1 - dist / max_dist))
            ambient = cv2.sepFilter2D(ambient, -1, self.thermal_kernel, self.thermal_kernel)
            self.ambient_cache = ambient
        return self.ambient_cache

    def generate_thermal(self, frame, detections):
        # Update thermal map less frequently on RPi
        if self.frame_count % self.thermal_update_interval != 0 and self.last_thermal is not None:
            return self.last_thermal
            
        height, width = frame.shape[:2]
        ambient = self._generate_ambient(height, width)
        heat_map = ambient.copy()

        # Precompute object heat signatures
        obj_temps = []
        obj_props = []
        obj_centers = []
        obj_radii = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            base_temp, temp_var, heat_radius = self._get_thermal_profile(class_name)

            obj_id = f"{class_name}_{x1}_{y1}"
            if obj_id not in self.heat_signatures:
                self.heat_signatures[obj_id] = base_temp + np.random.uniform(-temp_var, temp_var)

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            diagonal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            max_rs = diagonal * heat_radius

            obj_temps.append(self.heat_signatures[obj_id])
            obj_centers.append((center_x, center_y))
            obj_radii.append(max_rs)
            obj_props.append((x1, y1, x2, y2))

        # Create heat map using vectorized operations
        y, x = np.indices((height, width))
        for i in range(len(obj_temps)):
            cx, cy = obj_centers[i]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            intensity = np.maximum(0, 1.0 - np.minimum(1.0, dist / (obj_radii[i] + 1e-5)))
            obj_heat = obj_temps[i] * intensity

            # Apply only within bounding box
            x1, y1, x2, y2 = obj_props[i]
            region = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
            heat_map[region] = np.maximum(heat_map[region], obj_heat[region])

        heat_map = np.clip(heat_map * 255, 0, 255).astype(np.uint8)
        thermal = cv2.applyColorMap(heat_map, cv2.COLORMAP_INFERNO)

        noise = np.random.normal(0, 6, thermal.shape).astype(np.uint8)
        thermal = cv2.add(thermal, noise)

        if self.last_thermal is not None:
            thermal = cv2.addWeighted(thermal, 0.8, self.last_thermal, 0.2, 0)

        self.last_thermal = thermal.copy()
        return thermal

    def detect(self, frame):
        self.frame_count += 1
        
        # Skip frames for better performance on RPi
        if self.frame_count % self.process_every_n_frames != 0 and hasattr(self, 'last_detections'):
            return self.last_detections
        
        # Reduce image size for faster processing
        height, width = frame.shape[:2]
        scale_factor = 0.7
        small_frame = cv2.resize(frame, 
                               (int(width * scale_factor), int(height * scale_factor)))

        try:
            with torch.no_grad():
                results = self.model(small_frame, imgsz=416, verbose=False)

            detections = []
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original size
                boxes[:, [0, 2]] *= (width / (width * scale_factor))
                boxes[:, [1, 3]] *= (height / (height * scale_factor))

                valid_mask = (confs >= self.conf_thresh)
                boxes = boxes[valid_mask]
                confs = confs[valid_mask]
                class_ids = class_ids[valid_mask]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    if (x2 - x1) < 15 or (y2 - y1) < 15:
                        continue

                    class_name = self.class_names[class_ids[i]]
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confs[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': class_name
                    })

            detections = self._stability_filter(detections)
            self.last_detections = detections
            return detections

        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return getattr(self, 'last_detections', [])

    def _stability_filter(self, detections):
        current_frame = self.frame_count
        stable_dets = []

        for det in detections:
            cls_name = det['class_name']
            conf = det['confidence']
            key = f"{cls_name}_{det['bbox'][0]}_{det['bbox'][1]}"

            if key not in self.detection_memory:
                self.detection_memory[key] = {
                    'count': 1,
                    'total_conf': conf,
                    'last_seen': current_frame
                }
                if conf > self.conf_thresh + 0.2:
                    det['stability'] = 1.0
                    stable_dets.append(det)
            else:
                mem = self.detection_memory[key]
                mem['count'] += 1
                mem['total_conf'] += conf
                mem['last_seen'] = current_frame

                if mem['count'] > 2:
                    det['stability'] = min(1.0, mem['count'] / 5)
                    stable_dets.append(det)

        # Cleanup old detections
        for key in list(self.detection_memory.keys()):
            if current_frame - self.detection_memory[key]['last_seen'] > 30:
                del self.detection_memory[key]

        return stable_dets

    def draw_detections(self, frame, detections, mode='rgb'):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            conf = det['confidence']

            if mode == 'thermal':
                base_temp, _, _ = self._get_thermal_profile(cls_name)
                if base_temp > 0.7:
                    color = (0, 255, 255)
                    thickness = 2
                elif base_temp > 0.5:
                    color = (0, 165, 255)
                    thickness = 1
                else:
                    color = (255, 255, 255)
                    thickness = 1
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if mode == 'thermal':
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 3, color, -1)

            label = f"{cls_name} {conf:.1f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)

            text_color = (0, 0, 0) if mode == 'thermal' else (255, 255, 255)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return annotated

    def draw_ui_overlay(self, frame, detections, fps, mode, window_size):
        h, w = window_size[:2]
        cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)

        if mode == 'thermal':
            title = "🔥 THERMAL IMAGING"
            title_color = (0, 200, 255)
        else:
            title = "📹 RGB CAMERA"
            title_color = (255, 255, 255)

        cv2.putText(frame, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)

        stats = f"FPS:{fps:.1f} OBJ:{len(detections)}"
        cv2.putText(frame, stats, (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if mode == 'thermal':
            cx, cy = w // 2, h // 2
            crosshair_color = (0, 255, 255)
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), crosshair_color, 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), crosshair_color, 1)

        return frame


def main():
    print("🔥 DUAL VIEW EOIR DETECTION SYSTEM")
    print("🌡️ Thermal | 📹 RGB")
    print("=" * 40)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = DualViewEOIRDetector(
        model_size='n',
        conf_thresh=0.4,
        iou_thresh=0.45
    )

    if not hasattr(detector, 'model'):
        return

    print("\nCONTROLS:")
    print("  Q - Quit")
    print("  S - Save both views")
    print("  + - Increase sensitivity")
    print("  - - Decrease sensitivity")
    print("  R - Reset thermal signatures")
    print("\nStarting dual view system...")

    fps_counter = deque(maxlen=10)
    last_time = time.time()
    window_width = 400
    window_height = 300

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (window_width, window_height))
            start_time = time.time()
            detections = detector.detect(frame_resized)
            thermal_view = detector.generate_thermal(frame_resized, detections)
            rgb_view = frame_resized.copy()

            thermal_annotated = detector.draw_detections(thermal_view, detections, 'thermal')
            rgb_annotated = detector.draw_detections(rgb_view, detections, 'rgb')

            current_time = time.time()
            fps = 1 / (current_time - last_time + 1e-5)
            last_time = current_time
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)

            thermal_final = detector.draw_ui_overlay(thermal_annotated, detections,
                                                     avg_fps, 'thermal',
                                                     (window_width, window_height))
            rgb_final = detector.draw_ui_overlay(rgb_annotated, detections,
                                                 avg_fps, 'rgb',
                                                 (window_width, window_height))

            combined_view = np.hstack((thermal_final, rgb_final))
            cv2.imshow('🔥 DUAL VIEW EOIR SYSTEM - Thermal | RGB', combined_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"thermal_{timestamp}.jpg", thermal_final)
                cv2.imwrite(f"rgb_{timestamp}.jpg", rgb_final)
                cv2.imwrite(f"combined_{timestamp}.jpg", combined_view)
                print(f"💾 Saved both views with timestamp {timestamp}")
            elif key in (ord('+'), ord('=')):
                detector.conf_thresh = max(0.1, detector.conf_thresh - 0.05)
                detector.model.conf = detector.conf_thresh
                print(f"🔼 Sensitivity: {detector.conf_thresh:.2f}")
            elif key == ord('-'):
                detector.conf_thresh = min(0.9, detector.conf_thresh + 0.05)
                detector.model.conf = detector.conf_thresh
                print(f"🔽 Sensitivity: {detector.conf_thresh:.2f}")
            elif key == ord('r'):
                detector.heat_signatures = {}
                print("🔄 Reset thermal signatures")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Dual view system shutdown")


if __name__ == "__main__":
    main()
