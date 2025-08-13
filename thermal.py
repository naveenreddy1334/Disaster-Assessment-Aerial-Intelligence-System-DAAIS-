#!/usr/bin/env python3
"""
Thermal-Only Human Detection System
Uses MLX90640 thermal camera to detect humans based on temperature patterns
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Fix Qt platform plugin issue

import cv2
import numpy as np
import time
import warnings
import threading
import queue
import board
import busio
import adafruit_mlx90640
from scipy import ndimage

warnings.filterwarnings('ignore')


class ThermalHumanDetector:
    def __init__(self):
        self.frame_count = 0
        self.last_detections = []
        self.detection_history = []
        
        # Thermal detection parameters
        self.human_temp_min = 30.0  # Minimum human body temperature
        self.human_temp_max = 40.0  # Maximum human body temperature
        self.min_human_area = 15    # Minimum pixels for human detection
        self.max_human_area = 200   # Maximum pixels for human detection
        
        # Real thermal camera setup
        self.thermal_queue = queue.Queue(maxsize=3)
        self.current_thermal = None
        self.thermal_running = True
        self.ambient_temp = 25.0    # Estimated ambient temperature
        
        # Initialize thermal camera
        self._init_thermal_camera()
        
        print("Thermal-Only Human Detection System initialized")

    def _init_thermal_camera(self):
        """Initialize MLX90640 thermal camera"""
        try:
            print("Initializing MLX90640 thermal camera...")
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
            
            # Thermal data storage
            self.thermal_frame = [0] * 768
            
            # Start thermal data collection thread
            self.thermal_thread = threading.Thread(target=self._collect_thermal_data)
            self.thermal_thread.daemon = True
            self.thermal_thread.start()
            
            print("✅ MLX90640 thermal camera ready")
            return True
            
        except Exception as e:
            print(f"❌ Thermal camera initialization failed: {e}")
            return False

    def _collect_thermal_data(self):
        """Continuously collect thermal data in background"""
        while self.thermal_running:
            try:
                self.mlx.getFrame(self.thermal_frame)
                thermal_array = np.array(self.thermal_frame).reshape((24, 32))
                
                # Update ambient temperature estimate
                self.ambient_temp = np.percentile(thermal_array, 10)  # 10th percentile as ambient
                
                # Add to queue
                if not self.thermal_queue.full():
                    self.thermal_queue.put(thermal_array)
                else:
                    try:
                        self.thermal_queue.get_nowait()
                        self.thermal_queue.put(thermal_array)
                    except queue.Empty:
                        pass
                        
            except ValueError:
                continue
            except Exception as e:
                print(f"Thermal data error: {e}")
                time.sleep(0.1)

    def get_thermal_data(self):
        """Get latest thermal camera data"""
        if not self.thermal_queue.empty():
            try:
                self.current_thermal = self.thermal_queue.get_nowait()
            except queue.Empty:
                pass
        
        return self.current_thermal

    def detect_humans_thermal(self, thermal_array):
        """Detect humans using thermal data only"""
        if thermal_array is None:
            return []
        
        self.frame_count += 1
        
        # Create binary mask for human temperature range
        human_mask = ((thermal_array >= self.human_temp_min) & 
                     (thermal_array <= self.human_temp_max)).astype(np.uint8)
        
        # Also look for areas significantly warmer than ambient
        temp_diff_mask = (thermal_array > (self.ambient_temp + 5.0)).astype(np.uint8)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(human_mask, temp_diff_mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components (potential humans)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        detections = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area (human size)
            if self.min_human_area <= area <= self.max_human_area:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate center
                center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
                
                # Extract temperature data for this region
                region_mask = (labels == i)
                region_temps = thermal_array[region_mask]
                
                if len(region_temps) > 0:
                    avg_temp = region_temps.mean()
                    max_temp = region_temps.max()
                    min_temp = region_temps.min()
                    
                    # Calculate confidence based on temperature characteristics
                    temp_range = max_temp - min_temp
                    temp_consistency = 1.0 - (temp_range / 10.0)  # More consistent = higher confidence
                    temp_human_like = 1.0 if (32.0 <= avg_temp <= 38.0) else 0.5  # Body temp range
                    
                    confidence = min(1.0, (temp_consistency * temp_human_like * area / 50.0))
                    
                    # Filter by minimum confidence
                    if confidence > 0.3:
                        detections.append({
                            'bbox': (x, y, x + w, y + h),
                            'center': (center_x, center_y),
                            'confidence': confidence,
                            'avg_temp': avg_temp,
                            'max_temp': max_temp,
                            'min_temp': min_temp,
                            'area': area,
                            'thermal_coords': (x, y, w, h)  # Keep thermal coordinates
                        })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Temporal filtering for stability
        self.last_detections = self._apply_temporal_filter(detections)
        
        return self.last_detections

    def _apply_temporal_filter(self, detections):
        """Apply temporal filtering to reduce false positives"""
        # Keep detection history
        self.detection_history.append(detections)
        if len(self.detection_history) > 5:
            self.detection_history.pop(0)
        
        # For now, just return current detections
        # Could implement more sophisticated tracking here
        return detections

    def create_thermal_visualization(self, thermal_array, detections, target_size=(640, 480)):
        """Create thermal visualization with human detections"""
        if thermal_array is None:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Normalize thermal data for visualization
        temp_min, temp_max = thermal_array.min(), thermal_array.max()
        
        if temp_max > temp_min:
            normalized = ((thermal_array - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        else:
            normalized = np.full_like(thermal_array, 128, dtype=np.uint8)
        
        # Resize thermal to target size
        thermal_resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        thermal_colored = cv2.applyColorMap(thermal_resized, cv2.COLORMAP_JET)
        
        # Scale detection coordinates to display size
        scale_x = target_size[0] / 32
        scale_y = target_size[1] / 24
        
        # Draw detections on thermal image
        for i, det in enumerate(detections):
            thermal_x, thermal_y, thermal_w, thermal_h = det['thermal_coords']
            
            # Scale to display coordinates
            x1 = int(thermal_x * scale_x)
            y1 = int(thermal_y * scale_y)
            x2 = int((thermal_x + thermal_w) * scale_x)
            y2 = int((thermal_y + thermal_h) * scale_y)
            
            confidence = det['confidence']
            avg_temp = det['avg_temp']
            
            # Color based on confidence and temperature
            if avg_temp > 35.0:  # Very human-like temperature
                color = (0, 255, 0)  # Green
                thickness = 3
            elif avg_temp > 32.0:  # Human-like temperature
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:  # Warm but uncertain
                color = (0, 165, 255)  # Orange
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(thermal_colored, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(thermal_colored, (center_x, center_y), 4, color, -1)
            
            # Temperature label
            label = f"Human {i+1}: {avg_temp:.1f}°C ({confidence:.2f})"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(thermal_colored, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            
            # Text
            cv2.putText(thermal_colored, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return thermal_colored, temp_min, temp_max

    def draw_thermal_info(self, frame, thermal_array, detections):
        """Draw thermal information panel"""
        if thermal_array is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Info panel
        panel_width = 280
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Thermal statistics
        temp_min, temp_max = thermal_array.min(), thermal_array.max()
        temp_avg = thermal_array.mean()
        
        y_offset = 20
        cv2.putText(frame, "THERMAL HUMAN DETECTION", (w - panel_width + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Temperature Range: {temp_min:.1f} - {temp_max:.1f}°C", 
                   (w - panel_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Average: {temp_avg:.1f}°C | Ambient: {self.ambient_temp:.1f}°C", 
                   (w - panel_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Humans Detected: {len(detections)}", 
                   (w - panel_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Individual human info
        if detections:
            y_offset += 20
            cv2.putText(frame, "Detected Temperatures:", 
                       (w - panel_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            for i, det in enumerate(detections[:3]):  # Show up to 3 humans
                y_offset += 15
                temp_text = f"  Human {i+1}: {det['avg_temp']:.1f}°C"
                cv2.putText(frame, temp_text, 
                           (w - panel_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        return frame

    def stop(self):
        """Stop thermal data collection"""
        self.thermal_running = False
        if hasattr(self, 'thermal_thread') and self.thermal_thread.is_alive():
            self.thermal_thread.join()


def main():
    print("🔥 THERMAL-ONLY HUMAN DETECTION SYSTEM")
    print("MLX90640 Temperature-Based Human Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = ThermalHumanDetector()
    
    if not hasattr(detector, 'thermal_thread'):
        print("❌ Thermal camera initialization failed")
        return
    
    print("\n🎮 CONTROLS:")
    print("  Q - Quit")
    print("  S - Save thermal frame")
    print("  + - Increase sensitivity (lower min temp)")
    print("  - - Decrease sensitivity (higher min temp)")
    print("  A - Adjust ambient temperature")
    print("\n🚀 Starting thermal human detection...")
    
    # Performance tracking
    fps_history = []
    last_time = time.time()
    save_counter = 0
    
    try:
        while True:
            # Get thermal data
            thermal_array = detector.get_thermal_data()
            
            if thermal_array is not None:
                # Detect humans using thermal data
                detections = detector.detect_humans_thermal(thermal_array)
                
                # Create thermal visualization
                thermal_display, temp_min, temp_max = detector.create_thermal_visualization(
                    thermal_array, detections, (640, 480)
                )
                
                # Add information panel
                final_frame = detector.draw_thermal_info(thermal_display, thermal_array, detections)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time + 0.001)
                last_time = current_time
                
                fps_history.append(fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                
                # Add main title
                cv2.putText(final_frame, f"THERMAL HUMAN DETECTION | FPS: {avg_fps:.0f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display
                cv2.imshow('Thermal Human Detection', final_frame)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if thermal_array is not None:
                    timestamp = int(time.time())
                    filename = f"thermal_humans_{timestamp}.jpg"
                    cv2.imwrite(filename, final_frame)
                    save_counter += 1
                    print(f"💾 Saved: {filename} (#{save_counter})")
            elif key == ord('+') or key == ord('='):
                detector.human_temp_min = max(25.0, detector.human_temp_min - 1.0)
                print(f"📈 Increased sensitivity - Min temp: {detector.human_temp_min:.1f}°C")
            elif key == ord('-'):
                detector.human_temp_min = min(35.0, detector.human_temp_min + 1.0)
                print(f"📉 Decreased sensitivity - Min temp: {detector.human_temp_min:.1f}°C")
            elif key == ord('a'):
                if thermal_array is not None:
                    detector.ambient_temp = np.percentile(thermal_array, 15)
                    print(f"🌡️ Ambient temperature updated: {detector.ambient_temp:.1f}°C")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        detector.stop()
        cv2.destroyAllWindows()
        print("🔌 System shutdown complete")


if __name__ == "__main__":
    main()
