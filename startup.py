#!/usr/bin/env python3
"""
Raspberry Pi 5 Auto-Start Script for Disaster Surveillance System
Handles GPIO signals, system startup, and automatic execution
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
import json
from datetime import datetime

# Add your project directory to Python path
PROJECT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_DIR))

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO not available - running in simulation mode")
    GPIO = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/disaster_surveillance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DisasterSurveillanceStarter:
    def __init__(self):
        self.running = False
        self.surveillance_process = None
        self.thermal_process = None
        self.start_signal_pin = 18  # GPIO 18 for start signal
        self.stop_signal_pin = 19   # GPIO 19 for stop signal
        self.status_led_pin = 21    # GPIO 21 for status LED
        self.emergency_pin = 20     # GPIO 20 for emergency stop
        
        # System status
        self.system_ready = False
        self.last_health_check = time.time()
        
        # Configuration
        self.config = {
            "surveillance_mode": "disaster",  # or "thermal"
            "auto_start_delay": 5,  # seconds
            "surveillance_duration": 1080,  # 18 minutes
            "video_source": 0,
            "enable_thermal": True,
            "enable_gpu": True,
            "log_level": "INFO"
        }
        
        self.load_config()
        self.setup_gpio()
        
    def load_config(self):
        """Load configuration from file"""
        config_file = PROJECT_DIR / "surveillance_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save current configuration"""
        config_file = PROJECT_DIR / "surveillance_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def setup_gpio(self):
        """Initialize GPIO pins"""
        if GPIO is None:
            logger.warning("GPIO not available - running without hardware control")
            return
            
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Input pins with pull-up resistors
            GPIO.setup(self.start_signal_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.stop_signal_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.emergency_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Output pin for status LED
            GPIO.setup(self.status_led_pin, GPIO.OUT)
            
            # Setup interrupt callbacks
            GPIO.add_event_detect(self.start_signal_pin, GPIO.FALLING, 
                                callback=self.start_signal_callback, bouncetime=1000)
            GPIO.add_event_detect(self.stop_signal_pin, GPIO.FALLING, 
                                callback=self.stop_signal_callback, bouncetime=1000)
            GPIO.add_event_detect(self.emergency_pin, GPIO.FALLING, 
                                callback=self.emergency_callback, bouncetime=500)
            
            logger.info("GPIO initialized successfully")
            self.system_ready = True
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
    
    def start_signal_callback(self, channel):
        """Handle start signal from GPIO"""
        logger.info("START signal received via GPIO")
        threading.Thread(target=self.start_surveillance, daemon=True).start()
    
    def stop_signal_callback(self, channel):
        """Handle stop signal from GPIO"""
        logger.info("STOP signal received via GPIO")
        self.stop_surveillance()
    
    def emergency_callback(self, channel):
        """Handle emergency stop"""
        logger.critical("EMERGENCY STOP activated!")
        self.emergency_stop()
    
    def blink_status_led(self, pattern="normal"):
        """Control status LED patterns"""
        if GPIO is None:
            return
            
        patterns = {
            "normal": [0.5, 0.5],      # Slow blink - ready
            "starting": [0.1, 0.1],    # Fast blink - starting
            "running": [2.0, 0.2],     # Long on, short off - running
            "error": [0.05, 0.05],     # Very fast blink - error
            "emergency": [0.02, 0.02]  # Strobe - emergency
        }
        
        if pattern not in patterns:
            pattern = "normal"
            
        on_time, off_time = patterns[pattern]
        
        def blink_loop():
            while self.running or pattern == "emergency":
                try:
                    GPIO.output(self.status_led_pin, GPIO.HIGH)
                    time.sleep(on_time)
                    GPIO.output(self.status_led_pin, GPIO.LOW)
                    time.sleep(off_time)
                except:
                    break
        
        threading.Thread(target=blink_loop, daemon=True).start()
    
    def check_system_health(self):
        """Check system health and requirements"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "project_dir": str(PROJECT_DIR),
            "gpu_available": False,
            "thermal_camera": False,
            "disk_space_mb": 0,
            "memory_mb": 0,
            "cpu_temp": 0.0
        }
        
        try:
            # Check GPU availability
            import torch
            health_status["gpu_available"] = torch.cuda.is_available()
        except:
            pass
        
        try:
            # Check thermal camera
            import board
            import busio
            import adafruit_mlx90640
            i2c = busio.I2C(board.SCL, board.SDA)
            mlx = adafruit_mlx90640.MLX90640(i2c)
            health_status["thermal_camera"] = True
        except:
            pass
        
        try:
            # System stats
            import psutil
            health_status["disk_space_mb"] = psutil.disk_usage('/').free // (1024*1024)
            health_status["memory_mb"] = psutil.virtual_memory().available // (1024*1024)
            
            # CPU temperature (Raspberry Pi specific)
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                health_status["cpu_temp"] = int(f.read()) / 1000.0
        except:
            pass
        
        self.last_health_check = time.time()
        logger.info(f"System health check: {health_status}")
        return health_status
    
    def start_surveillance(self):
        """Start the surveillance system"""
        if self.running:
            logger.warning("Surveillance already running")
            return
        
        logger.info("Starting surveillance system...")
        self.running = True
        self.blink_status_led("starting")
        
        try:
            # Pre-flight checks
            health = self.check_system_health()
            
            if health["disk_space_mb"] < 500:
                raise Exception("Insufficient disk space (< 500MB)")
            
            if health["memory_mb"] < 512:
                raise Exception("Insufficient memory (< 512MB)")
            
            if health["cpu_temp"] > 80.0:
                raise Exception(f"CPU temperature too high: {health['cpu_temp']:.1f}°C")
            
            # Start appropriate surveillance mode
            if self.config["surveillance_mode"] == "thermal":
                self.start_thermal_surveillance()
            else:
                self.start_disaster_surveillance()
            
            self.blink_status_led("running")
            logger.info("Surveillance system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start surveillance: {e}")
            self.running = False
            self.blink_status_led("error")
    
    def start_disaster_surveillance(self):
        """Start disaster surveillance system"""
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "disaster_surveillance.py"),
            "--duration", str(self.config["surveillance_duration"]),
            "--video-source", str(self.config["video_source"])
        ]
        
        if self.config["enable_gpu"]:
            cmd.append("--gpu")
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        self.surveillance_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_DIR
        )
        
        # Monitor process
        threading.Thread(target=self.monitor_surveillance, daemon=True).start()
    
    def start_thermal_surveillance(self):
        """Start thermal surveillance system"""
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "thermal_human_detection.py")
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        self.thermal_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=PROJECT_DIR
        )
        
        # Monitor process
        threading.Thread(target=self.monitor_thermal, daemon=True).start()
    
    def monitor_surveillance(self):
        """Monitor disaster surveillance process"""
        if not self.surveillance_process:
            return
        
        try:
            stdout, stderr = self.surveillance_process.communicate(timeout=self.config["surveillance_duration"] + 120)
            
            if stdout:
                logger.info(f"Surveillance output: {stdout.decode()}")
            if stderr:
                logger.error(f"Surveillance error: {stderr.decode()}")
                
            return_code = self.surveillance_process.returncode
            logger.info(f"Surveillance completed with return code: {return_code}")
            
        except subprocess.TimeoutExpired:
            logger.warning("Surveillance process timeout - terminating")
            self.surveillance_process.terminate()
        except Exception as e:
            logger.error(f"Surveillance monitoring error: {e}")
        finally:
            self.running = False
            self.surveillance_process = None
            self.blink_status_led("normal")
    
    def monitor_thermal(self):
        """Monitor thermal surveillance process"""
        if not self.thermal_process:
            return
        
        try:
            # Thermal runs indefinitely, just monitor health
            while self.running and self.thermal_process.poll() is None:
                time.sleep(10)
                if time.time() - self.last_health_check > 300:  # 5 minutes
                    self.check_system_health()
            
            if self.thermal_process.returncode is not None:
                logger.info(f"Thermal surveillance ended with code: {self.thermal_process.returncode}")
                
        except Exception as e:
            logger.error(f"Thermal monitoring error: {e}")
        finally:
            self.running = False
            self.thermal_process = None
            self.blink_status_led("normal")
    
    def stop_surveillance(self):
        """Stop surveillance system"""
        if not self.running:
            logger.info("No surveillance running to stop")
            return
        
        logger.info("Stopping surveillance system...")
        self.running = False
        
        # Terminate processes
        for process in [self.surveillance_process, self.thermal_process]:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    logger.error(f"Error stopping process: {e}")
        
        self.surveillance_process = None
        self.thermal_process = None
        self.blink_status_led("normal")
        logger.info("Surveillance stopped")
    
    def emergency_stop(self):
        """Emergency stop all operations"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.blink_status_led("emergency")
        
        # Kill all surveillance processes immediately
        try:
            subprocess.run(["pkill", "-f", "disaster_surveillance"], timeout=5)
            subprocess.run(["pkill", "-f", "thermal_human_detection"], timeout=5)
        except:
            pass
        
        self.running = False
        self.surveillance_process = None
        self.thermal_process = None
        
        # Log emergency stop
        with open("/var/log/emergency_stop.log", "a") as f:
            f.write(f"{datetime.now().isoformat()} - Emergency stop activated\n")
    
    def handle_system_signals(self):
        """Handle system shutdown signals"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} - shutting down gracefully")
            self.stop_surveillance()
            if GPIO:
                GPIO.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run(self):
        """Main run loop"""
        logger.info("Disaster Surveillance Auto-Start System initialized")
        logger.info(f"Project directory: {PROJECT_DIR}")
        logger.info(f"Configuration: {self.config}")
        
        self.handle_system_signals()
        
        if self.system_ready:
            self.blink_status_led("normal")
            logger.info("System ready - waiting for start signal")
            logger.info("GPIO Controls:")
            logger.info(f"  Start: GPIO {self.start_signal_pin} (pull to ground)")
            logger.info(f"  Stop: GPIO {self.stop_signal_pin} (pull to ground)")
            logger.info(f"  Emergency: GPIO {self.emergency_pin} (pull to ground)")
            logger.info(f"  Status LED: GPIO {self.status_led_pin}")
        else:
            logger.error("System not ready - check GPIO and hardware")
            return
        
        try:
            # Main loop
            while True:
                time.sleep(1)
                
                # Periodic health check
                if time.time() - self.last_health_check > 600:  # 10 minutes
                    self.check_system_health()
                
                # Check for file-based signals
                signal_file = PROJECT_DIR / "START_SURVEILLANCE"
                if signal_file.exists():
                    signal_file.unlink()
                    logger.info("START signal received via file")
                    threading.Thread(target=self.start_surveillance, daemon=True).start()
                
                stop_file = PROJECT_DIR / "STOP_SURVEILLANCE"
                if stop_file.exists():
                    stop_file.unlink()
                    logger.info("STOP signal received via file")
                    self.stop_surveillance()
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop_surveillance()
            if GPIO:
                GPIO.cleanup()
            logger.info("Auto-start system shutdown complete")


def main():
    """Main entry point"""
    print("🚀 Disaster Surveillance Auto-Start System")
    print("=" * 50)
    
    # Check if running as root (required for GPIO)
    if os.geteuid() != 0 and GPIO is not None:
        print("⚠️  Warning: GPIO access requires root privileges")
        print("   Run with: sudo python3 disaster_surveillance_startup.py")
    
    # Create and run the starter
    starter = DisasterSurveillanceStarter()
    starter.run()


if __name__ == "__main__":
    main()
