import subprocess
import os
from pathlib import Path
import tempfile

class FFmpegCapture:
    def __init__(self, ffmpeg_path=None):
        if ffmpeg_path is None:
            # Use absolute path from workspace root
            workspace_root = Path(__file__).parent.parent
            ffmpeg_path = str(workspace_root / "tools" / "ffmpeg" / "ffmpeg.exe")
        
        self.ffmpeg_path = ffmpeg_path
        
        # Check if FFmpeg exists
        if not os.path.exists(self.ffmpeg_path):
            print(f"[ERROR] FFmpeg not found at: {self.ffmpeg_path}")
            print("[INFO] Download FFmpeg from: https://ffmpeg.org/download.html")
            print("[INFO] Place it in: tools/ffmpeg/")
            raise FileNotFoundError(f"FFmpeg not found: {self.ffmpeg_path}")
    
    def capture_single(self, output_path=None, camera_id=1, timeout=5):
        """Capture single image using FFmpeg"""
        try:
            # Create temp file if no output path specified
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get available cameras using FFmpeg
            cameras = self._get_available_cameras()
            if not cameras:
                print("[ERROR] No cameras detected with FFmpeg")
                return None
            
            # Select camera (default to first, or by ID if specified)
            camera_index = min(camera_id, len(cameras) - 1) if camera_id < len(cameras) else 0
            camera_name = cameras[camera_index]
            
            print(f"[FFmpeg] Using camera: {camera_name}")
            
            # FFmpeg command for single frame capture
            cmd = [
                self.ffmpeg_path,
                "-f", "dshow",
                "-i", f"video={camera_name}",
                "-frames:v", "1",
                "-y",
                str(output_path)
            ]
            
            print(f"[FFmpeg] Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0 and output_path.exists():
                print(f"[FFmpeg] ✓ Captured: {output_path} ({output_path.stat().st_size} bytes)")
                return str(output_path)
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"[ERROR] FFmpeg capture failed: {error_msg[:200]}")
                return None
                
        except subprocess.TimeoutExpired:
            print("[ERROR] FFmpeg capture timeout")
            return None
        except Exception as e:
            print(f"[ERROR] FFmpeg error: {e}")
            return None
    
    def _get_available_cameras(self):
        """Get list of available cameras using FFmpeg"""
        try:
            print("[FFmpeg] Detecting cameras...")
            
            # Different command for Windows
            if os.name == 'nt':  # Windows
                cmd = [
                    self.ffmpeg_path,
                    "-list_devices", "true",
                    "-f", "dshow",
                    "-i", "dummy"
                ]
            else:  # Linux/Mac
                cmd = [
                    self.ffmpeg_path,
                    "-f", "avfoundation",
                    "-list_devices", "true",
                    "-i", ""
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            output = result.stdout + result.stderr
            cameras = []
            
            if os.name == 'nt':  # Windows parsing
                for line in output.split('\n'):
                    if '(video)' in line and '"' in line:
                        start = line.find('"')
                        end = line.find('"', start + 1)
                        if start != -1 and end != -1:
                            camera_name = line[start + 1:end]
                            cameras.append(camera_name)
                            print(f"  Found: {camera_name}")
            else:  # Linux/Mac parsing
                for line in output.split('\n'):
                    if 'video' in line.lower() and ']' in line:
                        # Parse avfoundation format
                        parts = line.split(']')
                        if len(parts) > 1:
                            camera_info = parts[1].strip()
                            cameras.append(camera_info)
                            print(f"  Found: {camera_info}")
            
            if not cameras:
                print("[WARNING] No cameras found. Trying alternative method...")
                # Try alternative listing
                cameras = ["Integrated Camera", "USB Camera", "Webcam", "Camera"]
            
            return cameras
            
        except Exception as e:
            print(f"[ERROR] Camera detection failed: {e}")
            return ["Integrated Camera", "USB Camera"]  # Fallback names
