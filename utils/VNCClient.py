import io
import os
from PIL import Image
import time
import uuid
# import copy
import paramiko
from vncdotool import api
from vncdotool.client import KEYMAP
from sshtunnel import SSHTunnelForwarder
from utils.log import print_message
from utils.vmware_utils import VMwareTools
import subprocess


if not hasattr(paramiko, "DSSKey"):
    # Paramiko 4 removed DSSKey, but sshtunnel 0.4.0 still references it.
    class _UnsupportedDSSKey(paramiko.PKey):
        @classmethod
        def from_private_key_file(cls, *args, **kwargs):
            raise paramiko.SSHException("DSA keys are unsupported in paramiko>=4")

    paramiko.DSSKey = _UnsupportedDSSKey

class AttributeContainer:
    pass

class VNCClient:
    def __init__(self, host, username, password):
        self.vnc_host = host
        self.guest_username = username
        self.guest_password = password
        self.client = None

    def connect(self):
        """Connect to the VNC server."""
        self.client = api.connect(self.vnc_host, username=self.guest_username, password=self.guest_password)

    def capture_screenshot(self):
        """Capture a screenshot and return it as a PIL Image."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        fp = io.BytesIO()
        fp.name = 'screenshot.png'
        self.client.captureScreen(fp)
        fp.seek(0)
        image = Image.open(fp)
        del fp
        return image

    def left_click(self):
        """Perform a left mouse click."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.mouseDown(1)
        self.client.mouseUp(1)

    def middle_click(self):
        """Perform a middle mouse click."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.mouseDown(2)
        self.client.mouseUp(2)

    def right_click(self):
        """Perform a right mouse click."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.mouseDown(3)
        self.client.mouseUp(3)

    def move_to(self, x, y):
        """Move the mouse to the coordinates (x, y)."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.mouseMove(x, y)

    def key_press(self, key):
        """Press a key on the keyboard."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.keyPress(key)

    def type_text(self, text):
        """Type a string of text."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        for char in text:
            self.client.keyPress(char)

    def disconnect(self):
        """Disconnect from the VNC server."""
        if self.client is None:
            raise ConnectionError("VNC client is not connected.")
        
        self.client.disconnect()
        self.client = None


import time

class VNCClient_SSH:
    def __init__(self, guest_username, guest_password, ssh_host, ssh_pkey, retry_attempts=3, retry_delay=5, action_interval_seconds=1, vmx_path=None, vnc_connection_timeout=600):
        self.guest_username = guest_username
        self.guest_password = guest_password
        self.ssh_host = ssh_host
        self.ssh_pkey = ssh_pkey
        self.tunnel = None
        self.client = None
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.action_interval_seconds = action_interval_seconds
        self.vmx_path = vmx_path
        self.vnc_connection_timeout = vnc_connection_timeout

        if self.vmx_path is not None:
            self.vmware_tools = VMwareTools(
                guest_username = guest_username,
                guest_password = guest_password,
                ssh_host = ssh_host,
                ssh_pkey = ssh_pkey,
                vmx_path = vmx_path
            )

    def check_ssh_connectivity(self):
        """Check if SSH connection can be established. Returns True if successful, False otherwise."""
        try:
            ssh_command = self._ssh_base_command() + ['exit 0']
            result = subprocess.run(
                ssh_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _ssh_base_command(self):
        return [
            'ssh',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'ConnectTimeout=10',
            '-i', self.ssh_pkey,
            f'{self.guest_username}@{self.ssh_host}',
        ]

    @staticmethod
    def _is_transient_ssh_error(message: str) -> bool:
        if not isinstance(message, str):
            return False
        transient_patterns = (
            'Connection reset by peer',
            'Connection closed by',
            'kex_exchange_identification',
            'Connection timed out',
            'Operation timed out',
            'ssh: connect to host',
            'Broken pipe',
        )
        return any(pattern in message for pattern in transient_patterns)
        
    def run_ssh_command(self, command: str) -> str:
        ssh_command = self._ssh_base_command() + [command]
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                output = subprocess.check_output(
                    ssh_command,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).strip()
                return True, output
            except subprocess.CalledProcessError as e:
                output = (e.output or "").strip()
                last_error = output or str(e)
                if attempt < self.retry_attempts and self._is_transient_ssh_error(last_error):
                    time.sleep(self.retry_delay)
                    continue
                return False, e
            except Exception as e: # subprocess.CalledProcessError
                last_error = str(e)
                if attempt < self.retry_attempts and self._is_transient_ssh_error(last_error):
                    time.sleep(self.retry_delay)
                    continue
                return False, e
        return False, RuntimeError(last_error or "Unknown SSH command failure")

    def connect(self):
        """Connect to the VNC server, with retries on failure."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                print_message(title = 'VNC Client', content = 'Connecting')
                self.tunnel = SSHTunnelForwarder(
                    (self.ssh_host, 22),
                    ssh_username=self.guest_username,
                    ssh_pkey=self.ssh_pkey,
                    remote_bind_address=('localhost', 5900)
                )
                self.tunnel.start()
                self.client = api.connect(f'localhost::{self.tunnel.local_bind_port}',
                                          username=self.guest_username,
                                          password=self.guest_password,
                                          timeout=self.vnc_connection_timeout)
                return
            except Exception as e:
                print_message(title = 'VNC Client', content = f"Connection attempt {attempt} failed: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError("Failed to connect to VNC server after multiple attempts.")

    def capture_screenshot(self):
        """Capture a screenshot and return it as a PIL Image."""
        image = None
        if self.vmx_path is None:
            # Capture a screenshot using VNC
            self._ensure_connection()
            fp = io.BytesIO()
            fp.name = 'screenshot.png'
            self.client.captureScreen(fp)
            fp.seek(0)
            image = Image.open(fp)
            del fp
        else:
            # Capture a screenshot using VMware (VNC screenshot could be slow on VMware machines)
            cache_image_path = f'./{uuid.uuid4().hex}.png'
            max_trials = 5

            for _ in range(max_trials):
                try:
                    # Reload vmware tools
                    if not self.vmware_tools.reload_vmware_tools():
                        print(f'Error reloading VMware Tools. Screen capture failed.', title = 'Error')
                        continue

                    screen_capture_command = f'vmrun -gu {self.guest_username} -gp {self.guest_password} captureScreen "{self.vmx_path}" {cache_image_path}'

                    # Capture screenshot to cache path
                    screen_capture_result = subprocess.run(screen_capture_command, shell=True, text=True, capture_output=True, encoding="utf-8", env=os.environ.copy())

                    # Debug printing
                    if screen_capture_result.returncode != 0:
                        print(f'Screen capture failed.\nSTDOUT: {screen_capture_result.stdout}\nSTDERR: {screen_capture_result.stderr}', title = 'Error')

                    # Read in from cache path and clear cache
                    image = Image.open(cache_image_path)
                    os.remove(cache_image_path)

                    # Update resolution
                    self.client.screen = AttributeContainer()
                    self.client.screen.width, self.client.screen.height = image.size
                except Exception:
                    continue
            
        if image == None:
            raise RuntimeError(f'Screen capture failed after maximum trials')
        return image
    
    def mouse_down(self, button):
        """Press and hold a specified mouse button."""
        self._ensure_connection()
        if button.lower() == "left":
            self.client.mouseDown(1)
        elif button.lower() == "middle":
            self.client.mouseDown(2)
        elif button.lower() == "right":
            self.client.mouseDown(3)

    def mouse_up(self, button):
        """Release a specified mouse button."""
        self._ensure_connection()
        if button.lower() == "left":
            self.client.mouseUp(1)
        elif button.lower() == "middle":
            self.client.mouseUp(2)
        elif button.lower() == "right":
            self.client.mouseUp(3)

    def left_click(self):
        """Perform a left mouse click."""
        self._ensure_connection()
        self.client.mouseDown(1)
        self.client.mouseUp(1)

    def middle_click(self):
        """Perform a middle mouse click."""
        self._ensure_connection()
        self.client.mouseDown(2)
        self.client.mouseUp(2)

    def right_click(self):
        """Perform a right mouse click."""
        self._ensure_connection()
        self.client.mouseDown(3)
        self.client.mouseUp(3)

    def double_click(self):
        """Perform a double left mouse click."""
        self._ensure_connection()
        self.client.mouseDown(1)
        self.client.mouseUp(1)
        self.client.mouseDown(1)
        self.client.mouseUp(1)

    def triple_click(self):
        """Perform a triple left mouse click."""
        self._ensure_connection()
        self.client.mouseDown(1)
        self.client.mouseUp(1)
        self.client.mouseDown(1)
        self.client.mouseUp(1)
        self.client.mouseDown(1)
        self.client.mouseUp(1)

    def drag_to(self, x, y):
        """Perform a drag action by holding down the left mouse button, moving to (x, y), then releasing."""
        self._ensure_connection()
        # Ensure client screen width/height is not None
        if self.client.screen is None:
            _ = self.capture_screenshot()
        # Calculate pixel coordinates as in move_to:
        x_scaled = int(round(x * (self.client.screen.width - 1)))
        y_scaled = int(round(y * (self.client.screen.height - 1)))
        self.client.mouseDown(1)
        self.client.mouseMove(x_scaled, y_scaled)
        self.client.mouseUp(1)

    def scroll_down(self, amount, by_pixel=False):
        """Perform a scrolling down. 
        
        If `by_pixel`, `amount` is the number of pixels to scroll down. Should be non-negative integer. Otherwise, `amount` is the proportion of pixels to scroll down, should be a float value between 0 and 1."""
        self._ensure_connection()

        if by_pixel:
            scaled_amount = amount
        else:
            scaled_amount = int(round(amount * self.client.screen.height))
        scaled_amount = max(0, scaled_amount)
        for _ in range(scaled_amount):
            self.client.mouseDown(5)
            self.client.mouseUp(5)

    def scroll_up(self, amount, by_pixel=False):
        """Perform a mouse scrolling up. 
        
        If `by_pixel`, `amount` is the number of pixels to scroll up. Should be non-negative integer. Otherwise, `amount` is the proportion of pixels to scroll up, should be a float value between 0 and 1."""
        self._ensure_connection()

        if by_pixel:
            scaled_amount = amount
        else:
            scaled_amount = int(round(amount * self.client.screen.height))
        scaled_amount = max(0, scaled_amount)
        for _ in range(scaled_amount):
            self.client.mouseDown(4)
            self.client.mouseUp(4)

    def scroll_left(self, amount, by_pixel=False):
        """Perform a mouse scrolling up. 
        
        If `by_pixel`, `amount` is the number of pixels to scroll left. Should be non-negative integer. Otherwise, `amount` is the proportion of pixels to scroll left, should be a float value between 0 and 1."""
        self._ensure_connection()

        if by_pixel:
            scaled_amount = amount
        else:
            scaled_amount = int(round(amount * self.client.screen.width))
        scaled_amount = max(0, scaled_amount)
        for _ in range(scaled_amount):
            self.client.mouseDown(6)
            self.client.mouseUp(6)

    def scroll_right(self, amount, by_pixel=False):
        """Perform a mouse scrolling up. 
        
        If `by_pixel`, `amount` is the number of pixels to scroll right. Should be non-negative integer. Otherwise, `amount` is the proportion of pixels to scroll right, should be a float value between 0 and 1."""
        self._ensure_connection()

        if by_pixel:
            scaled_amount = amount
        else:
            scaled_amount = int(round(amount * self.client.screen.width))
        scaled_amount = max(0, scaled_amount)
        for _ in range(scaled_amount):
            self.client.mouseDown(7)
            self.client.mouseUp(7)

    def move_to(self, x, y):
        """Move the mouse to the normalised coordinates (x, y).
        
        `(x, y)` should be float values between 0 and 1."""
        self._ensure_connection()

        if self.client.screen is None:
            _ = self.capture_screenshot()

        x_scaled = int(round(x * (self.client.screen.width - 1)))
        y_scaled = int(round(y * (self.client.screen.height - 1)))
        
        x_scaled = max(0, min(self.client.screen.width, x_scaled))
        y_scaled = max(0, min(self.client.screen.height, y_scaled))

        self.client.mouseMove(x_scaled, y_scaled)

    def move_to_pixel(self, x, y):
        """Move the mouse to the pixel coordinates (x, y)."""       
        self._ensure_connection()
        self.client.mouseMove(x, y)

    def key_press(self, key):
        """Press a key on the keyboard.
        
        Keys available: single ASCII characters, ctrl, command, option, backspace, tab, enter, esc, del, left, up, right, down"""
        key = self._filter_key(key)
        if key is None:
            return
        self._ensure_connection()
        self.client.keyPress(key)

    def key_press_and_hold(self, key, duration_seconds: int):
        """Press a key or a key combination on the keyboard; hold for `duration_seconds` seconds before releasing.
        
        Keys available: single ASCII characters, ctrl, command, option, backspace, tab, enter, esc, del, left, up, right, down"""
        key = self._filter_key(key)
        if key is None:
            return
        self._ensure_connection()
        self.client.keyDown(key)
        time.sleep(duration_seconds)
        self.client.keyUp(key)

    def type_text(self, text):
        """Type a string of (ASCII characters only)."""
        text = self._filter_text(text)
        if text is None:
            return
        self._ensure_connection()
        for char in text:
            self.client.keyPress(char)
            time.sleep(0.1)

    def disconnect(self):
        """Disconnect from the VNC server."""
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None

    def _filter_text(self, text):
        if not isinstance(text, str):
            return None
        # Remove all non-ascii characters
        return ''.join(char for char in text if ord(char) < 128)

    def _filter_key(self, key):
        if not isinstance(key, str):
            return None
        if len(key) == 0:
            return None
        if len(key) == 1:
            return self._filter_text(key)
        
        # Split the string using `-` into a list of strings
        substrings = key.split('-')
        processed_substrings = []
        
        for substring in substrings:
            if len(substring) >= 2:
                substring = substring.lower()
                # Key mapping tested on keyboardtester.com; mapping is different
                if substring == 'option':
                    substring = 'meta'
                elif substring == 'command':
                    substring = 'alt'
                elif substring == 'cmd':
                    substring = 'alt'
                elif substring == 'backspace':
                    substring = 'bsp'
                if substring in KEYMAP:
                    processed_substrings.append(substring)
            elif len(substring) == 1:
                if ord(substring) < 128:  # Check if it's an ASCII character
                    processed_substrings.append(substring)
        
        # Reconnect and return all the sub-strings using `-`; if nothing is left, return None (don't return an empty string)
        result = '-'.join(processed_substrings)
        return result if result else None

    def _ensure_connection(self):
        """Ensure that the VNC client is connected, and attempt to reconnect if needed."""
        if self.client is None:
            self.connect()
