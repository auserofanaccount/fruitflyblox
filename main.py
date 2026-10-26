from pynput.keyboard import Controller, Key
import pandas as pd
import time
import numpy as np
import cv2
from mss import MSS
from scipy.sparse import lil_matrix, csr_matrix
import os
import ctypes
import win32gui
import win32con
import subprocess

keyboard = Controller()
sct = MSS()

# Windows API for window management
def find_roblox_window():
    """Find the RobloxPlayerBeta.exe window"""
    def enum_windows(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Roblox" in title or "RobloxPlayerBeta" in title:
                lParam.append(hwnd)
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows, windows)
    return windows[0] if windows else None

def focus_roblox_window(hwnd):
    """Bring Roblox window to focus"""
    try:
        # Restore window if minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # Set focus
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        print(f"  Could not focus Roblox window: {e}")
        return False

def is_roblox_focused(hwnd):
    """Check if Roblox window is currently focused"""
    if hwnd is None:
        return False
    try:
        focused_hwnd = win32gui.GetForegroundWindow()
        return focused_hwnd == hwnd
    except:
        return False

print("loading REAL fruit fly VISUAL-MOTOR PATHWAY... 🧠👁️")

# Find and focus Roblox window
print("\n Looking for RobloxPlayerBeta.exe...")
roblox_hwnd = find_roblox_window()

if roblox_hwnd:
    print(f"    Found Roblox window!")
    focus_roblox_window(roblox_hwnd)
    print(f"    Roblox window focused!")
else:
    print(f"     WARNING: RobloxPlayerBeta.exe not found!")
    print(f"   Please open Roblox before running this script")
    print(f"   Continuing anyway - will send input to focused window")
    roblox_hwnd = None

monitor = sct.monitors[1]
w_box, h_box = 400, 300
left_pos = (monitor["width"] // 2) - (w_box // 2)
top_pos = (monitor["height"] // 2) - (h_box // 2)
bounding_box = {"top": top_pos, "left": left_pos, "width": w_box, "height": h_box}

# STEP 1: Load Classification Data
print("\n Building visual-motor pathways from real fly connectome...\n")

print(" Loading neuron classifications...")
classification = pd.read_csv("classification.csv")
print(f"    Loaded {len(classification)} neurons with classifications")

# Identify neuron types by flow/class
visual_sensory = classification[classification['super_class'] == 'optic']
central_processing = classification[classification['super_class'] == 'central']
motor_descending = classification[classification['super_class'] == 'descending']
motor_output = classification[classification['class'] == 'brain_motor_neuron']

print(f"    Visual (Optic Lobe): {len(visual_sensory)} neurons")
print(f"    Central (Processing): {len(central_processing)} neurons")
print(f"    Descending (Motor): {len(motor_descending)} neurons")
print(f"    Motor Output: {len(motor_output)} neurons")

# Build neuron IDs for each pathway
visual_ids = set(visual_sensory['root_id'].values)
central_ids = set(central_processing['root_id'].values)
descending_ids = set(motor_descending['root_id'].values)
motor_ids = set(motor_output['root_id'].values)

print(f"\n Loading full connectome...")
print("   This might take a moment (261MB file)...")

connections = pd.read_csv("connections_princeton.csv")
print(f"    Loaded {len(connections)} synaptic connections")

# Build neuron map for ALL neurons
all_neuron_ids = pd.concat([connections['pre_root_id'], connections['post_root_id']]).unique()
neuron_map = {nid: idx for idx, nid in enumerate(all_neuron_ids)}
n_neurons = len(all_neuron_ids)

print(f"   Total unique neurons: {n_neurons}")

# STEP 2: Extract pathway-specific connections
print(f"\n Extracting visual→central→motor pathways...")

# Visual to Central connections
visual_to_central = connections[
    (connections['pre_root_id'].isin(visual_ids)) & 
    (connections['post_root_id'].isin(central_ids))
]

# Central to Descending connections
central_to_descending = connections[
    (connections['pre_root_id'].isin(central_ids)) & 
    (connections['post_root_id'].isin(descending_ids))
]

# Descending to Motor connections
descending_to_motor = connections[
    (connections['pre_root_id'].isin(descending_ids)) & 
    (connections['post_root_id'].isin(motor_ids))
]

print(f"    Visual→Central: {len(visual_to_central)} connections")
print(f"    Central→Descending: {len(central_to_descending)} connections")
print(f"    Descending→Motor: {len(descending_to_motor)} connections")

# STEP 3: Build SPARSE synaptic weight matrix
print(f"\n Building visual-motor circuit matrix (sparse)...")

# Use LIL (list-of-lists) format for efficient construction
synaptic_weights = lil_matrix((n_neurons, n_neurons), dtype=np.float32)

print("   Building visual→central connections...")
# Add pathway connections with weights
for _, row in visual_to_central.iterrows():
    if row['pre_root_id'] in neuron_map and row['post_root_id'] in neuron_map:
        pre_idx = neuron_map[row['pre_root_id']]
        post_idx = neuron_map[row['post_root_id']]
        weight = row['syn_count'] / 100.0
        synaptic_weights[pre_idx, post_idx] = weight

print("   Building central→descending connections...")
for _, row in central_to_descending.iterrows():
    if row['pre_root_id'] in neuron_map and row['post_root_id'] in neuron_map:
        pre_idx = neuron_map[row['pre_root_id']]
        post_idx = neuron_map[row['post_root_id']]
        weight = row['syn_count'] / 100.0
        synaptic_weights[pre_idx, post_idx] = weight

print("   Building descending→motor connections...")
for _, row in descending_to_motor.iterrows():
    if row['pre_root_id'] in neuron_map and row['post_root_id'] in neuron_map:
        pre_idx = neuron_map[row['pre_root_id']]
        post_idx = neuron_map[row['post_root_id']]
        weight = row['syn_count'] / 100.0
        synaptic_weights[pre_idx, post_idx] = weight

# Convert to CSR (compressed sparse row) format for fast multiplication
synaptic_weights = synaptic_weights.tocsr()

print(f"    Sparse circuit matrix built: {synaptic_weights.nnz} active connections")
print(f"    Memory usage: ~{synaptic_weights.data.nbytes / 1e6:.1f} MB (vs {n_neurons**2 * 8 / 1e9:.0f} GB dense!)")

# Create mapping from root_id to neuron index
visual_neuron_indices = np.array([neuron_map[nid] for nid in visual_ids if nid in neuron_map], dtype=np.int32)
central_neuron_indices = np.array([neuron_map[nid] for nid in central_ids if nid in neuron_map], dtype=np.int32)
descending_neuron_indices = np.array([neuron_map[nid] for nid in descending_ids if nid in neuron_map], dtype=np.int32)
motor_neuron_indices = np.array([neuron_map[nid] for nid in motor_ids if nid in neuron_map], dtype=np.int32)

print(f"\n   Visual neurons in circuit: {len(visual_neuron_indices)}")
print(f"   Central neurons in circuit: {len(central_neuron_indices)}")
print(f"   Descending neurons in circuit: {len(descending_neuron_indices)}")
print(f"   Motor neurons in circuit: {len(motor_neuron_indices)}")

# FIXED: Define number of features to extract
n_features = min(len(visual_neuron_indices), 20)  # Use up to 20 features

def extract_visual_features(img):
    """Extract visual features from Roblox screenshot"""
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    region_h = h // 4
    region_w = w // 4
    
    features = []
    
    # Spatial visual features (like visual receptive fields) - 16 features
    for i in range(4):
        for j in range(4):
            region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            brightness = np.mean(region) / 255.0
            features.append(brightness)
    
    # Edge features (motion/structure detection) - 4 features
    for i in range(2):
        for j in range(2):
            region = gray[i*region_h*2:(i+1)*region_h*2, j*region_w*2:(j+1)*region_w*2]
            edges = cv2.Canny(region, 50, 150)
            edge_density = np.sum(edges > 0) / max(1, (region_h * region_w * 4))
            features.append(edge_density)
    
    # Color (Roblox player detection) - 1 feature
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    lower_roblox = np.array([5, 40, 60])
    upper_roblox = np.array([25, 200, 255])
    mask = cv2.inRange(hsv, lower_roblox, upper_roblox)
    roblox_density = np.sum(mask > 0) / max(1, (h * w))
    features.append(roblox_density)
    
    # Use first n_features
    features_array = np.array(features[:n_features], dtype=np.float32)
    
    return features_array

def propagate_through_pathway(visual_input):
    """
    Propagate visual input through visual→central→motor pathway
    Using sparse matrix multiplication for efficiency!
    """
    # Initialize neuron activations
    neuron_activations = np.zeros(n_neurons, dtype=np.float32)
    
    # Visual neurons receive input
    if len(visual_neuron_indices) > 0:
        neuron_activations[visual_neuron_indices[:len(visual_input)]] = visual_input
    
    # Propagate through circuit (multiple time steps)
    for time_step in range(8):
        # Sparse matrix multiplication: O(nnz) instead of O(n^2)!
        synaptic_input = synaptic_weights.T.dot(neuron_activations)
        
        # Spiking nonlinearity (rectified linear) - amplified to be more decisive
        new_activations = np.clip(synaptic_input * 0.5, 0, 1.0).astype(np.float32)
        
        # Temporal dynamics (leak/memory)
        neuron_activations = new_activations * 0.85 + neuron_activations * 0.15
    
    # Read out motor neuron activity with amplification
    if len(motor_neuron_indices) > 0:
        motor_output = np.clip(neuron_activations[motor_neuron_indices] * 2.0, 0, 1.0)
    else:
        motor_output = np.array([0], dtype=np.float32)
    
    return motor_output

print("\n" + "="*70)
print("="*70)
print(f"Visual→Central→Descending→Motor pathways wired and ready!")
print("="*70)
print("\n JUMP BEHAVIOR ENABLED:")
print("   - Fly jumps when neural excitation > 0.5")
print("   - Jump height proportional to motor neuron activation")
print("   - Can combine with forward/turning movements!")
if roblox_hwnd:
    print("\n ROBLOX WINDOW FOCUS:")
    print("   Keyboard input ONLY goes to RobloxPlayerBeta.exe")
    print("   Window will auto-focus if you click elsewhere")
else:
    print("\n  WINDOW FOCUS DISABLED:")
    print("   Input will go to whichever window is focused")
    print("   Start Roblox to enable safe mode!")
print("="*70 + "\n")

frame_count = 0
processing_times = []

try:
    while True:
        frame_start = time.time()
        
        try:
            img = np.array(sct.grab(bounding_box))
        except Exception as e:
            print(f"Screenshot error: {e}")
            time.sleep(0.1)
            continue
        
        # Extract visual features (what the fly sees)
        visual_input = extract_visual_features(img)
        
        # Process through visual-motor pathway
        motor_output = propagate_through_pathway(visual_input)
        
        # Decode motor output to movement commands
        if len(motor_output) >= 3:
            forward_drive = np.mean(motor_output[:len(motor_output)//3])
            left_drive = np.mean(motor_output[len(motor_output)//3:2*len(motor_output)//3])
            right_drive = np.mean(motor_output[2*len(motor_output)//3:])
        else:
            forward_drive = np.mean(motor_output) if len(motor_output) > 0 else 0
            left_drive = 0
            right_drive = 0
        
        # Add noise for stochasticity (real neurons are noisy!)
        forward_drive += np.random.normal(0, 0.05)
        left_drive += np.random.normal(0, 0.05)
        right_drive += np.random.normal(0, 0.05)
        
        # Clip to valid range
        forward_drive = np.clip(forward_drive, 0, 1.0)
        left_drive = np.clip(left_drive, 0, 1.0)
        right_drive = np.clip(right_drive, 0, 1.0)
        
        processing_time = time.time() - frame_start
        processing_times.append(processing_time)
        if len(processing_times) > 30:
            processing_times.pop(0)
        
        # Print brain activity
        avg_visual_activation = np.mean(visual_input)
        print(f"👁️ Visual:{avg_visual_activation:.3f} | Fwd:{forward_drive:.2f} L:{left_drive:.2f} R:{right_drive:.2f} | {np.mean(processing_times)*1000:.1f}ms", end=" ")
        
        # Determine dominant motor command
        max_drive = max(forward_drive, left_drive, right_drive)
        
        # JUMP BEHAVIOR: Calculate total neural excitation
        # Real fruit flies jump when startled/excited - use sum of all motor neurons
        total_excitation = np.sum(motor_output)
        jump_threshold = 0.5  # Fly jumps when sufficiently excited
        jump_happened = False
        
        try:
            # Check if Roblox window is still focused
            if roblox_hwnd and not is_roblox_focused(roblox_hwnd):
                focus_roblox_window(roblox_hwnd)
            
            # JUMP BEHAVIOR: High neural excitation triggers jumping
            if total_excitation > jump_threshold:
                jump_happened = True
                # Proportional jump height/duration based on excitation
                jump_intensity = min(1.0, total_excitation / 2.0)  # Normalize to [0, 1]
                jump_height_factor = 0.3 + (jump_intensity * 0.4)  # 0.3-0.7s hold time
                
                keyboard.press(Key.space)
                time.sleep(jump_height_factor)
                keyboard.release(Key.space)
                jump_label = "MASSIVE JUMP!" if jump_intensity > 0.7 else "🦗 JUMP!"
                print(f"| {jump_label} (excitation:{jump_intensity:.2f})", end="")
            
            if forward_drive > max(left_drive, right_drive) and forward_drive > 0.1:
                # Forward movement
                keyboard.press('w')  # 'w' is a regular character key
                time.sleep(0.3)
                keyboard.release('w')
                if not jump_happened:
                    print(f"| FORWARD! (0.3s)")
                else:
                    print(f" + FORWARD!")
                
            elif left_drive > right_drive and left_drive > 0.1:
                # Left camera turn
                # 90 degree turn = 0.6 seconds
                # 45 degree turn = 0.3 seconds (linear relationship)
                turn_time = left_drive * 0.6  # Maps [0, 1] to [0, 0.6] seconds
                degrees = (turn_time / 0.6) * 90  # Convert time to degrees
                
                keyboard.press(Key.left)  # Left arrow key for camera turn
                time.sleep(turn_time)
                keyboard.release(Key.left)
                if not jump_happened:
                    print(f"| LEFT TURN ({degrees:.0f}°, {turn_time:.2f}s)")
                else:
                    print(f" + LEFT TURN ({degrees:.0f}°)")
                
            elif right_drive > 0.1:
                # Right camera turn
                turn_time = right_drive * 0.6  # Maps [0, 1] to [0, 0.6] seconds
                degrees = (turn_time / 0.6) * 90  # Convert time to degrees
                
                keyboard.press(Key.right)  # Right arrow key for camera turn
                time.sleep(turn_time)
                keyboard.release(Key.right)
                if not jump_happened:
                    print(f"| RIGHT TURN ({degrees:.0f}°, {turn_time:.2f}s)")
                else:
                    print(f" + RIGHT TURN ({degrees:.0f}°)")
                
            else:
                if not jump_happened:
                    print(f"| Thinking...")
                else:
                    print()  # Just jumped, no extra text
            
        except Exception as e:
            print(f"|  Keyboard error: {e}")
        
        frame_count += 1
        time.sleep(0.05)

except KeyboardInterrupt:
    print(f"Total frames processed: {frame_count}")
