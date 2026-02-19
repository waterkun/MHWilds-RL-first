"""Quick test: see what XInput values your controller sends."""
import ctypes, time

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', ctypes.c_ushort),
        ('bLeftTrigger', ctypes.c_ubyte),
        ('bRightTrigger', ctypes.c_ubyte),
        ('sThumbLX', ctypes.c_short),
        ('sThumbLY', ctypes.c_short),
        ('sThumbRX', ctypes.c_short),
        ('sThumbRY', ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ('dwPacketNumber', ctypes.c_ulong),
        ('Gamepad', XINPUT_GAMEPAD),
    ]

dll = ctypes.windll.xinput1_4

# First, detect which controllers are connected
print("Scanning all controller slots (0-3)...")
for cid in range(4):
    state = XINPUT_STATE()
    res = dll.XInputGetState(cid, ctypes.byref(state))
    status = "CONNECTED" if res == 0 else "not connected"
    print(f"  Controller {cid}: {status}")

print()
print("Press any button on your GAMING controller (15 seconds)")
print("-" * 50)

prev = {i: None for i in range(4)}
for _ in range(900):
    for cid in range(4):
        state = XINPUT_STATE()
        if dll.XInputGetState(cid, ctypes.byref(state)) == 0:
            gp = state.Gamepad
            current = (gp.wButtons, gp.bLeftTrigger, gp.bRightTrigger)
            if current != (0, 0, 0) and current != prev[cid]:
                print(f"  [Controller {cid}] buttons=0x{gp.wButtons:04X}  LT={gp.bLeftTrigger:3d}  RT={gp.bRightTrigger:3d}")
            prev[cid] = current
    time.sleep(1/60)

print("Done.")
