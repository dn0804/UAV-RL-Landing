"""
Stage 3: Motor axis verification.
Tests each RC axis one at a time at low power.

SAFETY:
  - Fly in a large open space, nothing breakable nearby.
  - Be ready to grab the drone or let it crash-land.
  - Each movement is 1.5s at RC=20, then 2s hover.
  - Total flight time: ~25 seconds.

After each axis, note which direction the drone moved.
If it moved opposite to expected, flip the corresponding
_SIGN constant in tello_runner.py.
"""

import time
from djitellopy import Tello

RC = 20       # gentle — increase if movement is too subtle to see
MOVE_S = 1.5  # seconds per test
PAUSE_S = 2.0 # hover between tests


def test_axis(tello, name, lr, fb, ud, yaw):
    print(f"\n>>> Testing {name}:  lr={lr:+d}  fb={fb:+d}  ud={ud:+d}  yaw={yaw:+d}")
    print(f"    Expected: {name}")
    input("    Press Enter to execute (or Ctrl+C to abort)...")
    tello.send_rc_control(lr, fb, ud, yaw)
    time.sleep(MOVE_S)
    tello.send_rc_control(0, 0, 0, 0)
    actual = input(f"    What happened? (e.g. 'went forward', 'went backward'): ")
    print(f"    Logged: {name} → {actual}")
    time.sleep(PAUSE_S)
    return actual


tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

print("\nThis test will take off, then test each axis with a pause between.")
print("You will confirm each movement before the next one runs.")
input("Press Enter to take off...")

tello.takeoff()
time.sleep(3)
print("Airborne. Hovering.\n")

results = {}

# fb positive = should go forward (away from you if facing away)
results["FORWARD (fb=+)"] = test_axis(tello, "FORWARD (fb=+)", 0, RC, 0, 0)

# fb negative = should go backward
results["BACKWARD (fb=-)"] = test_axis(tello, "BACKWARD (fb=-)", 0, -RC, 0, 0)

# lr positive = should go right
results["RIGHT (lr=+)"] = test_axis(tello, "RIGHT (lr=+)", RC, 0, 0, 0)

# lr negative = should go left
results["LEFT (lr=-)"] = test_axis(tello, "LEFT (lr=-)", -RC, 0, 0, 0)

# ud positive = should go up
results["UP (ud=+)"] = test_axis(tello, "UP (ud=+)", 0, 0, RC, 0)

# ud negative = should go down
results["DOWN (ud=-)"] = test_axis(tello, "DOWN (ud=-)", 0, 0, -RC, 0)

# yaw positive = should rotate clockwise (from above)
results["YAW CW (yaw=+)"] = test_axis(tello, "YAW CW (yaw=+)", 0, 0, 0, RC)

# yaw negative = should rotate counter-clockwise
results["YAW CCW (yaw=-)"] = test_axis(tello, "YAW CCW (yaw=-)", 0, 0, 0, -RC)

print("\nLanding...")
tello.land()
tello.end()

print("\n" + "=" * 50)
print("AXIS TEST RESULTS")
print("=" * 50)
for test, actual in results.items():
    print(f"  {test:25s} → {actual}")

print("\nIf any axis moved opposite to its name, flip the")
print("corresponding _SIGN constant in tello_runner.py.")