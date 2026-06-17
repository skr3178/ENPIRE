RED_LO,  RED_HI  = (0, 120, 70),   (10, 255, 255)    # low-hue red band (HSV)
RED_WLO, RED_WHI = (170, 120, 70), (180, 255, 255)   # red wraps the hue circle
CONTOUR_EPS_FRAC = 0.02      # approxPolyDP tolerance, as a fraction of the perimeter
T_CORNERS        = 8         # a clean T silhouette has 8 corners
DETECT_RETRIES   = 100       # frames to wait out motion blur / occlusion
RETRY_DELAY_S    = 0.1
 
# Segment the red T from an RGB frame. Red straddles the 0/180 hue seam, so we
# OR two HSV ranges into one binary mask of the T's pixels.
def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, RED_LO, RED_HI) | cv2.inRange(hsv, RED_WLO, RED_WHI)
 
# Take the largest blob in the mask, simplify it to a polygon, and verify it
# really is the T: exactly 8 corners arranged in the expected T topology.
def find_t_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t = max(cnts, key=cv2.contourArea)
    corners = cv2.approxPolyDP(t, CONTOUR_EPS_FRAC * cv2.arcLength(t, True), True)
    assert len(corners) == T_CORNERS and is_t_topology(corners), "not a clean T"
    return corners
 
# Recover the T's planar pose from its corners: the centroid gives (x, y) and
# the longest edge (the T's stem) gives the heading yaw.
def solve_t_pose(corners):
    stem = max(t_edges(corners), key=edge_length)
    return polygon_centroid(corners), edge_angle(stem)    # (x, y, yaw)
 
# Grab a top-camera frame and return (mask, pose); on a blurred or occluded
# view the contour check fails, so retry until a clean T is found or we time out.
def detect_t():
    for _ in range(DETECT_RETRIES):
        mask = red_mask(get_camera_image("top"))
        try:
            return mask, solve_t_pose(find_t_contour(mask))
        except AssertionError:
            time.sleep(RETRY_DELAY_S)
 
# Drive the T back to its start pose: home the arm, exit early if it is already
# on the goal, un-flip an upside-down T, then grasp it and place it at the start.
def reset_pusht(side):
    go_home()
    mask, pose = detect_t()
    # goal_mask: stored reset reference, compared in image space
    if goal_match(mask, goal_mask):                       # already at the goal pose
        return
    if t_is_upside_down(pose):                            # normalize a flipped T first
        regrasp_and_flip(side, pose)                      # grasp, lift, rotate, set down
        mask, pose = detect_t()
    freespace_move(side, hover_pose(pose))                # locate -> hover (RRT-connect)
    descend(); close_gripper(side, RESET_HOLD_WIDTH)      # descend -> grasp
    freespace_move(side, hover_pose(start_pose))          # carry to the recorded start
    descend(); open_gripper(side)                         # descend -> release
    go_home()