from vuer import Vuer, VuerSession
from vuer.schemas import MotionControllers
from asyncio import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

# ============================================================================
# 【方案A：话题后缀命名】
# 右臂保持原话题不变，左臂话题加 _left 后缀
# 左臂启动命令需加话题重映射：
# ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
#     can_port:=can1 arm_type:=nero effector_type:=none \
#     speed_percent:=15 auto_enable:=true \
#     remap:=/control/move_p:=/control/move_p_left \
#     remap:=/gripper_ratio:=/gripper_ratio_left
# ============================================================================
# ---------- 公共标定参数（两台Nero机械臂一致） ----------
ARM_AXIS_SIGN = [-1, -1, 1]
ARM_ORIENTATION_SIGN = [1, 1, 1]
ARM_ORIENTATION_OFFSET = [0.0, 0.0, 0.0]

# ---------- 右手机械臂配置（与原单臂完全一致，无需修改） ----------
RIGHT_ARM_INIT_POSITION = [-0.39, 0.01, 0.266]
RIGHT_ARM_INIT_EULER = [-96, 32.87, 175]
RIGHT_ARM_POSE_TOPIC = "/control/move_p"
RIGHT_GRIPPER_TOPIC = "/gripper_ratio"

# ---------- 左手机械臂配置（新增机械臂） ----------
# 初始位姿请根据左臂实际安装位置微调
LEFT_ARM_INIT_POSITION = [-0.39, -0.01, 0.266]
LEFT_ARM_INIT_EULER = [-96, 32.87, 175]
LEFT_ARM_POSE_TOPIC = "/left_arm/control/move_p"
LEFT_GRIPPER_TOPIC = "/gripper_left_ratio"

# ============================================================================


# ===================== 矩阵/姿态转换工具函数 =====================
def matrix_to_pos_quat(matrix16):
    """
    Vuer返回的16元素列主序4x4矩阵 → 位置向量 + 四元数[w,x,y,z]
    """
    m = np.array(matrix16, dtype=float).reshape(4, 4, order='F')
    pos = m[:3, 3]
    rot_mat = m[:3, :3]
    
    trace = np.trace(rot_mat)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        x = 0.25 * s
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        y = 0.25 * s
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z])
    return pos, quat


def openxr_to_standard_coords(openxr_pos):
    """OpenXR原生坐标系 → 标准右手坐标系（位置）"""
    standard_x = -openxr_pos[2]
    standard_y = -openxr_pos[0]
    standard_z = openxr_pos[1]
    return np.array([standard_x, standard_y, standard_z])


def openxr_to_standard_quat(openxr_quat):
    """OpenXR原生坐标系 → 标准右手坐标系（四元数[w,x,y,z]）"""
    w, x, y, z = openxr_quat
    qx = -z
    qy = -x
    qz = y
    return np.array([w, qx, qy, qz])


def quaternion_to_euler(q, degrees=True):
    """四元数[x,y,z,w]转欧拉角[roll,pitch,yaw]"""
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(roll, pitch, yaw, degrees=True):
    """欧拉角[roll,pitch,yaw]转四元数[x,y,z,w]"""
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])


def compute_relative_pose(curr_pos, curr_quat, init_pos, init_quat):
    """计算当前位姿相对于初始位姿的偏移"""
    rel_pos = curr_pos - init_pos

    init_w, init_x, init_y, init_z = init_quat
    init_inv = np.array([init_w, -init_x, -init_y, -init_z])
    
    w1, x1, y1, z1 = curr_quat
    w2, x2, y2, z2 = init_inv
    rel_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    rel_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    rel_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    rel_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    rel_quat = np.array([rel_w, rel_x, rel_y, rel_z])

    return rel_pos, rel_quat


def relative_to_arm_pose(rel_pos, rel_quat, arm_init_pos, arm_init_euler):
    """手柄相对偏移 → 机械臂目标绝对位姿"""
    agx_x = arm_init_pos[0] + ARM_AXIS_SIGN[0] * rel_pos[0]
    agx_y = arm_init_pos[1] + ARM_AXIS_SIGN[1] * rel_pos[1]
    agx_z = arm_init_pos[2] + ARM_AXIS_SIGN[2] * rel_pos[2]

    rel_euler = quaternion_to_euler([rel_quat[1], rel_quat[2], rel_quat[3], rel_quat[0]])
    target_euler = arm_init_euler + rel_euler + np.array(ARM_ORIENTATION_OFFSET)
    target_euler = target_euler * np.array(ARM_ORIENTATION_SIGN)

    q_x, q_y, q_z, q_w = euler_to_quaternion(target_euler[0], target_euler[1], target_euler[2])

    msg = PoseStamped()
    msg.header.frame_id = "base_link"
    msg.pose.position.x = float(agx_x)
    msg.pose.position.y = float(agx_y)
    msg.pose.position.z = float(agx_z)
    msg.pose.orientation.w = float(q_w)
    msg.pose.orientation.x = float(q_x)
    msg.pose.orientation.y = float(q_y)
    msg.pose.orientation.z = float(q_z)
    return msg


# ===================== ROS2 双机械臂控制节点 =====================
class ArmControlNode(Node):
    def __init__(self):
        super().__init__("vuer_dual_arm_teleop")
        # 右手机械臂发布者
        self.right_pose_pub = self.create_publisher(PoseStamped, RIGHT_ARM_POSE_TOPIC, 10)
        self.right_gripper_pub = self.create_publisher(Float32, RIGHT_GRIPPER_TOPIC, 10)
        # 左手机械臂发布者
        self.left_pose_pub = self.create_publisher(PoseStamped, LEFT_ARM_POSE_TOPIC, 10)
        self.left_gripper_pub = self.create_publisher(Float32, LEFT_GRIPPER_TOPIC, 10)
        
        # 机械臂初始位姿
        self.right_arm_init_pos = np.array(RIGHT_ARM_INIT_POSITION)
        self.right_arm_init_euler = np.array(RIGHT_ARM_INIT_EULER)
        self.left_arm_init_pos = np.array(LEFT_ARM_INIT_POSITION)
        self.left_arm_init_euler = np.array(LEFT_ARM_INIT_EULER)
        
        # 手柄参考位姿缓存（左右独立首帧校准）
        self.right_hand_init_pos = None
        self.right_hand_init_quat = None
        self.left_hand_init_pos = None
        self.left_hand_init_quat = None

    def publish_single_arm(self, hand_side, curr_pos, curr_quat, trigger_val):
        """
        通用单臂发布函数
        :param hand_side: "right" / "left"
        """
        if hand_side == "right":
            init_pos = self.right_hand_init_pos
            init_quat = self.right_hand_init_quat
            arm_init_pos = self.right_arm_init_pos
            arm_init_euler = self.right_arm_init_euler
            pose_pub = self.right_pose_pub
            gripper_pub = self.right_gripper_pub
        else:
            init_pos = self.left_hand_init_pos
            init_quat = self.left_hand_init_quat
            arm_init_pos = self.left_arm_init_pos
            arm_init_euler = self.left_arm_init_euler
            pose_pub = self.left_pose_pub
            gripper_pub = self.left_gripper_pub

        # 首帧自动记录参考位姿
        if init_pos is None:
            if hand_side == "right":
                self.right_hand_init_pos = curr_pos
                self.right_hand_init_quat = curr_quat
                print("\n[INFO] 右手柄校准完成，右臂进入相对控制")
            else:
                self.left_hand_init_pos = curr_pos
                self.left_hand_init_quat = curr_quat
                print("\n[INFO] 左手柄校准完成，左臂进入相对控制")
            return None

        # 计算相对偏移
        rel_pos, rel_quat = compute_relative_pose(curr_pos, curr_quat, init_pos, init_quat)

        # 生成并发布位姿指令
        pose_msg = relative_to_arm_pose(rel_pos, rel_quat, arm_init_pos, arm_init_euler)
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_pub.publish(pose_msg)

        # 发布夹爪指令
        gripper_msg = Float32()
        gripper_msg.data = float(trigger_val)
        gripper_pub.publish(gripper_msg)

        return rel_pos, pose_msg, trigger_val


# ===================== Vuer 双手柄事件处理 =====================
app = Vuer()
ros_node = None
right_display = None
left_display = None


@app.add_handler("CONTROLLER_MOVE")
async def handler(event, session: VuerSession):
    global ros_node, right_display, left_display

    # 处理右手柄 → 控制右臂
    if "right" in event.value:
        try:
            hand_matrix = event.value["right"]
            hand_state = event.value.get("rightState", {})
            openxr_pos, openxr_quat = matrix_to_pos_quat(hand_matrix)
            trigger_val = hand_state.get("triggerValue", 0.0)

            standard_pos = openxr_to_standard_coords(openxr_pos)
            standard_quat = openxr_to_standard_quat(openxr_quat)

            result = ros_node.publish_single_arm("right", standard_pos, standard_quat, trigger_val)
            if result:
                rel_pos, pose_msg, trig = result
                arm_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
                right_display = (rel_pos, arm_pos, trig)
        except Exception:
            pass

    # 处理左手柄 → 控制左臂
    if "left" in event.value:
        try:
            hand_matrix = event.value["left"]
            hand_state = event.value.get("leftState", {})
            openxr_pos, openxr_quat = matrix_to_pos_quat(hand_matrix)
            trigger_val = hand_state.get("triggerValue", 0.0)

            standard_pos = openxr_to_standard_coords(openxr_pos)
            standard_quat = openxr_to_standard_quat(openxr_quat)

            result = ros_node.publish_single_arm("left", standard_pos, standard_quat, trigger_val)
            if result:
                rel_pos, pose_msg, trig = result
                arm_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
                left_display = (rel_pos, arm_pos, trig)
        except Exception:
            pass

    # 终端统一打印
    if right_display or left_display:
        line = ""
        if right_display:
            r_rel, r_arm, r_trig = right_display
            line += f"[右臂] 偏移:[{r_rel[0]:+.2f},{r_rel[1]:+.2f},{r_rel[2]:+.2f}]m  目标:[{r_arm[0]:+.2f},{r_arm[1]:+.2f},{r_arm[2]:+.2f}]m  扳机:{r_trig:.2f}  |  "
        if left_display:
            l_rel, l_arm, l_trig = left_display
            line += f"[左臂] 偏移:[{l_rel[0]:+.2f},{l_rel[1]:+.2f},{l_rel[2]:+.2f}]m  目标:[{l_arm[0]:+.2f},{l_arm[1]:+.2f},{l_arm[2]:+.2f}]m  扳机:{l_trig:.2f}"
        print(f"\r{line}", end="", flush=True)


@app.spawn(start=True)
async def main(session: VuerSession):
    global ros_node

    rclpy.init()
    ros_node = ArmControlNode()
    ros_node.get_logger().info("双机械臂VR遥操作节点启动")
    ros_node.get_logger().info("控制映射：右手柄→右臂 | 左手柄→左臂")
    ros_node.get_logger().info(f"右臂话题：{RIGHT_ARM_POSE_TOPIC} | {RIGHT_GRIPPER_TOPIC}")
    ros_node.get_logger().info(f"左臂话题：{LEFT_ARM_POSE_TOPIC} | {LEFT_GRIPPER_TOPIC}")

    session.upsert(MotionControllers(
        stream=True,
        key="motion-controller",
        left=True,
        right=True
    ))

    while True:
        rclpy.spin_once(ros_node, timeout_sec=0.001)
        await sleep(0.01)


# ===================== 程序入口 =====================
if __name__ == "__main__":
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] 正在退出...")
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()
        print("[INFO] 节点已关闭")
