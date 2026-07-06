from vuer import Vuer, VuerSession
from vuer.schemas import MotionControllers
from asyncio import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
# ========== 【修改1】新增夹爪消息导入 ==========
from std_msgs.msg import Float32

# ============================================================================
# 【统一配置区】所有参数统一在此修改
# ============================================================================
# 1. 机械臂初始笛卡尔位姿（base_link坐标系，单位：米 / 度）
ARM_INIT_POSITION = [-0.39,0.01, 0.266]    # 初始位置 [X, Y, Z]
ARM_INIT_EULER = [-96, 32.87, 175] # 初始姿态 [roll, pitch, yaw]
# 2. 坐标系标定参数（复用原验证参数，无需随意修改）
ARM_AXIS_SIGN = [-1, -1, 1]             # 轴方向反转系数
ARM_POSITION_OFFSET = [0.0, 0.0, 0.3]   # 机械臂固定位置偏移
ARM_ORIENTATION_SIGN = [1, 1, 1]        # 姿态轴镜像系数
#ARM_ORIENTATION_OFFSET = [-110.0, 20.0, -160.0]
ARM_ORIENTATION_OFFSET = [0.0, 0.0, 0.0] # 姿态固定偏移（度）
# 3. 控制手柄选择：right=右手 / left=左手
CONTROL_HAND = "right"
# ============================================================================

# ===================== 矩阵/姿态转换工具函数 =====================
def matrix_to_pos_quat(matrix16):
    """
    Vuer返回的16元素列主序4x4矩阵 → 位置向量 + 四元数[w,x,y,z]
    矩阵格式：列主序，第4列前3个元素为平移，左上3x3为旋转矩阵
    """
    m = np.array(matrix16, dtype=float).reshape(4, 4, order='F')  # 列主序重塑
    # 提取平移向量
    pos = m[:3, 3]  # [x, y, z]
    # 提取3x3旋转矩阵
    rot_mat = m[:3, :3]
    
    # 旋转矩阵 → 四元数[w,x,y,z]
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
    """OpenXR原生坐标系 → 标准右手坐标系（位置）：X右 Y上 Z后 → X前 Y左 Z上"""
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
    """
    计算当前位姿相对于初始位姿的偏移
    :param curr_pos: 当前位置 [x,y,z]
    :param curr_quat: 当前四元数 [w,x,y,z]
    :param init_pos: 初始位置 [x,y,z]
    :param init_quat: 初始四元数 [w,x,y,z]
    :return: 相对位置、相对四元数 [w,x,y,z]
    """
    # 位置偏移：当前 - 初始
    rel_pos = curr_pos - init_pos
    # 姿态偏移：q_rel = q_curr * q_init^{-1}（单位四元数逆=共轭）
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
    """
    手柄相对偏移 → 机械臂目标绝对位姿
    :param rel_pos: 手柄相对位置
    :param rel_quat: 手柄相对四元数 [w,x,y,z]
    :param arm_init_pos: 机械臂初始位置
    :param arm_init_euler: 机械臂初始欧拉角
    :return: PoseStamped消息
    """
    # 位置：初始位置 + 相对偏移 × 轴反转系数
    agx_x = arm_init_pos[0] + ARM_AXIS_SIGN[0] * rel_pos[0]
    agx_y = arm_init_pos[1] + ARM_AXIS_SIGN[1] * rel_pos[1]
    agx_z = arm_init_pos[2] + ARM_AXIS_SIGN[2] * rel_pos[2]
    # 姿态：初始欧拉角 + 相对姿态欧拉角 + 固定偏移 + 镜像
    rel_euler = quaternion_to_euler([rel_quat[1], rel_quat[2], rel_quat[3], rel_quat[0]])
    target_euler = arm_init_euler + rel_euler + np.array(ARM_ORIENTATION_OFFSET)
    target_euler = target_euler * np.array(ARM_ORIENTATION_SIGN)
    # 转回四元数
    q_x, q_y, q_z, q_w = euler_to_quaternion(target_euler[0], target_euler[1], target_euler[2])
    # 构造ROS2消息
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

# ===================== ROS2 发布节点 =====================
class ArmControlNode(Node):
    def __init__(self):
        super().__init__("vuer_arm_teleop")
        self.pose_pub = self.create_publisher(PoseStamped, "/control/move_p", 10)
        # ========== 【修改2】新增夹爪比例发布者 ==========
        self.gripper_pub = self.create_publisher(Float32, "/gripper_ratio", 10)
        
        # 机械臂初始位姿
        self.arm_init_pos = np.array(ARM_INIT_POSITION)
        self.arm_init_euler = np.array(ARM_INIT_EULER)
        
        # 手柄参考位姿缓存（首次收到数据自动赋值）
        self.hand_init_pos = None
        self.hand_init_quat = None

    def publish_target_pose(self, curr_hand_pos, curr_hand_quat, trigger_val):
        """发布机械臂目标位姿，首次收到数据自动记录参考位姿"""
        # 首次收到数据，静默记录初始参考位姿
        if self.hand_init_pos is None:
            self.hand_init_pos = curr_hand_pos
            self.hand_init_quat = curr_hand_quat
            print("\n[INFO] 手柄参考位姿已记录，相对控制已启动")
        # 计算手柄相对偏移
        rel_pos, rel_quat = compute_relative_pose(
            curr_hand_pos, curr_hand_quat,
            self.hand_init_pos, self.hand_init_quat
        )
        # 转换为机械臂目标位姿并发布
        pose_msg = relative_to_arm_pose(
            rel_pos, rel_quat,
            self.arm_init_pos, self.arm_init_euler
        )
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(pose_msg)

        # ========== 修复点：强制转原生 float ==========
        gripper_msg = Float32()
        gripper_msg.data = float(trigger_val)
        self.gripper_pub.publish(gripper_msg)

        # 终端实时打印
        arm_pos = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        print(
            f"\r手柄相对偏移: [{rel_pos[0]:+.3f}, {rel_pos[1]:+.3f}, {rel_pos[2]:+.3f}] m  "
            f"机械臂目标: [{arm_pos[0]:+.3f}, {arm_pos[1]:+.3f}, {arm_pos[2]:+.3f}] m  "
            f"扳机值: {trigger_val:.2f}",
            end="", flush=True
        )




# ===================== Vuer 手柄读取主逻辑 =====================
app = Vuer()
ros_node = None

@app.add_handler("CONTROLLER_MOVE")
async def handler(event, session: VuerSession):
    global ros_node
    # 提取控制手柄数据
    if CONTROL_HAND not in event.value:
        return
    # Vuer返回的是16元素变换矩阵（列主序4x4）
    hand_matrix = event.value[CONTROL_HAND]
    hand_state = event.value.get(f"{CONTROL_HAND}State", {})
    # 解析：矩阵 → 位置 + 四元数
    try:
        openxr_pos, openxr_quat = matrix_to_pos_quat(hand_matrix)
        # 扳机值字段为 triggerValue
        trigger_val = hand_state.get("triggerValue", 0.0)
    except Exception:
        return
    # 坐标系转换（OpenXR → 标准右手系）
    standard_pos = openxr_to_standard_coords(openxr_pos)
    standard_quat = openxr_to_standard_quat(openxr_quat)
    # 发布目标位姿
    ros_node.publish_target_pose(standard_pos, standard_quat, trigger_val)

@app.spawn(start=True)
async def main(session: VuerSession):
    global ros_node
    # 初始化ROS2
    rclpy.init()
    ros_node = ArmControlNode()
    ros_node.get_logger().info("Vuer 机械臂遥操作节点启动")
    ros_node.get_logger().info(f"控制手柄: {CONTROL_HAND} | 控制模式: 相对位姿跟随 + 扳机夹爪")
    # 启动手柄数据流
    session.upsert(MotionControllers(
        stream=True,
        key="motion-controller",
        left=True,
        right=True
    ))
    # 主循环
    while True:
        rclpy.spin_once(ros_node, timeout_sec=0.001)
        await sleep(0.01)  # 约100Hz更新频率

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
