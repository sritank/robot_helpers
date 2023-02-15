from actionlib import SimpleActionClient
import numpy as np
import rospy

import franka_msgs.msg
from franka_gripper.msg import *
from sensor_msgs.msg import JointState
import ipdb

class UR5ArmClient:
    def __init__(self):
        rospy.Subscriber(
            "/joint_states",
            JointState,
            self._state_cb,
            queue_size=1,
        )

    def get_state(self):
        # ipdb.set_trace()
        q = np.asarray(self._state_msg.position)
        dq = np.asarray(self._state_msg.velocity)
        return q, dq

    def recover(self):
        msg = franka_msgs.msg.ErrorRecoveryGoal()
        self._error_recovery_client.send_goal_and_wait(msg)
        rospy.loginfo("Recovered from errors.")

    def _state_cb(self, msg):
        self._state_msg = msg

