#!/usr/bin/env python

import behavior_container
import rospy

from geometry_msgs.msg import Quaternion
from move_base_behavior import MoveBaseBehavior

class TrashcanEmptyingBehavior(behavior_container.BehaviorContainer):

	# ========================================================================
	# Description:
	# Class which contains the behavior of trashcan emptying
	# > Go to the trashcan
	# > Take the trashcan
	# > Go to the trolley
	# > Empty the trashcan
	# > Go to the trashcan location
	# > Leave the trashcan
	# ========================================================================

	def __init__(self, behavior_name, interrupt_var, move_base_service_str):
		super(TrashcanEmptyingBehavior, self).__init__(behavior_name, interrupt_var)
		self.move_base_handler_ = MoveBaseBehavior("MoveBaseBehavior", self.interrupt_var_, move_base_service_str)
		(self.trolley_position_, self.trashcan_position_) = (None, None)

	# Method for setting parameters for the behavior
	def setParameters(self, trashcan_position, trolley_position):
		self.trashcan_position_ = trashcan_position
		self.trolley_position_ = trolley_position


	# Method for returning to the standard pose of the robot
	def returnToRobotStandardState(self):
		# save current data if necessary
		# undo or check whether everything has been undone
		pass

	def moveToGoal(self, goal):
		self.move_base_handler_.setParameters(
			goal_position=goal,
			goal_orientation=Quaternion(x=0., y=0., z=0., w=1.),
			header_frame_id='base_link'
		)
		self.move_base_handler_.executeBehavior()

	def takeTrashcan(self):
		# todo (rmb-ma)
		rospy.sleep(2.)

	def emptyTrashcan(self):
		# todo (rmb-ma)
		rospy.sleep(2)

	def leaveTrashcan(self):
		# todo (rmb-ma)
		rospy.sleep(2)

	# Implemented Behavior
	def executeCustomBehavior(self):
		assert(self.trashcan_position_ is not None and self.trolley_position_ is not None)

		self.printMsg("Executing trashcan behavior located on ({}, {})".format(self.trashcan_position_.x, self.trashcan_position_.y))

		# todo (rmb-ma): see how we can go there + see the locations to clean it

		self.printMsg("> Moving to the trashcan")
		self.moveToGoal(self.trashcan_position_)
		if self.move_base_handler_.failed():
			self.printMsg('Trashcan is not accessible. Failed to for emptying trashcan ({}, {})'.format(self.trashcan_position_.x, self.trashcan_position_.y))
			self.state_ = 4
			return
		if self.handleInterrupt() >= 1:
			return

		self.printMsg("> Todo. Take the trashcan")
		self.takeTrashcan()
		if self.handleInterrupt() >= 1:
			return

		self.printMsg("> Moving to the trolley located on ({}, {})".format(self.trolley_position_.x, self.trolley_position_.y))
		self.moveToGoal(self.trolley_position_)
		if self.move_base_handler_.failed():
			self.printMsg('Trolley is not accessible. Failed to for emptying trashcan ({}, {})'.format(self.trashcan_position_.x, self.trashcan_position_.y))
			self.state_ = 4
			return
		if self.handleInterrupt() >= 1:
			return

		self.printMsg("> Todo. Empty the trashcan")
		self.emptyTrashcan()
		if self.handleInterrupt() >= 1:
			return

		self.printMsg("> Going to the trashcan location")
		self.moveToGoal(self.trashcan_position_)
		if self.handleInterrupt() >= 1:
			return
		if self.move_base_handler_.failed():
			self.printMsg('Trashcan location is not accessible. Failed to for emptying trashcan ({}, {})'.format(self.trashcan_position_.x, self.trashcan_position_.y))
			self.state_ = 4
			return

		self.printMsg("> Todo. Leave the trashcan")
		self.leaveTrashcan()
		if self.handleInterrupt() >= 1:
			return

