#!/usr/bin/env python

import actionlib
from scitos_msgs.msg import MoveBasePathAction, MoveBasePathGoal

import behavior_container


class MoveBasePathBehavior(behavior_container.BehaviorContainer):

	# ========================================================================
	# Description:
	# Class which contains the behavior for making the robot follow 
	# a specified trajectory
	# ========================================================================

	def __init__(self, behavior_name, interrupt_var, service_str):
		super(MoveBasePathBehavior, self).__init__(behavior_name, interrupt_var)
		self.service_str_ = service_str

	# Method for returning to the standard pose of the robot
	def returnToRobotStandardState(self):
		# save current position (pose, room)
		# return to standard pose / stop all motors
		pass

	# Method for setting parameters for the behavior
	def setParameters(self, target_poses, area_map, path_tolerance, goal_angle_tolerance=1.57, goal_position_tolerance=0.5):
		self.target_poses_ = target_poses
		self.area_map_ = area_map
		self.path_tolerance_ = path_tolerance
		self.goal_position_tolerance_ = goal_position_tolerance
		self.goal_angle_tolerance_ = goal_angle_tolerance
		self.is_finished = False

	def computeNewGoalFromPausedResult(self, prev_action_goal, result):
		last_visited_index = result.last_visited_index
		prev_action_goal.target_poses = prev_action_goal.target_poses[last_visited_index:]
		return prev_action_goal

	# Implemented Behavior
	def executeCustomBehavior(self):
		move_base_path_goal = MoveBasePathGoal()
		move_base_path_goal.target_poses = self.target_poses_
		move_base_path_goal.area_map = self.area_map_
		move_base_path_goal.path_tolerance = self.path_tolerance_
		move_base_path_goal.goal_position_tolerance = self.goal_position_tolerance_
		move_base_path_goal.goal_angle_tolerance = self.goal_angle_tolerance_
		move_base_path_client = actionlib.SimpleActionClient(self.service_str_, MoveBasePathAction)
		self.printMsg("Running move_base_path action...")
		self.move_base_path_result_ = self.runAction(move_base_path_client, move_base_path_goal)['result']
		self.printMsg("move_base_path completed.")