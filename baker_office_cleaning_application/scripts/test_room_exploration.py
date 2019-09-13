#!/usr/bin/env python

import roslib
roslib.load_manifest('baker_office_cleaning_application')
import actionlib
import rospy
import tf

#import baker_office_cleaning_application.scripts

#move_base_msgs
from baker_msgs.srv import *
from ipa_building_msgs.msg import *
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose2D, Point32
from sensor_msgs.msg import Image
from move_base_msgs.msg import *
from scitos_msgs.msg import MoveBasePathAction
from scitos_msgs.msg import MoveBasePathGoal

#import simple_script_server
#sss = simple_script_server.simple_script_server()

class TestClass():
	def explore(self):
		print "TestClass WORKS"

#class SeqControl(simple_script_server.script):
class SeqControl():
	def explore(self):
		rospy.init_node('exploration_node')

		# receive the navigation map in sensor_msgs/Image format
		print "Waiting for service '/baker/get_map_image' to become available ..."
		rospy.wait_for_service('/baker/get_map_image')
		try:
			get_map = rospy.ServiceProxy('/baker/get_map_image', GetMap)
			self.map_data = get_map()
		except rospy.ServiceException, e:
			print "Service call failed: %s" % e
		print "Map received with resolution: ", self.map_data.map_resolution, "   and origin: ", self.map_data.map_origin
		
		# compute map division into rooms
		# todo:

		# compute exploration path
		planning_mode = 2 # viewpoint planning
		fov_points = [Point32(x=0.04035, y=0.136), Point32(x=0.04035, y=-0.364), Point32(x=0.54035, y=-0.364), Point32(x=0.54035, y=0.136)] # this field of view represents the off-center iMop floor wiping device
		#fov_points = [Point32(x=0.15, y=0.35), Point32(x=0.15, y=-0.35), Point32(x=1.15, y=-0.65), Point32(x=1.15, y=0.65)] # this field of view fits a Asus Xtion sensor mounted at 0.63m height (camera center) pointing downwards to the ground in a respective angle
		# ... or ...
		#planning_mode = 1 # footprint planning
		#fov_points = [Point32(x=-0.3, y=0.3), Point32(x=-0.3, y=-0.3), Point32(x=0.3, y=-0.3), Point32(x=0.3, y=0.3)] # this is the working area of a vacuum cleaner with 60 cm width
		fov_origin = Point32(x=0., y=0.)
		exploration_goal = RoomExplorationGoal()
		exploration_goal.input_map = self.map_data.map
		exploration_goal.map_resolution = self.map_data.map_resolution
		exploration_goal.map_origin = self.map_data.map_origin
		exploration_goal.robot_radius = 0.3
		exploration_goal.coverage_radius = 0.3
		exploration_goal.field_of_view = fov_points
		exploration_goal.field_of_view_origin = fov_origin
		exploration_goal.starting_position = Pose2D(x=1., y=0., theta=0.)
		exploration_goal.planning_mode = planning_mode
		print "Waiting for action '/room_exploration/room_exploration_server' to become available ..."
		exploration_client = actionlib.SimpleActionClient('/room_exploration/room_exploration_server', RoomExplorationAction)
		exploration_client.wait_for_server()
		print "Sending goal ..."
		exploration_client.send_goal(exploration_goal)
		exploration_client.wait_for_result()
		exploration_result = exploration_client.get_result()
		print "Exploration path received with length ", len(exploration_result.coverage_path_pose_stamped)

		# command path follow movement
		move_base_path_goal = MoveBasePathGoal()
		move_base_path_goal.target_poses = exploration_result.coverage_path_pose_stamped
		move_base_path_goal.path_tolerance = 0.2 #0.1
		move_base_path_goal.goal_position_tolerance = 0.5 #0.25 #0.1
		move_base_path_goal.goal_angle_tolerance = 1.57 #0.7 #0.087
		print "Waiting for action 'move_base_path' to become available ..."
		move_base_path = actionlib.SimpleActionClient('/move_base_path', MoveBasePathAction)
		move_base_path.wait_for_server()
		print "Sending goal ..."
		move_base_path.send_goal(move_base_path_goal)
		move_base_path.wait_for_result()
		print move_base_path.get_result()
		
		# command wall following movement
		move_base_path_goal = MoveBasePathGoal()
		move_base_path_goal.target_poses = exploration_result.coverage_path_pose_stamped
		move_base_path_goal.path_tolerance = 0.2 #0.1
		move_base_path_goal.goal_position_tolerance = 0.4 #0.25 #0.1
		move_base_path_goal.goal_angle_tolerance = 3.14 #0.7 #0.087
		print "Waiting for action 'move_base_path' to become available ..."
		move_base_path = actionlib.SimpleActionClient('/move_base_wall_follow', MoveBasePathAction)
		move_base_path.wait_for_server()
		print "Sending goal ..."
		move_base_path.send_goal(move_base_path_goal)
		move_base_path.wait_for_result()
		print move_base_path.get_result()

if __name__ == '__main__':
	try:
		command = 'script_1 = TestClass()'
		exec(command)
		script_1.explore()

		script = SeqControl()
		script.explore()
	except rospy.ROSInterruptException:
		print "Keyboard Interrupt"
