#!/usr/bin/env python

import csv
import datetime
import os

import database
import database_classes


class CSVToJsonEncoder():

	# Required attributes
	csv_file_path_ = ""
	database_file_path = "" # This must be the path of the folder where the "resources" folder is contained.
	csv_room_plan_ = None
	csv_territory_plan_ = None 
	database_ = None
	
	
	# Constructor
	def __init__(self, csv_file_path="", database_file_path=""):
		self.csv_file_path_ = csv_file_path
		self.database_ = database.Database(extracted_file_path=database_file_path)
		#try:
			# There is a database in the specified directory
		self.database_.loadDatabase()
		#except:
			# There is no database in the specified directory
		#	print "There is no database in the specified directory!"
		#	exit(1)
	
	
	# Load a CSV file
	def loadCSVFiles(self, room_plan_name_sub = "ROOMPLAN.csv", territory_plan_name_sub = "TERRITORYPLAN.csv"):
		# Load the room plan
		file = open(str(self.csv_file_path_) + str(room_plan_name_sub), "r")
		self.csv_room_plan_ = csv.reader(file, dialect="excel")
		# Load the territory plan
		file = open(str(self.csv_file_path_) + str(territory_plan_name_sub), "r")
		self.csv_territory_plan_ = csv.reader(file, dialect="excel")

	# Fill in the database lists
	def feedDatabaseWithCSVData(self):

		# Feed in all the data from the room plan
		existing_rooms = []
		for row in self.csv_room_plan_:
			# Check if the room is already documented and create a new RoomItem otherwise
			room_id = int(row[3])
			room = self.database_.getRoom(room_id)
			if (room == None):
				# Actually, an error should be risen, but for now...
				print "ROOM WITH ID " + str(room_id) + " DOES NOT EXIST!"
				room = database_classes.RoomItem()
				self.database_.rooms_.append(room)
			# Update the data of the concerning RoomItem		
			room.room_position_id_ = row[0]
			room.room_floor_id_ = row[1]
			room.room_building_id_ = row[2]
			room.room_id_ = room_id
			room.room_name_ = row[4]
			room.room_surface_type_ = int(row[6])
			room.room_cleaning_type_ = int(row[7])
			room.room_surface_area_ = float(row[8])
			room.room_trashcan_count_ = int(row[9])
			existing_rooms.append(room.room_id_)
			
		# Save the database
		self.database_.saveRoomDatabase(temporal=False)	

		# Feed in all the data from the territory plan
		for row in self.csv_territory_plan_:
			room_id = int(row[3])
			room = self.database_.getRoom(room_id)
			if (room == None):
				# Actually, an error should be risen, but for now...
				print "ROOM WITH ID " + str(room_id) + " DOES NOT EXIST!"
				room = database_classes.RoomItem()
				self.database_.rooms_.append(room)
			# Update the data of the concerning RoomItem
			scheduled_days = []
			for day in range(14):
				scheduled_days.append(row[11 + day])
			room.room_scheduled_days_ = scheduled_days
			
		# Save the database
		self.database_.saveRoomDatabase(temporal=False)	

		# Remove all rooms which were not listed in the room plan
		for room in self.database_.rooms_:
			if (not(room.room_id_ in existing_rooms)):
				self.database_.rooms_.remove(room)
			# Save the database
			self.database_.saveRoomDatabase(temporal=False)
		
		# Save the database
		self.database_.saveRoomDatabase(temporal=False)	
		
	
	
	# Save the database
	def saveDatabaseToFile(self):
		self.database_.saveRoomDatabase(temporal=False)
		
		
	# Public method. This one shall be called. Only this one shall be called.
	def makeDatabase(self, room_plan_name = "ROOMPLAN.csv", territory_plan_name = "TERRITORYPLAN.csv"):
		self.loadCSVFiles(room_plan_name_sub = room_plan_name, territory_plan_name_sub = territory_plan_name)
		self.feedDatabaseWithCSVData()
		self.saveDatabaseToFile()
		
	

# =========================================================================================
# Test routine
# =========================================================================================

# Initialize and load data from the files
encoder = CSVToJsonEncoder(csv_file_path="csv/", database_file_path="")
encoder.makeDatabase()
