# timetable.py

from data_models import Room, Instructor, Course, Group, TimeRange
from typing import List
import json
import os

class TimeTable:
    def __init__(self):
        self.rooms: List[Room] = []
        self.instructors: List[Instructor] = []
        self.courses: List[Course] = []
        self.groups: List[Group] = []
        self.classes: List[Class] = []

        self.load_data_from_files()

    def load_data_from_files(self):
        # Get the directory of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Build absolute paths to the JSON data files
        rooms_file = os.path.join(base_dir, 'input_data/rooms.json')
        instructors_file = os.path.join(base_dir, 'input_data/instructors.json')
        courses_file = os.path.join(base_dir, 'input_data/courses.json')
        groups_file = os.path.join(base_dir, 'input_data/groups.json')

        # Load data from JSON files
        with open(rooms_file, 'r') as f:
            rooms_data = json.load(f)
        with open(instructors_file, 'r') as f:
            instructors_data = json.load(f)
        with open(courses_file, 'r') as f:
            courses_data = json.load(f)
        with open(groups_file, 'r') as f:
            groups_data = json.load(f)

        # Initialize Rooms
        for room in rooms_data:
            self.rooms.append(Room(
                id=room['id'],
                number=room['number'],
                capacity=room['capacity'],
                room_type=room['room_type']
            ))

        # Initialize Instructors
        for instr in instructors_data:
            availability = {}
            for day_str, times in instr['availability'].items():
                availability[day_str] = [TimeRange(t['start_time'], t['end_time']) for t in times]
            self.instructors.append(Instructor(
                id=instr['id'],
                name=instr['name'],
                type=instr['type'],
                availability=availability
            ))

        # Initialize Courses
        for course in courses_data:
            self.courses.append(Course(
                id=course['id'],
                code=course['code'],
                name=course['name'],
                required_room_type=course['required_room_type'],
                allowed_instructors=course['allowed_instructors'],
                session_type=course['session_type']
            ))

        # Initialize Groups
        for group in groups_data:
            self.groups.append(Group(
                id=group['id'],
                major=group['major'],
                year=group['year'],
                specialization=group['specialization'],
                group_name=group['group_name'],
                section=group['section'],
                size=group['size'],
                course_ids=group['course_ids']
            ))
