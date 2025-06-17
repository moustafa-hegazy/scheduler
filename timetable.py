import json
import os
from typing import List
from data_models import Room, Instructor, Course, Group, Class, TimeRange

class TimeTable:
    def __init__(self):
        self.rooms: List[Room] = []
        self.instructors: List[Instructor] = []
        self.courses: List[Course] = []
        self.groups: List[Group] = []
        self.classes: List[Class] = []

        self.load_data_from_files()

    def load_data_from_files(self, data_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = data_dir if data_dir else os.path.join(base_dir, 'input_data')

        rooms_file = os.path.join(data_dir, 'rooms.json')
        instructors_file = os.path.join(data_dir, 'instructors.json')
        courses_file = os.path.join(data_dir, 'courses.json')
        groups_file = os.path.join(data_dir, 'groups.json')

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
        instructor_objs = []
        for instr in instructors_data:
            availability_slots = {}
            if 'availability' in instr and isinstance(instr['availability'], dict):
                for day_str, slots in instr['availability'].items():
                    # slots are given as [1,2,3,4,5,6,7,8] or similar
                    # convert to zero-based [0..7] if not already
                    zero_based = [s - 1 for s in slots] if slots and all(isinstance(s, (int, float)) for s in slots) else list(range(8))
                    availability_slots[day_str] = zero_based
            else:
                # Default to full availability if 'availability' is missing or invalid
                availability_slots = {"Monday": list(range(8)), "Tuesday": list(range(8)), "Wednesday": list(range(8)), 
                                     "Thursday": list(range(8)), "Friday": list(range(8)), "Saturday": list(range(8)), 
                                     "Sunday": list(range(8))}
            instructor_objs.append(Instructor(
                id=instr['id'],
                name=instr['name'],
                type=instr['type'],
                availability_slots=availability_slots
            ))
        self.instructors = instructor_objs

        # Initialize Courses
        for course in courses_data:
            # duration given directly in slots, no conversion needed
            self.courses.append(Course(
                id=course['id'],
                code=course['code'],
                name=course['name'],
                required_room_type=course['required_room_type'],
                allowed_instructors=course['allowed_instructors'],
                session_type=course['session_type'],
                duration=course['duration']  # already in slots
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