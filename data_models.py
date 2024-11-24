from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TimeRange:
    start_time: str  # Format: "HH:MM"
    end_time: str    # Format: "HH:MM"

@dataclass
class Room:
    id: int
    number: str
    capacity: int
    room_type: str  # e.g., "classroom"

@dataclass
class Instructor:
    id: int
    name: str
    type: str  # "TA" or "Professor"
    availability: Dict[str, List[TimeRange]]  # Availability per day, e.g., "Monday"

@dataclass
class Course:
    id: int
    code: str
    name: str
    required_room_type: str
    allowed_instructors: List[int]  # List of Instructor IDs
    session_type: str  # "Lecture", "Lab", "Tutorial"

@dataclass
class Group:
    id: int
    major: str
    year: int
    specialization: str
    group_name: str
    section: str
    size: int
    course_ids: List[int]  # List of Course IDs a group needs to take

@dataclass
class Class:
    id: int
    group_id: int
    course_id: int
    instructor_id: int
    room_id: int
    day: str  
    time_range: TimeRange
