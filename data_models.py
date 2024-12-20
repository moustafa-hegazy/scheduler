# data_models.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TimeRange:
    start_time: str  # e.g. "Slot 1"
    end_time: str    # e.g. "Slot 3"

@dataclass
class Room:
    id: int
    number: str
    capacity: int
    room_type: str

@dataclass
class Instructor:
    id: int
    name: str
    type: str
    availability_slots: Dict[str, List[int]]  # day -> list of zero-based slot indices available

@dataclass
class Course:
    id: int
    code: str
    name: str
    required_room_type: str
    allowed_instructors: List[int]
    session_type: str
    duration: int  # now in slots directly

@dataclass
class Group:
    id: float
    major: str
    year: int
    specialization: str
    group_name: str
    section: str
    size: int
    course_ids: List[int]

@dataclass
class Class:
    id: int
    group_id: float
    course_id: int
    instructor_id: int
    room_id: int
    day: str
    time_range: TimeRange