from dataclasses import dataclass
from typing import List, Dict
@dataclass
class TimeRange:
    start_time: str  
    end_time: str    
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
    availability: Dict[str, List[TimeRange]]  
@dataclass
class Course:
    id: int
    code: str
    name: str
    required_room_type: str
    allowed_instructors: List[int]
    session_type: str
    duration: float  # new field for duration in hours for example 0.75 means 45 minutes
@dataclass
class Group:
    id: int
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
    group_id: int
    course_id: int
    instructor_id: int
    room_id: int
    day: str  
    time_range: TimeRange
