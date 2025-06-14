# schedule_printer.py

from data_models import Class
from timetable import TimeTable
from typing import List
import json
import os

def print_schedule(timetable: TimeTable, classes: List[Class]):
    print("\nGenerated Schedule:")
    print("-" * 100)
    # Sort by day and then by start slot
    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    def slot_key(c: Class):
        start_slot = int(c.time_range.start_time.split()[1])
        return (day_order.index(c.day), start_slot)

    sorted_classes = sorted(classes, key=slot_key)

    current_day = None
    for class_ in sorted_classes:
        room = next(r for r in timetable.rooms if r.id == class_.room_id)
        instructor = next(i for i in timetable.instructors if i.id == class_.instructor_id)
        course = next(c for c in timetable.courses if c.id == class_.course_id)
        group = next(g for g in timetable.groups if g.id == class_.group_id)
        if current_day != class_.day:
            current_day = class_.day
            print(f"\n{current_day}")
            print("-" * 100)
        print(f"Slots: {class_.time_range.start_time}-{class_.time_range.end_time} | "
              f"Course: {course.code} ({course.name}) [{course.session_type}] | "
              f"Group: {group.major} Year {group.year} {group.group_name}-{group.section} | "
              f"Room: {room.number} | "
              f"Instructor: {instructor.name} ({instructor.type})")

    # Save to JSON
    schedule_output = []
    for class_ in sorted_classes:
        room = next(r for r in timetable.rooms if r.id == class_.room_id)
        instructor = next(i for i in timetable.instructors if i.id == class_.instructor_id)
        course = next(c for c in timetable.courses if c.id == class_.course_id)
        group = next(g for g in timetable.groups if g.id == class_.group_id)
        schedule_output.append({
            'day': class_.day,
            'slots': f"{class_.time_range.start_time}-{class_.time_range.end_time}",
            'course_code': course.code,
            'course_name': course.name,
            'session_type': course.session_type,
            'group': f"{group.major} Year {group.year} {group.group_name}-{group.section}",
            'room': room.number,
            'instructor': instructor.name,
            'instructor_type': instructor.type,
            'year': group.year
        })

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'schedule_output.json')
    with open(output_file, 'w') as f:
        json.dump(schedule_output, f, indent=4)