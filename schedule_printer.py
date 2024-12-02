# schedule_printer.py

from data_models import Class
from timetable import TimeTable
from typing import List
import json
import os  # Import os module to handle file paths

def print_schedule(timetable: TimeTable, classes: List[Class]):
    """Print the generated schedule in a readable format"""
    print("\nGenerated Schedule:")
    print("-" * 100)

    # Sort classes by day and time
    sorted_classes = sorted(classes,
                            key=lambda c: (c.day,
                                           c.time_range.start_time))

    current_day = None
    for class_ in sorted_classes:
        # Get related objects
        room = next(r for r in timetable.rooms if r.id == class_.room_id)
        instructor = next(i for i in timetable.instructors if i.id == class_.instructor_id)
        course = next(c for c in timetable.courses if c.id == class_.course_id)
        group = next(g for g in timetable.groups if g.id == class_.group_id)

        # Print day header if new day
        if current_day != class_.day:
            current_day = class_.day
            print(f"\n{current_day}")
            print("-" * 100)

        print(f"Time: {class_.time_range.start_time}-{class_.time_range.end_time} | "
              f"Course: {course.code} ({course.name}) [{course.session_type}] | "
              f"Group: {group.major} Year {group.year} {group.group_name}-{group.section} | "
              f"Room: {room.number} | "
              f"Instructor: {instructor.name} ({instructor.type})")

    # Output the schedule to a JSON file
    schedule_output = []
    for class_ in sorted_classes:
        room = next(r for r in timetable.rooms if r.id == class_.room_id)
        instructor = next(i for i in timetable.instructors if i.id == class_.instructor_id)
        course = next(c for c in timetable.courses if c.id == class_.course_id)
        group = next(g for g in timetable.groups if g.id == class_.group_id)

        schedule_output.append({
            'day': class_.day,
            'time': f"{class_.time_range.start_time}-{class_.time_range.end_time}",
            'course_code': course.code,
            'course_name': course.name,
            'session_type': course.session_type,
            'group': f"{group.major} Year {group.year} {group.group_name}-{group.section}",
            'room': room.number,
            'instructor': instructor.name,
            'instructor_type': instructor.type,
            'year': group.year
        })

    # Ensure the output directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    # Build the full path to the output file
    output_file = os.path.join(output_dir, 'schedule_output.json')

    # Write the schedule to the JSON file
    with open(output_file, 'w') as f:
        json.dump(schedule_output, f, indent=4)
