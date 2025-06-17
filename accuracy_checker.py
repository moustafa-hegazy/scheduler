import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from timetable import TimeTable
from genetic_algorithm import GeneticAlgorithm
from BSO_algorithm import BSOalgorithm
from AntColony import AntCOlonyAlgorithm
from schedule_printer import print_schedule
from data_models import Room, Instructor, Course, Group, Class, TimeRange

def main():
    # Base directory for input and output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_comparison_dir = os.path.join(base_dir, 'input_data_comparison')
    output_dir = os.path.join(base_dir, 'output_data')
    os.makedirs(input_comparison_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load and convert CSV data to JSON-compatible format
    rooms_df = pd.read_csv(os.path.join(base_dir, 'rooms.csv'))
    rooms = [{"id": int(row['room_id']), "number": row['room_code'], "capacity": int(row['capacity']), "room_type": "Lab" if row['is_lab'] == 1 else "Lecture"} for _, row in rooms_df.iterrows()]
    with open(os.path.join(input_comparison_dir, 'rooms.json'), 'w') as f:
        json.dump(rooms, f, indent=4)

    instructors_df = pd.read_csv(os.path.join(base_dir, 'instructors.csv'))
    instructors = [{"id": row['instr_id'], "name": row['instructor_name'], "type": "Professor", "availability": {"Monday": [1, 2, 3, 4, 5, 6, 7, 8], "Tuesday": [1, 2, 3, 4, 5, 6, 7, 8], "Wednesday": [1, 2, 3, 4, 5, 6, 7, 8], "Thursday": [1, 2, 3, 4, 5, 6, 7, 8], "Friday": [1, 2, 3, 4, 5, 6, 7, 8], "Saturday": [1, 2, 3, 4, 5, 6, 7, 8], "Sunday": [1, 2, 3, 4, 5, 6, 7, 8]}} for _, row in instructors_df.iterrows()]
    with open(os.path.join(input_comparison_dir, 'instructors.json'), 'w') as f:
        json.dump(instructors, f, indent=4)

    courses_df = pd.read_csv(os.path.join(base_dir, 'courses.csv'))
    courses = [{"id": i, "code": row['course_code'], "name": row['course_code'], "required_room_type": "Lab" if row['is_lab'] == 1 else "Lecture", "allowed_instructors": [instr['id'] for instr in instructors], "session_type": "Lab" if row['is_lab'] == 1 else "Lecture", "duration": 1} for i, row in courses_df.iterrows()]
    with open(os.path.join(input_comparison_dir, 'courses.json'), 'w') as f:
        json.dump(courses, f, indent=4)

    groups_df = pd.read_csv(os.path.join(base_dir, 'groups.csv'))
    course_offerings_df = pd.read_csv(os.path.join(base_dir, 'course_offerings.csv'))
    groups = []
    for _, row in groups_df.iterrows():
        group_id = int(row['group_id'])
        course_ids = [c['id'] for c in courses if c['code'] in course_offerings_df[course_offerings_df['group_id'] == group_id]['course_code'].values]
        groups.append({"id": group_id, "major": "CSIT", "year": 1, "specialization": "", "group_name": row['group_name'], "section": "0", "size": 30, "course_ids": course_ids})
    with open(os.path.join(input_comparison_dir, 'groups.json'), 'w') as f:
        json.dump(groups, f, indent=4)

    # Initialize timetable with comparison data
    timetable = TimeTable()
    timetable.load_data_from_files(input_comparison_dir)

    # Define algorithms to compare
    algorithms = {
        "GA": GeneticAlgorithm(timetable),
        "BSO": BSOalgorithm(timetable),
        "ACS": AntCOlonyAlgorithm(timetable)
    }

    # Run each algorithm and collect data
    results = {}
    for name, algo in algorithms.items():
        start_time = time.time()
        if name == "GA":
            solution, generations = algo.run(), getattr(algo, 'max_iterations', 500)  # Assume max_iterations if not tracked
        else:
            solution = algo.run()
            generations = getattr(algo, 'max_iterations', 0)  # Proxy for generations
        end_time = time.time()
        execution_time = end_time - start_time
        scheduled_offerings = len(set((c.group_id, c.course_id) for c in solution))
        clashes = algo.count_clashes(solution) if hasattr(algo, 'count_clashes') else sum(1 for i, c1 in enumerate(solution) for j, c2 in enumerate(solution) if i < j and c1.day == c2.day and c1.group_id == c2.group_id and c1.time_range == c2.time_range)
        fitness = max(0, (len(course_offerings_df) - clashes) / len(course_offerings_df)) if len(course_offerings_df) > 0 else 0
        results[name] = {
            "solution": solution,
            "generations": generations,
            "fitness": fitness,
            "execution_time": execution_time,
            "scheduled_offerings": scheduled_offerings,
            "clashes": clashes
        }
        print(f"{name} Results:")
        print_schedule(timetable, solution)
        print(f"Fitness: {fitness:.2%}, Generations: {generations}, Execution Time: {execution_time:.2f}s, Scheduled Offerings: {scheduled_offerings}/{len(course_offerings_df)}, Clashes: {clashes}\n")

    # Generate graphs
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(range(result["generations"]), [result["fitness"]] * result["generations"], label=name, marker='o')
    plt.xlabel("Generations")
    plt.ylabel("Fitness (%)")
    plt.title("Fitness Comparison Across Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fitness_comparison.png"))
    plt.close()

    # Summary
    print("Comparison Summary:")
    for name, result in results.items():
        print(f"{name}: Fitness = {result['fitness']:.2%}, Generations = {result['generations']}, Execution Time = {result['execution_time']:.2f}s, Clashes = {result['clashes']}")

if __name__ == '__main__':
    main()