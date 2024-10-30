from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random
from enum import Enum
import json
import datetime

# Enums for different types
class RoomType(Enum):
    CLASSROOM = "classroom"
    COMPUTER_LAB = "computer_lab"
    PHYSICS_LAB = "physics_lab"
    CHEMISTRY_LAB = "chemistry_lab"

class DayOfWeek(Enum):
    FRIDAY = "Friday"
    Saturday = "Saturday"
    SUNDAY = "Sunday"
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    

class InstructorType(Enum):
    TA = "TA"
    PROFESSOR = "Professor"

class SessionType(Enum):
    LECTURE = "Lecture"
    LAB = "Lab"
    TUTORIAL = "Tutorial"

@dataclass
class TimeRange:
    start_time: str  # Format: "HH:MM"
    end_time: str    # Format: "HH:MM"

@dataclass
class Room:
    id: int
    number: str
    capacity: int
    room_type: RoomType

@dataclass
class Instructor:
    id: int
    name: str
    type: InstructorType
    availability: Dict[DayOfWeek, List[TimeRange]]  # Availability per day

@dataclass
class Course:
    id: int
    code: str
    name: str
    required_room_type: RoomType
    allowed_instructors: List[int]  # List of Instructor IDs
    session_type: SessionType

@dataclass
class Group:
    id: int
    major: str
    year: int
    specialization: str
    group_name: str
    section: str
    size: int
    course_ids: List[int]  # List of Course IDs this group needs to take

@dataclass
class Class:
    id: int
    group_id: int
    course_id: int
    instructor_id: int
    room_id: int
    day: DayOfWeek
    time_range: TimeRange

class TimeTable:
    def __init__(self):
        self.rooms: List[Room] = []
        self.instructors: List[Instructor] = []
        self.courses: List[Course] = []
        self.groups: List[Group] = []
        self.classes: List[Class] = []

        self.load_data_from_files()

    def load_data_from_files(self):
        # Load data from JSON files
        with open('rooms.json', 'r') as f:
            rooms_data = json.load(f)
        with open('instructors.json', 'r') as f:
            instructors_data = json.load(f)
        with open('courses.json', 'r') as f:
            courses_data = json.load(f)
        with open('groups.json', 'r') as f:
            groups_data = json.load(f)

        # Initialize Rooms
        for room in rooms_data:
            self.rooms.append(Room(
                id=room['id'],
                number=room['number'],
                capacity=room['capacity'],
                room_type=RoomType(room['room_type'])
            ))

        # Initialize Instructors
        for instr in instructors_data:
            availability = {}
            for day_str, times in instr['availability'].items():
                day = DayOfWeek(day_str)
                availability[day] = [TimeRange(t['start_time'], t['end_time']) for t in times]
            self.instructors.append(Instructor(
                id=instr['id'],
                name=instr['name'],
                type=InstructorType(instr['type']),
                availability=availability
            ))

        # Initialize Courses
        for course in courses_data:
            self.courses.append(Course(
                id=course['id'],
                code=course['code'],
                name=course['name'],
                required_room_type=RoomType(course['required_room_type']),
                allowed_instructors=course['allowed_instructors'],
                session_type=SessionType(course['session_type'])
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

class GeneticAlgorithm:
    def __init__(self, timetable: TimeTable):
        self.timetable = timetable
        self.population_size = 100
        self.elite_size = 10
        self.tournament_size = 5
        self.mutation_rate = 0.1
        self.max_generations = 1000

        # Generate possible time slots based on instructors' availability and standard class durations
        self.time_slots = self.generate_time_slots()

    def generate_time_slots(self) -> List[Tuple[DayOfWeek, str, str]]:
        """Generate possible time slots based on the standard class durations"""
        time_slots = []
        class_duration = datetime.timedelta(hours=2)  # Assuming classes are 2 hours
        time_format = "%H:%M"

        for day in DayOfWeek:
            # Find earliest and latest times from all instructors for this day
            times = []
            for instr in self.timetable.instructors:
                if day in instr.availability:
                    for tr in instr.availability[day]:
                        start = datetime.datetime.strptime(tr.start_time, time_format)
                        end = datetime.datetime.strptime(tr.end_time, time_format)
                        times.append((start, end))

            if not times:
                continue

            earliest = min(start for start, end in times)
            latest = max(end for start, end in times)

            current_time = earliest
            while current_time + class_duration <= latest:
                start_time_str = current_time.strftime(time_format)
                end_time = current_time + class_duration
                end_time_str = end_time.strftime(time_format)
                time_slots.append((day, start_time_str, end_time_str))
                current_time += datetime.timedelta(minutes=15)  # 15-minute intervals

        return time_slots

    def create_individual(self) -> List[int]:
        """Creates a random valid individual (chromosome)"""
        num_classes = sum(len(group.course_ids) for group in self.timetable.groups)
        chromosome = []

        for _ in range(num_classes):
            timeslot_idx = random.randint(0, len(self.time_slots) - 1)
            room_id = random.choice([room.id for room in self.timetable.rooms])
            instructor_id = random.choice([instr.id for instr in self.timetable.instructors])
            chromosome.extend([timeslot_idx, room_id, instructor_id])

        return chromosome

    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate fitness of a chromosome based on constraint violations"""
        classes = self.create_classes(chromosome)
        clashes = self.count_clashes(classes)
        return 1 / (clashes + 1)  # Fitness is inverse of number of clashes

    def create_classes(self, chromosome: List[int]) -> List[Class]:
        """Create Class objects from chromosome"""
        classes = []
        gene_idx = 0
        class_id = 1

        for group in self.timetable.groups:
            for course_id in group.course_ids:
                timeslot_idx = chromosome[gene_idx]
                room_id = chromosome[gene_idx + 1]
                instructor_id = chromosome[gene_idx + 2]

                day, start_time, end_time = self.time_slots[timeslot_idx]
                time_range = TimeRange(start_time, end_time)

                classes.append(Class(
                    id=class_id,
                    group_id=group.id,
                    course_id=course_id,
                    instructor_id=instructor_id,
                    room_id=room_id,
                    day=day,
                    time_range=time_range
                ))

                class_id += 1
                gene_idx += 3

        return classes

    def count_clashes(self, classes: List[Class]) -> int:
        """Count number of constraint violations"""
        clashes = 0

        for i, class1 in enumerate(classes):
            # Get related objects
            room = next(r for r in self.timetable.rooms if r.id == class1.room_id)
            group = next(g for g in self.timetable.groups if g.id == class1.group_id)
            course = next(c for c in self.timetable.courses if c.id == class1.course_id)
            instructor = next(instr for instr in self.timetable.instructors if instr.id == class1.instructor_id)

            # Check room capacity
            if group.size > room.capacity:
                clashes += 1

            # Check room type
            if room.room_type != course.required_room_type:
                clashes += 1

            # Check instructor availability
            day_availability = instructor.availability.get(class1.day, [])
            time_conflict = True
            for tr in day_availability:
                if self.time_overlap(tr, class1.time_range):
                    time_conflict = False
                    break
            if time_conflict:
                clashes += 1

            # Check instructor course assignment
            if class1.instructor_id not in course.allowed_instructors:
                clashes += 1

            # Check instructor type for course session type
            if (course.session_type == SessionType.LAB or course.session_type == SessionType.TUTORIAL) and instructor.type != InstructorType.TA:
                clashes += 1
            if course.session_type == SessionType.LECTURE and instructor.type != InstructorType.PROFESSOR:
                clashes += 1

            # Check for conflicts with other classes
            for j in range(i + 1, len(classes)):
                class2 = classes[j]
                if class1.day == class2.day and self.time_overlap(class1.time_range, class2.time_range):
                    # Same room at same time
                    if class1.room_id == class2.room_id:
                        clashes += 1
                    # Same instructor at same time
                    if class1.instructor_id == class2.instructor_id:
                        clashes += 1
                    # Same group at same time
                    if class1.group_id == class2.group_id:
                        clashes += 1

        return clashes

    def time_overlap(self, tr1: TimeRange, tr2: TimeRange) -> bool:
        """Check if two time ranges overlap"""
        format_str = "%H:%M"
        start1 = datetime.datetime.strptime(tr1.start_time, format_str)
        end1 = datetime.datetime.strptime(tr1.end_time, format_str)
        start2 = datetime.datetime.strptime(tr2.start_time, format_str)
        end2 = datetime.datetime.strptime(tr2.end_time, format_str)
        return max(start1, start2) < min(end1, end2)

    def run(self) -> List[Class]:
        """Run the genetic algorithm to find a solution"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]

        for generation in range(self.max_generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.calculate_fitness(ind) for ind in population]

            # Check if we found a perfect solution
            best_fitness = max(fitness_scores)
            if best_fitness == 1.0:
                best_idx = fitness_scores.index(best_fitness)
                return self.create_classes(population[best_idx])

            # Create new population
            new_population = []

            # Elitism
            elite_indices = sorted(range(len(fitness_scores)),
                                   key=lambda i: fitness_scores[i],
                                   reverse=True)[:self.elite_size]
            new_population.extend([population[i] for i in elite_indices])

            # Fill rest of population with crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_select(population, fitness_scores)
                parent2 = self.tournament_select(population, fitness_scores)

                # Crossover
                child = self.uniform_crossover(parent1, parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            population = new_population

        # Return best solution found
        best_idx = fitness_scores.index(max(fitness_scores))
        return self.create_classes(population[best_idx])

    def tournament_select(self, population: List[List[int]],
                          fitness_scores: List[float]) -> List[int]:
        """Select individual using tournament selection"""
        tournament = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]
        winner_idx = tournament[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform uniform crossover between two parents"""
        child = []
        for p1_gene, p2_gene in zip(parent1, parent2):
            child.append(p1_gene if random.random() < 0.5 else p2_gene)
        return child

    def mutate(self, individual: List[int]) -> List[int]:
        """Perform mutation on an individual"""
        mutated = individual.copy()
        for i in range(0, len(mutated), 3):
            if random.random() < self.mutation_rate:
                mutated[i] = random.randint(0, len(self.time_slots) - 1)
            if random.random() < self.mutation_rate:
                mutated[i + 1] = random.choice([room.id for room in self.timetable.rooms])
            if random.random() < self.mutation_rate:
                mutated[i + 2] = random.choice([instr.id for instr in self.timetable.instructors])
        return mutated

def print_schedule(timetable: TimeTable, classes: List[Class]):
    """Print the generated schedule in a readable format"""
    print("\nGenerated Schedule:")
    print("-" * 100)

    # Sort classes by day and time
    sorted_classes = sorted(classes,
                            key=lambda c: (c.day.value,
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
            print(f"\n{current_day.value}")
            print("-" * 100)

        print(f"Time: {class_.time_range.start_time}-{class_.time_range.end_time} | "
              f"Course: {course.code} ({course.name}) [{course.session_type.value}] | "
              f"Group: {group.major} Year {group.year} {group.group_name}-{group.section} | "
              f"Room: {room.number} | "
              f"Instructor: {instructor.name} ({instructor.type.value})")

    # Output the schedule to a JSON file
    schedule_output = []
    for class_ in sorted_classes:
        room = next(r for r in timetable.rooms if r.id == class_.room_id)
        instructor = next(i for i in timetable.instructors if i.id == class_.instructor_id)
        course = next(c for c in timetable.courses if c.id == class_.course_id)
        group = next(g for g in timetable.groups if g.id == class_.group_id)

        schedule_output.append({
            'day': class_.day.value,
            'time': f"{class_.time_range.start_time}-{class_.time_range.end_time}",
            'course_code': course.code,
            'course_name': course.name,
            'session_type': course.session_type.value,
            'group': f"{group.major} Year {group.year} {group.group_name}-{group.section}",
            'room': room.number,
            'instructor': instructor.name,
            'instructor_type': instructor.type.value
        })

    with open('schedule_output.json', 'w') as f:
        json.dump(schedule_output, f, indent=4)

def main():
    # Create timetable from external data
    timetable = TimeTable()

    # Create and run genetic algorithm
    ga = GeneticAlgorithm(timetable)
    solution = ga.run()

    # Print the generated schedule
    print_schedule(timetable, solution)

if __name__ == "__main__":
    main()