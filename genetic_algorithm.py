# genetic_algorithm.py

from data_models import TimeRange, Class
from timetable import TimeTable
from typing import List, Tuple
import datetime
import random
from pathos.multiprocessing import ProcessingPool as Pool

class GeneticAlgorithm:
    def __init__(self, timetable: TimeTable):
        self.timetable = timetable
        self.population_size = 1000
        self.elite_size = 25
        self.tournament_size = 10
        self.mutation_rate = 0.15
        self.max_generations = 10000

        # Generate possible time slots based on instructors' availability and standard class durations
        self.time_slots = self.generate_time_slots()

        # Map class index to (group_id, course_id)
        self.class_mappings = []
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                self.class_mappings.append((group.id, course_id))

    def generate_time_slots(self) -> List[Tuple[str, str, str]]:
        """Generate possible time slots based on the standard class durations"""
        time_slots = []
        class_duration = datetime.timedelta(hours=2)  # Assuming classes are 2 hours
        time_format = "%H:%M"

        days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


        for day in days_of_week:
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
                # Increment by class duration to prevent overlapping and allowing admin to adjust the break time later on their own
                current_time += class_duration

        return time_slots

    def create_individual(self) -> List[int]:
        """Creates a random valid individual (chromosome)"""
        chromosome = []

        for group in self.timetable.groups:
            for course_id in group.course_ids:
                course = next(c for c in self.timetable.courses if c.id == course_id)

                # Select an instructor from allowed instructors
                instructor_id = random.choice(course.allowed_instructors)
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)

                # Get available time slots for instructor
                available_time_slots = []
                for idx, (day, start_time, end_time) in enumerate(self.time_slots):
                    if day in instructor.availability:
                        for tr in instructor.availability[day]:
                            timeslot_tr = TimeRange(start_time, end_time)
                            if self.time_overlap(tr, timeslot_tr):
                                available_time_slots.append(idx)
                                break  # Found a matching time slot

                if not available_time_slots:
                    # Instructor not available at any time slot, pick any time slot
                    timeslot_idx = random.randint(0, len(self.time_slots) - 1)
                else:
                    timeslot_idx = random.choice(available_time_slots)

                # Select a room that meets the course's room type and has enough capacity
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if not suitable_rooms:
                    # No suitable room, pick any room
                    room_id = random.choice([room.id for room in self.timetable.rooms])
                else:
                    room_id = random.choice([room.id for room in suitable_rooms])

                chromosome.extend([timeslot_idx, room_id, instructor_id])

        return chromosome

    def calculate_total_possible_clashes(self) -> int:
        """Calculate the maximum possible number of clashes."""
        num_classes = sum(len(group.course_ids) for group in self.timetable.groups)

        # Maximum possible clashes between classes in terms of room, instructor, and group
        # For each pair of classes, there are up to 3 possible clashes (room, instructor, group)
        max_clashes = num_classes * (num_classes - 1) // 2 * 3
        return max_clashes

    def calculate_fitness(self, chromosome: List[int]) -> float:
        """Calculate fitness of a chromosome based on constraint violations"""
        classes = self.create_classes(chromosome)
        clashes = self.count_clashes(classes)
        total_possible_clashes = self.calculate_total_possible_clashes()
        # Fitness is based on the percentage of clashes avoided
        fitness = (total_possible_clashes - clashes) / total_possible_clashes
        return fitness

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
            if (course.session_type in ["Lab", "Tutorial"]) and instructor.type != "TA":
                clashes += 1
            if course.session_type == "Lecture" and instructor.type != "Professor":
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
        with Pool(processes=12) as pool:
            population = pool.map(lambda _: self.create_individual(), range(self.population_size))

        for generation in range(self.max_generations):
            # Calculate fitness for all individuals
            with Pool(processes=12) as pool:
                fitness_scores = pool.map(self.calculate_fitness, population)

            # Check if we found a perfect solution
            best_fitness = max(fitness_scores)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
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
        gene_length = 3
        num_classes = len(mutated) // gene_length
        for class_idx in range(num_classes):
            idx = class_idx * gene_length

            group_id, course_id = self.class_mappings[class_idx]
            group = next(g for g in self.timetable.groups if g.id == group_id)
            course = next(c for c in self.timetable.courses if c.id == course_id)

            if random.random() < self.mutation_rate:
                # Mutate timeslot
                instructor_id = mutated[idx + 2]
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)
                available_time_slots = []
                for idx_ts, (day, start_time, end_time) in enumerate(self.time_slots):
                    if day in instructor.availability:
                        for tr in instructor.availability[day]:
                            timeslot_tr = TimeRange(start_time, end_time)
                            if self.time_overlap(tr, timeslot_tr):
                                available_time_slots.append(idx_ts)
                                break  # Found a matching time slot
                if available_time_slots:
                    mutated[idx] = random.choice(available_time_slots)

            if random.random() < self.mutation_rate:
                # Mutate room
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if suitable_rooms:
                    mutated[idx + 1] = random.choice([room.id for room in suitable_rooms])

            if random.random() < self.mutation_rate:
                # Mutate instructor
                mutated[idx + 2] = random.choice(course.allowed_instructors)

        return mutated
