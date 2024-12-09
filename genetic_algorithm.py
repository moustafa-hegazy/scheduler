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
        self.elite_size = 100             
        self.tournament_size = 10       
        self.mutation_rate = 0.19     
        self.max_generations = 10000    
        self.time_slots = self.generate_time_slots()
        self.class_mappings = []
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                self.class_mappings.append((group.id, course_id))
    
    def generate_time_slots(self) -> List[Tuple[str, str]]:
        time_slots = []
        time_format = "%H:%M"
        increment = datetime.timedelta(minutes=15)  # 15-minute increments
        days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        for day in days_of_week:
            earliest = None
            latest = None
            # Determine the earliest start and latest end times across all instructors for the day
            for instr in self.timetable.instructors:
                if day in instr.availability:
                    for tr in instr.availability[day]:
                        start = datetime.datetime.strptime(tr.start_time, time_format)
                        end = datetime.datetime.strptime(tr.end_time, time_format)
                        if earliest is None or start < earliest:
                            earliest = start
                        if latest is None or end > latest:
                            latest = end
            if earliest is None or latest is None:
                continue  # No availability for this day
            
            current_time = earliest
            while current_time + increment <= latest:
                start_time_str = current_time.strftime(time_format)
                time_slots.append((day, start_time_str))
                current_time += increment
        return time_slots
    
    def create_individual(self) -> List[int]:
        chromosome = []
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                course = next(c for c in self.timetable.courses if c.id == course_id)
                instructor_id = random.choice(course.allowed_instructors)
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)
                
                # Calculate duration for the course
                duration_hours = course.duration
                duration = datetime.timedelta(hours=duration_hours)
                
                available_time_slots = []
                for idx, (day, start_time_str) in enumerate(self.time_slots):
                    day_availability = instructor.availability.get(day, [])
                    class_start = datetime.datetime.strptime(start_time_str, "%H:%M")
                    class_end = class_start + duration
                    # Check if the class fits within any available time range
                    fits = False
                    for tr in day_availability:
                        available_start = datetime.datetime.strptime(tr.start_time, "%H:%M")
                        available_end = datetime.datetime.strptime(tr.end_time, "%H:%M")
                        if available_start <= class_start and class_end <= available_end:
                            fits = True
                            break
                    if fits:
                        available_time_slots.append(idx)
                
                if not available_time_slots:
                    timeslot_idx = random.randint(0, len(self.time_slots) - 1)
                else:
                    timeslot_idx = random.choice(available_time_slots)
                
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if not suitable_rooms:
                    room_id = random.choice([room.id for room in self.timetable.rooms])
                else:
                    room_id = random.choice([room.id for room in suitable_rooms])
                chromosome.extend([timeslot_idx, room_id, instructor_id])
        return chromosome
    
    def calculate_total_possible_clashes(self) -> int:
        num_classes = sum(len(group.course_ids) for group in self.timetable.groups)
        max_clashes = num_classes * (num_classes - 1) // 2 * 3 
        return max_clashes
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        classes = self.create_classes(chromosome)
        clashes = self.count_clashes(classes)
        total_possible_clashes = self.calculate_total_possible_clashes()
        fitness = (total_possible_clashes - clashes) / total_possible_clashes
        return fitness
    
    def create_classes(self, chromosome: List[int]) -> List[Class]:
        classes = []
        gene_idx = 0
        class_id = 1  
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                course = next(c for c in self.timetable.courses if c.id == course_id)
                duration_hours = course.duration
                duration = datetime.timedelta(hours=duration_hours)
                
                timeslot_idx = chromosome[gene_idx]
                room_id = chromosome[gene_idx + 1]
                instructor_id = chromosome[gene_idx + 2]
                
                day, start_time_str = self.time_slots[timeslot_idx]
                time_format = "%H:%M"
                start_time = datetime.datetime.strptime(start_time_str, time_format)
                end_time = start_time + duration
                end_time_str = end_time.strftime(time_format)
                
                classes.append(Class(
                    id=class_id,
                    group_id=group.id,
                    course_id=course_id,
                    instructor_id=instructor_id,
                    room_id=room_id,
                    day=day,
                    time_range=TimeRange(start_time_str, end_time_str)
                ))
                class_id += 1
                gene_idx += 3  
        return classes
    
    def count_clashes(self, classes: List[Class]) -> int:
        clashes = 0 
        for i, class1 in enumerate(classes):
            room = next(r for r in self.timetable.rooms if r.id == class1.room_id)
            group = next(g for g in self.timetable.groups if g.id == class1.group_id)
            course = next(c for c in self.timetable.courses if c.id == class1.course_id)
            instructor = next(instr for instr in self.timetable.instructors if instr.id == class1.instructor_id)
            
            # Room capacity clash
            if group.size > room.capacity:
                clashes += 1 
            
            # Room type clash
            if room.room_type != course.required_room_type:
                clashes += 1 
            
            # Instructor availability clash
            day_availability = instructor.availability.get(class1.day, [])
            time_conflict = True  
            for tr in day_availability:
                available_start = datetime.datetime.strptime(tr.start_time, "%H:%M")
                available_end = datetime.datetime.strptime(tr.end_time, "%H:%M")
                class_start = datetime.datetime.strptime(class1.time_range.start_time, "%H:%M")
                class_end = datetime.datetime.strptime(class1.time_range.end_time, "%H:%M")
                if available_start <= class_start and class_end <= available_end:
                    time_conflict = False             
                    break
            if time_conflict:
                clashes += 1  
            
            # Instructor not allowed
            if class1.instructor_id not in course.allowed_instructors:
                clashes += 1  
            
            # Instructor type clash based on session type
            if (course.session_type in ["Lab", "Tutorial"]) and instructor.type != "TA":
                clashes += 1 
            if course.session_type == "Lecture" and instructor.type != "Professor":
                clashes += 1 
            
            # Overlapping classes clash
            for j in range(i + 1, len(classes)):    
                class2 = classes[j]
                if class1.day == class2.day and self.time_overlap(class1.time_range, class2.time_range):
                    if class1.room_id == class2.room_id:
                        clashes += 1
                    if class1.instructor_id == class2.instructor_id:
                        clashes += 1
                    if class1.group_id == class2.group_id:
                        clashes += 1
        return clashes
    
    def time_overlap(self, tr1: TimeRange, tr2: TimeRange) -> bool:
        format_str = "%H:%M" 
        start1 = datetime.datetime.strptime(tr1.start_time, format_str)
        end1 = datetime.datetime.strptime(tr1.end_time, format_str)
        start2 = datetime.datetime.strptime(tr2.start_time, format_str)
        end2 = datetime.datetime.strptime(tr2.end_time, format_str)
        return max(start1, start2) < min(end1, end2)
    
    def run(self) -> List[Class]:
        with Pool(processes=12) as pool:
            population = pool.map(lambda _: self.create_individual(), range(self.population_size))
        for generation in range(self.max_generations):
            with Pool(processes=12) as pool:
                fitness_scores = pool.map(self.calculate_fitness, population)
            best_fitness = max(fitness_scores)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            if best_fitness == 1.0:
                best_idx = fitness_scores.index(best_fitness)
                return self.create_classes(population[best_idx])
            new_population = []
            elite_indices = sorted(range(len(fitness_scores)),
                                   key=lambda i: fitness_scores[i],
                                   reverse=True)[:self.elite_size]
            new_population.extend([population[i] for i in elite_indices])
            while len(new_population) < self.population_size:
                parent1 = self.tournament_select(population, fitness_scores)
                parent2 = self.tournament_select(population, fitness_scores)
                child = self.uniform_crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)
            population = new_population
        best_idx = fitness_scores.index(max(fitness_scores))
        return self.create_classes(population[best_idx])
    
    def tournament_select(self, population: List[List[int]],
                          fitness_scores: List[float]) -> List[int]:
        tournament = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]
        winner_idx = tournament[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]
    
    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        child = []
        for p1_gene, p2_gene in zip(parent1, parent2):
            child.append(p1_gene if random.random() < 0.5 else p2_gene)
        return child
    
    def mutate(self, individual: List[int]) -> List[int]:
        mutated = individual.copy()
        gene_length = 3  
        num_classes = len(mutated) // gene_length                            
        for class_idx in range(num_classes):
            idx = class_idx * gene_length  
            group_id, course_id = self.class_mappings[class_idx]
            group = next(g for g in self.timetable.groups if g.id == group_id)
            course = next(c for c in self.timetable.courses if c.id == course_id)
            
            # Calculate duration for the course
            duration_hours = course.duration
            duration = datetime.timedelta(hours=duration_hours)
            
            # Mutation for time slot
            if random.random() < self.mutation_rate:
                instructor_id = mutated[idx + 2]
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)
                available_time_slots = []
                for idx_ts, (day, start_time_str) in enumerate(self.time_slots):
                    day_availability = instructor.availability.get(day, [])
                    class_start = datetime.datetime.strptime(start_time_str, "%H:%M")
                    class_end = class_start + duration
                    # Check if the class fits within any available time range
                    fits = False
                    for tr in day_availability:
                        available_start = datetime.datetime.strptime(tr.start_time, "%H:%M")
                        available_end = datetime.datetime.strptime(tr.end_time, "%H:%M")
                        if available_start <= class_start and class_end <= available_end:
                            fits = True
                            break
                    if fits:
                        available_time_slots.append(idx_ts)
                if available_time_slots:        
                    mutated[idx] = random.choice(available_time_slots)
            
            # Mutation for room
            if random.random() < self.mutation_rate:
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if suitable_rooms:
                    mutated[idx + 1] = random.choice([room.id for room in suitable_rooms])
            
            # Mutation for instructor
            if random.random() < self.mutation_rate:
                mutated[idx + 2] = random.choice(course.allowed_instructors)
        return mutated