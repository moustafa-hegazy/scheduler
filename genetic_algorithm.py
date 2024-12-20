# genetic_algorithm.py

from data_models import TimeRange, Class
from timetable import TimeTable
from typing import List, Dict
import random
from pathos.multiprocessing import ProcessingPool as Pool


class GeneticAlgorithm:
    def __init__(self, timetable: TimeTable):
        self.timetable = timetable
        self.population_size = 100000
        self.elite_size = 100
        self.tournament_size = 15
        self.mutation_rate = 0.20
        self.max_generations = 500

        self.max_no_improve = 30         

        self.days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        self.slots_per_day = 8
        self.time_slots = [(day, s) for day in self.days_of_week for s in range(self.slots_per_day)]

        # class_mappings is a list of tuples (group_id, course_id) for each class that must be scheduled
        self.class_mappings = []
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                self.class_mappings.append((group.id, course_id))

    def run(self) -> List[Class]:
        # Initialize population
        with Pool(processes=12) as pool:
            population = pool.map(lambda _: self.create_individual(), range(self.population_size))

        # Initialize variables to track fitness improvement
        best_fitness_overall = -1.0
        no_improve_count = 0
        max_no_improve = self.max_no_improve  # Terminate if no improvement in 10 consecutive generations

        # Evolve population
        for generation in range(self.max_generations):
            with Pool(processes=12) as pool:
                fitness_scores = pool.map(self.calculate_fitness, population)
            current_best_fitness = max(fitness_scores)
            print(f"Generation {generation}: Best Fitness = {current_best_fitness}")

            # Check for improvement
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                no_improve_count = 0  # Reset counter if improvement occurs
            else:
                no_improve_count += 1  # Increment counter if no improvement

            # Terminate early if no improvement for max_no_improve generations
            if no_improve_count >= max_no_improve:
                print(f"No improvement in {max_no_improve} consecutive generations. Terminating early.")
                break

            if current_best_fitness == 1.0:
                # Perfect solution found
                best_idx = fitness_scores.index(current_best_fitness)
                final_classes = self.create_classes(population[best_idx])
                # Optional: final_classes = self.repair_solution(final_classes)
                return final_classes

            # Selection + Reproduction
            new_population = []
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            new_population.extend([population[i] for i in elite_indices])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_select(population, fitness_scores)
                parent2 = self.tournament_select(population, fitness_scores)
                child = self.uniform_crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)

            population = new_population

        # If we reach here, no perfect solution was found
        # Take best solution and attempt repair
        with Pool(processes=12) as pool:
            fitness_scores = pool.map(self.calculate_fitness, population)
        best_idx = fitness_scores.index(max(fitness_scores))
        final_classes = self.create_classes(population[best_idx])
        final_classes = self.repair_solution(final_classes)
        return final_classes

    def create_individual(self) -> List[int]:
        # Create a random feasible-ish individual
        instructors = {i.id: i for i in self.timetable.instructors}
        courses = {c.id: c for c in self.timetable.courses}
        groups = {g.id: g for g in self.timetable.groups}

        chromosome = []
        for (group_id, course_id) in self.class_mappings:
            course = courses[course_id]
            group = groups[group_id]
            duration_slots = course.duration

            instructor_id = random.choice(course.allowed_instructors)
            instructor = instructors[instructor_id]

            feasible_indices = []
            for idx, (day, slot_index) in enumerate(self.time_slots):
                if self.can_instructor_teach(instructor, day, slot_index, duration_slots):
                    if slot_index + duration_slots <= self.slots_per_day:
                        feasible_indices.append(idx)

            if not feasible_indices:
                # No feasible slot found, pick random timeslot
                timeslot_idx = random.randint(0, len(self.time_slots)-1)
            else:
                timeslot_idx = random.choice(feasible_indices)

            room_id = self.select_most_efficient_room(course.required_room_type, group.size)
            chromosome.extend([timeslot_idx, room_id, instructor_id])
        return chromosome

    def can_instructor_teach(self, instructor, day: str, start_slot: int, duration_slots: int) -> bool:
        if day not in instructor.availability_slots:
            return False
        required_slots = range(start_slot, start_slot + duration_slots)
        for s in required_slots:
            if s not in instructor.availability_slots[day]:
                return False
        return True

    def select_most_efficient_room(self, course_required_type: str, group_size: int) -> int:
        suitable_rooms = [
            room for room in self.timetable.rooms
            if room.room_type == course_required_type and room.capacity >= group_size
        ]
        if not suitable_rooms:
            return random.choice([r.id for r in self.timetable.rooms])
        suitable_rooms_sorted = sorted(suitable_rooms, key=lambda r: r.capacity)
        min_capacity = suitable_rooms_sorted[0].capacity
        min_capacity_rooms = [r for r in suitable_rooms_sorted if r.capacity == min_capacity]
        return random.choice([r.id for r in min_capacity_rooms])

    def calculate_total_possible_clashes(self) -> int:
        num_classes = len(self.class_mappings)
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
        for (group_id, course_id) in self.class_mappings:
            course = next(c for c in self.timetable.courses if c.id == course_id)
            duration_slots = course.duration

            timeslot_idx = chromosome[gene_idx]
            room_id = chromosome[gene_idx+1]
            instructor_id = chromosome[gene_idx+2]

            day, start_slot = self.time_slots[timeslot_idx]
            end_slot = start_slot + duration_slots

            start_time_str = f"Slot {start_slot+1}"
            end_time_str = f"Slot {end_slot+1}"

            classes.append(Class(
                id=class_id,
                group_id=group_id,
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
        groups = {g.id: g for g in self.timetable.groups}
        courses = {c.id: c for c in self.timetable.courses}
        rooms = {r.id: r for r in self.timetable.rooms}
        instructors = {i.id: i for i in self.timetable.instructors}

        def get_slot_range(c):
            start_slot = int(c.time_range.start_time.split()[1]) - 1
            end_slot = int(c.time_range.end_time.split()[1]) - 1
            return range(start_slot, end_slot)

        clashes = 0
        for i, c1 in enumerate(classes):
            g1 = groups[c1.group_id]
            co1 = courses[c1.course_id]
            r1 = rooms[c1.room_id]
            in1 = instructors[c1.instructor_id]
            s1 = get_slot_range(c1)

            # Capacity
            if g1.size > r1.capacity:
                clashes += 1
            # Room type
            if r1.room_type != co1.required_room_type:
                clashes += 1
            # Instructor availability
            for sl in s1:
                if c1.day not in in1.availability_slots or sl not in in1.availability_slots[c1.day]:
                    clashes += 1
                    break
            # Instructor allowed
            if c1.instructor_id not in co1.allowed_instructors:
                clashes += 1
            # Instructor type
            if co1.session_type in ["Lab", "Tutorial"] and in1.type != "TA":
                clashes += 1
            if co1.session_type == "Lecture" and in1.type != "Professor":
                clashes += 1

            # Overlapping classes
            for j in range(i+1, len(classes)):
                c2 = classes[j]
                if c1.day == c2.day:
                    s2 = get_slot_range(c2)
                    overlap = any(s in s1 for s in s2)
                    if overlap:
                        # Same room
                        if c1.room_id == c2.room_id:
                            clashes += 1
                        # Same instructor
                        if c1.instructor_id == c2.instructor_id:
                            clashes += 1
                        # Same group
                        if c1.group_id == c2.group_id:
                            clashes += 1

        # Additional constraint: main-subgroup overlap
        group_name_to_main = {}
        group_name_to_sub = {}
        for cl in classes:
            grp = groups[cl.group_id]
            if grp.section == "0":
                group_name_to_main.setdefault(grp.group_name, []).append(cl)
            else:
                group_name_to_sub.setdefault(grp.group_name, []).append(cl)

        for gname, main_cls in group_name_to_main.items():
            sub_cls = group_name_to_sub.get(gname, [])
            for mcl in main_cls:
                m_s = get_slot_range(mcl)
                for scl in sub_cls:
                    if scl.day == mcl.day:
                        s_s = get_slot_range(scl)
                        if any(s in m_s for s in s_s):
                            clashes += 1

        return clashes

    def repair_solution(self, classes: List[Class]) -> List[Class]:
        # Attempt to fix small remaining clashes by local search
        chromosome = self.classes_to_chromosome(classes)
        initial_fitness = self.calculate_fitness(chromosome)
        if initial_fitness == 1.0:
            return classes

        max_attempts = 100
        for attempt in range(max_attempts):
            current_classes = self.create_classes(chromosome)
            clashes_info = self.identify_clashes(current_classes)
            if not clashes_info:
                # No clashes left
                break

            # Pick a random class causing clashes
            problematic_class_id = random.choice(list(clashes_info.keys()))
            improved = self.try_to_repair_class(chromosome, problematic_class_id)
            if improved:
                new_fitness = self.calculate_fitness(chromosome)
                if new_fitness == 1.0:
                    break

        return self.create_classes(chromosome)

    def classes_to_chromosome(self, classes: List[Class]) -> List[int]:
        gene_length = 3
        chromosome = [0]*(len(self.class_mappings)*gene_length)
        for i, (group_id, course_id) in enumerate(self.class_mappings):
            cl = next(c for c in classes if c.group_id == group_id and c.course_id == course_id)
            day = cl.day
            start_slot = int(cl.time_range.start_time.split()[1]) - 1
            timeslot_idx = self.time_slots.index((day, start_slot))
            idx = i*3
            chromosome[idx] = timeslot_idx
            chromosome[idx+1] = cl.room_id
            chromosome[idx+2] = cl.instructor_id
        return chromosome

    def identify_clashes(self, classes: List[Class]) -> Dict[int, List[str]]:
        groups = {g.id: g for g in self.timetable.groups}
        courses = {c.id: c for c in self.timetable.courses}
        rooms = {r.id: r for r in self.timetable.rooms}
        instructors = {i.id: i for i in self.timetable.instructors}

        def get_slot_range(c):
            start_slot = int(c.time_range.start_time.split()[1]) - 1
            end_slot = int(c.time_range.end_time.split()[1]) - 1
            return range(start_slot, end_slot)

        clashes_info = {}
        # Helper to record clash
        def add_clash(c_id, reason):
            if c_id not in clashes_info:
                clashes_info[c_id] = []
            clashes_info[c_id].append(reason)

        for i, c1 in enumerate(classes):
            g1 = groups[c1.group_id]
            co1 = courses[c1.course_id]
            r1 = rooms[c1.room_id]
            in1 = instructors[c1.instructor_id]
            s1 = get_slot_range(c1)

            if g1.size > r1.capacity:
                add_clash(c1.id, "capacity")
            if r1.room_type != co1.required_room_type:
                add_clash(c1.id, "room_type")
            for sl in s1:
                if c1.day not in in1.availability_slots or sl not in in1.availability_slots[c1.day]:
                    add_clash(c1.id, "instructor_availability")
                    break
            if c1.instructor_id not in co1.allowed_instructors:
                add_clash(c1.id, "instructor_not_allowed")
            if co1.session_type in ["Lab", "Tutorial"] and in1.type != "TA":
                add_clash(c1.id, "instructor_type_lab_tutorial")
            if co1.session_type == "Lecture" and in1.type != "Professor":
                add_clash(c1.id, "instructor_type_lecture")

            for j in range(i+1, len(classes)):
                c2 = classes[j]
                if c1.day == c2.day:
                    s2 = get_slot_range(c2)
                    overlap = any(s in s1 for s in s2)
                    if overlap:
                        if c1.room_id == c2.room_id:
                            add_clash(c1.id, "overlap_room")
                            add_clash(c2.id, "overlap_room")
                        if c1.instructor_id == c2.instructor_id:
                            add_clash(c1.id, "overlap_instructor")
                            add_clash(c2.id, "overlap_instructor")
                        if c1.group_id == c2.group_id:
                            add_clash(c1.id, "overlap_group")
                            add_clash(c2.id, "overlap_group")

        # Additional constraint: main-subgroup overlap
        group_name_to_main = {}
        group_name_to_sub = {}
        for cl in classes:
            grp = groups[cl.group_id]
            if grp.section == "0":
                group_name_to_main.setdefault(grp.group_name, []).append(cl)
            else:
                group_name_to_sub.setdefault(grp.group_name, []).append(cl)

        for gname, main_cls in group_name_to_main.items():
            sub_cls = group_name_to_sub.get(gname, [])
            for mcl in main_cls:
                m_s = get_slot_range(mcl)
                for scl in sub_cls:
                    if scl.day == mcl.day:
                        s_s = get_slot_range(scl)
                        if any(s in m_s for s in s_s):
                            add_clash(mcl.id, "main_subgroup_overlap")
                            add_clash(scl.id, "main_subgroup_overlap")

        return clashes_info

    def try_to_repair_class(self, chromosome: List[int], class_id: int) -> bool:
        # Attempt to fix a single class that causes clashes by adjusting its timeslot/instructor/room
        class_idx = class_id - 1
        gene_idx = class_idx * 3
        original = (chromosome[gene_idx], chromosome[gene_idx+1], chromosome[gene_idx+2])

        group_id, course_id = self.class_mappings[class_idx]
        course = next(c for c in self.timetable.courses if c.id == course_id)
        group = next(g for g in self.timetable.groups if g.id == group_id)
        duration_slots = course.duration

        instructors = {i.id: i for i in self.timetable.instructors}

        best_fitness = self.calculate_fitness(chromosome)
        improved = False

        for new_instructor_id in course.allowed_instructors:
            in_obj = instructors[new_instructor_id]
            feasible_slots = []
            for t_idx, (day, s_idx) in enumerate(self.time_slots):
                if self.can_instructor_teach(in_obj, day, s_idx, duration_slots):
                    if s_idx + duration_slots <= self.slots_per_day:
                        feasible_slots.append(t_idx)
            for new_t_idx in feasible_slots:
                new_room_id = self.select_most_efficient_room(course.required_room_type, group.size)
                chromosome[gene_idx] = new_t_idx
                chromosome[gene_idx+1] = new_room_id
                chromosome[gene_idx+2] = new_instructor_id

                new_fit = self.calculate_fitness(chromosome)
                if new_fit > best_fitness:
                    best_fitness = new_fit
                    improved = True
                    return True
                else:
                    # revert
                    chromosome[gene_idx] = original[0]
                    chromosome[gene_idx+1] = original[1]
                    chromosome[gene_idx+2] = original[2]

        return improved

    def tournament_select(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
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
        instructors = {i.id: i for i in self.timetable.instructors}
        courses = {c.id: c for c in self.timetable.courses}
        groups = {g.id: g for g in self.timetable.groups}

        for class_idx in range(num_classes):
            idx = class_idx * gene_length
            group_id, course_id = self.class_mappings[class_idx]
            group = groups[group_id]
            course = courses[course_id]
            duration_slots = course.duration

            if random.random() < self.mutation_rate:
                instructor_id = mutated[idx+2]
                instructor = instructors[instructor_id]
                feasible_indices = []
                for t_idx, (day, s_idx) in enumerate(self.time_slots):
                    if self.can_instructor_teach(instructor, day, s_idx, duration_slots):
                        if s_idx + duration_slots <= self.slots_per_day:
                            feasible_indices.append(t_idx)
                if feasible_indices:
                    mutated[idx] = random.choice(feasible_indices)

            if random.random() < self.mutation_rate:
                mutated[idx+1] = self.select_most_efficient_room(course.required_room_type, group.size)

            if random.random() < self.mutation_rate:
                mutated[idx+2] = random.choice(course.allowed_instructors)

        return mutated