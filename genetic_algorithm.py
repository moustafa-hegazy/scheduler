from data_models import TimeRange, Class
from timetable import TimeTable
from typing import List, Tuple
import datetime
import random
from pathos.multiprocessing import ProcessingPool as Pool

class GeneticAlgorithm:
    """
    A class to generate an optimized timetable using a genetic algorithm.
    كلاس عشان نولد جدول محاضرات مثالي باستخدام الجينيتك الجوريذم

    The genetic algorithm schedules classes by assigning time slots, rooms, and instructors
    while minimizing conflicts and adhering to various constraints like room capacity,
    instructor availability, and course requirements.
    الجينيتك الجوريذم بعمل جدول المحاضرات عن طريق تخصيص مواعيد وقاعات ودكاترة،
    وبنحاول نقلل التعارضات ونلتزم بالقيود زي سعة القاعة، توفر الدكتور، ومتطلبات الكورس.

    Attributes:
        timetable (TimeTable): The timetable data containing courses, instructors, rooms, and groups.
        جدول المحاضرات اللي فيه كل البيانات عن الكورسات، الدكاترة، القاعات، والمجموعات.
        population_size (int): The number of individuals in each generation.
        عدد الافراد في كل جيل.
        elite_size (int): The number of top individuals preserved in each generation.
        عدد الافراد المميزين اللي بنحتفظ بيهم في كل جيل.
        tournament_size (int): The number of individuals participating in tournament selection.
        عدد الافراد اللي بيشاركوا في التورنمنت سيليكشن.
        mutation_rate (float): The probability of mutating an individual.
        نسبة حدوث الميوتاشن للفرد.
        max_generations (int): The maximum number of generations to run the algorithm.
        اقصى عدد من الاجيال اللي هنشغل عليها الالجوريذم.
        time_slots (List[Tuple[str, str, str]]): Possible time slots based on availability and class durations.
        قائمة بكل الاوقات المتاحة بناء على توفر الدكاترة ومدد المحاضرات.
        class_mappings (List[Tuple[str, str]]): Mapping of class index to group and course IDs.
        خريطة بتربط كل محاضرة بالمجموعة والكورس الخاصين بيه.
    """

    def __init__(self, timetable: TimeTable):
        self.timetable = timetable
        self.population_size = 1000     # Total number of individuals (possible schedules) in each generation
                                        # عدد الأفراد (الجداول الممكنة) في كل جيل
        self.elite_size = 25            # Number of top individuals to carry over to the next generation unchanged
                                        # عدد الأفراد المميزين اللي هننقلهم للجيل اللي بعده زي ما هما
        self.tournament_size = 10       # Number of individuals competing in tournament selection
                                        # عدد الأفراد اللي هيتنافسوا في التورنمنت سيليكشن
        self.mutation_rate = 0.19       # Probability that a gene will mutate
                                        # نسبة احتمالية إن الجين يحصل له ميوتاشن
        self.max_generations = 10000    # Maximum number of generations to evolve
                                        # أقصى عدد من الأجيال للتطور

       # Generate all possible time slots based on instructor availability and class durations
        # بنولد كل الأوقات المتاحة بناءً على توفر الدكاترة ومدد المحاضرات
        self.time_slots = self.generate_time_slots()

        # Map each class to a unique index, linking group IDs and course IDs
        # بنربط كل محاضرة برقم معين عشان نقدر نتابع الجينات الخاصة بيه
        self.class_mappings = []
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                self.class_mappings.append((group.id, course_id))

    def generate_time_slots(self) -> List[Tuple[str, str, str]]:
        """
        Generate possible time slots based on the standard class durations.
        بنولد الاوقات المتاحة بناء على مدة المحاضرة العادية

        Returns:
            List[Tuple[str, str, str]]: A list of tuples containing day, start time, and end time.
            قائمة بالتواريخ والاوقات اللي بنقدر نحط فيها المحاضرات
        """
        time_slots = []
        class_duration = datetime.timedelta(hours=2)  # Assuming each class lasts 2 hours
                                                      # بنفترض إن كل محاضرة مدتها ساعتين
        time_format = "%H:%M"# Format for parsing and formatting time strings
                             # الفورمات اللي هنستخدمه في الاوقات

        days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Loop through each day to generate time slots
        # بنلف على كل يوم عشان نولد الاوقات المتاحة فيه
        for day in days_of_week:
             # Collect the earliest start time and latest end time from all instructors for this day
             # بنجمع ابدر وقت بداية واخر وقت انتهاء من كل الدكاترة في اليوم ده
            times = []
            for instr in self.timetable.instructors:
                if day in instr.availability:
                    for tr in instr.availability[day]:
                        # Convert start and end times to datetime objects
                        start = datetime.datetime.strptime(tr.start_time, time_format)
                        end = datetime.datetime.strptime(tr.end_time, time_format)
                        times.append((start, end))

            if not times:
                # If no instructors are available on this day, skip to the next day
                # لو مفيش دكاترة متاحة في اليوم ده، بنعديه
                continue


            # Find the earliest start time and latest end time among all instructors
            # بنلاقي ابدر وقت بدء وآخر وقت انتهاء بين كل الدكاترة
            earliest = min(start for start, end in times)
            latest = max(end for start, end in times)


            # Generate time slots from the earliest start time to the latest end time
            # بنولد الاوقات المتاحة من ابكر وقت لآخر وقت
            current_time = earliest
            while current_time + class_duration <= latest:
                # Format the start and end times as strings
                start_time_str = current_time.strftime(time_format)
                end_time = current_time + class_duration
                end_time_str = end_time.strftime(time_format)

                # Add the time slot to the list (day, start time, end time)
                # بنضيف الوقت المتاح للقائمة (اليوم، وقت البداية، وقت الانتهاء)
                time_slots.append((day, start_time_str, end_time_str))

                # Move to the next time slot by adding the class duration
                # بننتقل للوقت اللي بعده بإضافة مدة المحاضرة
                current_time += class_duration

        return time_slots

    def create_individual(self) -> List[int]:
        """
        Creates a random valid individual (chromosome) for the genetic algorithm.
        بنعمل فرد عشوائي صالح (كروموسوم) للجينيتك الجوريذم

        Each individual represents a possible timetable schedule.
        كل فرد بيمثل جدول محاضرات ممكن

        The chromosome is a list where each set of three consecutive genes represents:
        - Time slot index
        - Room ID
        - Instructor ID
        الكروموسوم هو قائمة وكل 3 جينات بيمثلوا:
        - رقم الوقت
        - رقم القاعة
        - رقم الدكتور

        Returns:
            List[int]: A list representing the chromosome.
            قائمة بتمثل الكروموسوم
        """
        chromosome = []

        # Loop through each group and their associated courses
        # بنلف على كل مجموعة والكورسات الخاصة بيها
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                # Get the course object based on the course ID
                # بنجيب الكورس بناء على رقمه
                course = next(c for c in self.timetable.courses if c.id == course_id)

                # Randomly select an instructor from the list of allowed instructors for this course
                # بنختار دكتور عشوائي من الدكاترة المسموحين للكورس ده
                instructor_id = random.choice(course.allowed_instructors)
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)

                # Find all time slots where the instructor is available
                # بنلاقي كل الاوقات اللي الدكتور متاح فيها
                available_time_slots = []
                for idx, (day, start_time, end_time) in enumerate(self.time_slots):
                    if day in instructor.availability:
                        for tr in instructor.availability[day]:
                            timeslot_tr = TimeRange(start_time, end_time)
                            if self.time_overlap(tr, timeslot_tr):
                                # If the instructor is available during this time slot, add it to the list
                                # لو الدكتور متاح في الوقت ده، بنضيفه للقائمة
                                available_time_slots.append(idx)
                                break  # Found a matching time slot
                                       #خلاص عمالنا الي عايزينه مش محتاجين نكمل

                if not available_time_slots:
                    # If the instructor is not available at any time slot, select a random time slot
                    # لو الدكتور مش متاح في اي وقت، بنختار وقت عشوائي
                    timeslot_idx = random.randint(0, len(self.time_slots) - 1)
                else:
                    # Randomly select one of the available time slots
                    # بنختار وقت متاح عشوائيًا
                    timeslot_idx = random.choice(available_time_slots)

                # Find suitable rooms that match the course's room type and have enough capacity
                # بنلاقي القاعات المناسبة اللي تتماشى مع نوع القاعة المطلوب وسعتها
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if not suitable_rooms:
                    # If no suitable rooms are found, select a random room
                    # لو مفيش قاعات مناسبة، بنختار قاعة عشوائيًا
                    room_id = random.choice([room.id for room in self.timetable.rooms])
                else:
                    # Randomly select one of the suitable rooms
                    # بنختار قاعة مناسبة عشوائيًا
                    room_id = random.choice([room.id for room in suitable_rooms])
                # Append the genes (time slot index, room ID, instructor ID) to the chromosome
                # بنضيف الجينات (رقم الوقت، رقم القاعة، رقم الدكتور) للكروموسوم
                chromosome.extend([timeslot_idx, room_id, instructor_id])

        return chromosome

    def calculate_total_possible_clashes(self) -> int:
        """
        Calculate the maximum possible number of clashes.
        بنحسب اقصى عدد ممكن من التعارضات

        This is used to normalize the fitness score.
        ده بيستخدم عشان نحسب الفتنس بشكل نسبي

        Returns:
            int: The maximum number of possible clashes between classes.
            اقصى عدد من التعارضات الممكنة بين المحاضرات
        """
        # Count the total number of classes to be scheduled
        # بنعد إجمالي عدد المحاضرات اللي هنسجلها
        num_classes = sum(len(group.course_ids) for group in self.timetable.groups)

        # Calculate the maximum number of clashes:
        # For each pair of classes, there could be a clash in room, instructor, or group
        # بنحسب اقصى عدد من التعارضات بين كل زوج من المحاضرات
        max_clashes = num_classes * (num_classes - 1) // 2 * 3 # Combination formula n(n-1)/2 times 3 types of clashes
        return max_clashes

    def calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Calculate the fitness of a chromosome based on constraint violations.
        بنحسب الفتنس للكروموسوم بناء على عدد التعارضات

        Fitness is calculated as the proportion of total possible clashes avoided.
        الفتنس بيتحسب كنسبة التعارضات اللي اتجنبناها من إجمالي التعارضات الممكنة
        A higher fitness score indicates a better timetable (fewer clashes).
        كل ما كان الفتنس اعلى، كان الجدول احسن (تعارضات اقل)

        Args:
            chromosome (List[int]): The chromosome to evaluate.
            الكروموسوم اللي هنقيمه

        Returns:
            float: The fitness score between 0 and 1.
            قيمة الفتنس بين 0 و1
        """

        # Create class schedules from the chromosome
        # بنعمل جداول المحاضرات من الكروموسوم
        classes = self.create_classes(chromosome)

        # Count the number of clashes (constraint violations)
        # بنحسب عدد التعارضات
        clashes = self.count_clashes(classes)

        # Calculate the total possible number of clashes
        # بنحسب إجمالي التعارضات الممكنة
        total_possible_clashes = self.calculate_total_possible_clashes()


        # Calculate fitness as the proportion of clashes avoided
        # بنحسب الفتنس كنسبة التعارضات اللي اتجنبناها
        fitness = (total_possible_clashes - clashes) / total_possible_clashes
        return fitness

    def create_classes(self, chromosome: List[int]) -> List[Class]:
        """
        Create Class objects from a chromosome.

        This method translates the chromosome (list of genes) into a list of Class objects
        that represent the scheduled classes.
        الطريقة دي بتحول الكروموسوم لقائمة من المحاضرات الجدول

        Args:
            chromosome (List[int]): The chromosome representing the schedule.
            الكروموسوم اللي بيمثل الجدول

        Returns:
            List[Class]: A list of Class objects derived from the chromosome.
            قائمة بالمحاضرات اللي طلعناها من الكروموسوم
        """
        classes = []
        gene_idx = 0  # Index to keep track of the current position in the chromosome
                      #  لمتابعة مكاننا في الكروموسوم
        class_id = 1  # Unique identifier for each class
                      # رقم  لكل محاضرة

        # Loop through each group and their courses to create class schedules
        # بنلف على كل مجموعة وكورساتها عشان نعمل جداول المحاضرات
        for group in self.timetable.groups:
            for course_id in group.course_ids:
                # Extract genes for this class: time slot index, room ID, instructor ID
                # بنجيب الجينات الخاصة بالمحاضرة ده: رقم الوقت، رقم القاعة، رقم الدكتور
                timeslot_idx = chromosome[gene_idx]
                room_id = chromosome[gene_idx + 1]
                instructor_id = chromosome[gene_idx + 2]

                # Get the day and time range for the time slot
                # بنجيب اليوم ونطاق الوقت للوقت ده
                day, start_time, end_time = self.time_slots[timeslot_idx]
                time_range = TimeRange(start_time, end_time)

                # Create a Class object with the extracted information
                # بنعمل اوبجيكت Class بالمعلومات اللي جمعناها
                classes.append(Class(
                    id=class_id,
                    group_id=group.id,
                    course_id=course_id,
                    instructor_id=instructor_id,
                    room_id=room_id,
                    day=day,
                    time_range=time_range
                ))

                # Update indices and class ID for the next iteration
                # بنحدث المؤشرات ورقم المحاضرة للمرة الجاية
                class_id += 1
                gene_idx += 3  # Each class has 3 genes in the chromosome

        return classes

    def count_clashes(self, classes: List[Class]) -> int:
        """
        Count the number of constraint violations (clashes) in a set of classes.
        بنحسب عدد التعارضات في مجموعة المحاضرات

        Clashes include:
        - Room capacity violations
        - Room type mismatches
        - Instructor availability conflicts
        - Instructor not allowed to teach the course
        - Instructor type not matching session type
        - Scheduling conflicts (room, instructor, or group booked at the same time)
        التعارضات بتشمل:
        - سعة القاعة غير كافية
        - نوع القاعة غير مناسب
        - الدكتور مش متاح
        - الدكتور مش مسموح له يدرس الكورس
        - نوع الدكتور مش مناسب لنوع المحاضرة
        - تعارض في جدول (قاعة، دكتور، مجموعة)

        Args:
            classes (List[Class]): The list of classes to check for clashes.
            قائمة المحاضرات اللي هنفحصها

        Returns:
            int: The total number of clashes detected.
            إجمالي عدد التعارضات اللي لقيناها
        """
        clashes = 0 # Initialize clash counter
                    # بنبدا عداد التعارضات

        # Compare each class with every other class to find clashes
        # بنقارن كل محاضرة مع باقي المحاضرات عشان نلاقي التعارضات
        for i, class1 in enumerate(classes):
            # Get related objects for class1
            # بنجيب البيانات المتعلقة بالمحاضرة الاول
            room = next(r for r in self.timetable.rooms if r.id == class1.room_id)
            group = next(g for g in self.timetable.groups if g.id == class1.group_id)
            course = next(c for c in self.timetable.courses if c.id == class1.course_id)
            instructor = next(instr for instr in self.timetable.instructors if instr.id == class1.instructor_id)

             # Check if the room's capacity is sufficient for the group's size
            # بنشوف إذا كانت سعة القاعة كافية لحجم المجموعة
            if group.size > room.capacity:
                clashes += 1 # Room capacity clash
                              # تعارض مساحة القاعة

            # Check if the room type matches the course's required room type
            # بنشوف إذا كان نوع القاعة مناسب لمتطلبات الكورس
            if room.room_type != course.required_room_type:
                clashes += 1 # Room type clash
                             # تعارض نوع القاعة


            # Check if the instructor is available at the class's scheduled time
            # بنشوف إذا كان الدكتور متاح في الوقت المحدد
            day_availability = instructor.availability.get(class1.day, [])
            time_conflict = True  # Assume there is a time conflict initially
                                  # نفترض في البداية إن في تعارض في الوقت
            for tr in day_availability:
                if self.time_overlap(tr, class1.time_range):
                    time_conflict = False # Instructor is available
                                          # الدكتور متاح
                    break
            if time_conflict:
                clashes += 1  # Instructor availability clash
                              # تعارض توفر الدكتور

            

            # Check if the instructor is allowed to teach this course
            # بنشوف إذا كان الدكتور مسموح له يدرس الكورس ده
            if class1.instructor_id not in course.allowed_instructors:
                clashes += 1  # Instructor assignment clash
                              # تعارض تعيين الدكتور

            # Check if the instructor's type matches the course's session type
            # بنشوف إذا كان نوع الدكتور مناسب لنوع المحاضرة
            if (course.session_type in ["Lab", "Tutorial"]) and instructor.type != "TA":
                clashes += 1 # Instructor type clash for Lab/Tutorial
                             # تعارض نوع الدكتور في لاب/تمرين
            if course.session_type == "Lecture" and instructor.type != "Professor":
                clashes += 1 # تعارض نوع الدكتور في محاضرة

            # Compare class1 with all subsequent classes to check for overlapping schedules
            # بنقارن المحاضرة الاول مع باقي المحاضرات عشان نشوف التعارضات في الجدول
            for j in range(i + 1, len(classes)):
                # Check if the classes are scheduled at the same time on the same day
                # بنشوف إذا كانت المحاضرات جدول في نفس الوقت ونفس اليوم
                class2 = classes[j]
                if class1.day == class2.day and self.time_overlap(class1.time_range, class2.time_range):
                    # If classes overlap in time, check for specific clashes
                    # لو المحاضرات بتتعارض في الوقت، بنشوف نوع التعارض

                    # Check if the same room is assigned to both classes
                    # بنشوف إذا كانت نفس القاعة مخصصة للمحاضرتين
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
        """
        Check if two time ranges overlap.
        بنشوف إذا كان فيه تداخل بين وقتين

        This method is used to determine if two classes are scheduled at overlapping times.
        الطريقة دي بنستخدمها عشان نعرف إذا كان فيه تعارض في جدول المحاضرات

        Args:
            tr1 (TimeRange): The first time range.
            الوقت الاول
            tr2 (TimeRange): The second time range.
            الوقت الثاني

        Returns:
            bool: True if the time ranges overlap, False otherwise.
            True لو الوقتين بيتداخلوا، False لو لا
        """
        format_str = "%H:%M"  # Time format for parsing
                              # صيغة الوقت للتحويل

        # Convert start and end times to datetime objects
        # بنحول اوقات البدء والانتهاء لكائنات datetime
        start1 = datetime.datetime.strptime(tr1.start_time, format_str)
        end1 = datetime.datetime.strptime(tr1.end_time, format_str)
        start2 = datetime.datetime.strptime(tr2.start_time, format_str)
        end2 = datetime.datetime.strptime(tr2.end_time, format_str)

        # Check if there is an overlap between the two time ranges
        # بنشوف إذا كان فيه تداخل بين الوقتين
        return max(start1, start2) < min(end1, end2)

    def run(self) -> List[Class]:
        """
        Run the genetic algorithm to find an optimized timetable.
        بنشغل الجينيتك الجوريذم عشان نلاقي جدول محاضرات مثالي

        This method orchestrates the evolution process, iterating through generations
        to improve the population of schedules.
        الطريقة دي بتدير عملية التطور، وبنكررها علي مدي الاجيال لتحسين الجداول

        Returns:
            List[Class]: A list of Class objects representing the best found schedule.
            قائمة بالمحاضرات اللي بتمثل احسن جدول وصلنا له
        """
        # Initialize the population with random individuals
        with Pool(processes=12) as pool:
            # Create a pool of processes to generate individuals in parallel
            population = pool.map(lambda _: self.create_individual(), range(self.population_size))
         # Iterate over generations

        for generation in range(self.max_generations):
            # Calculate fitness scores for all individuals in the population
            # بنحسب الفتنس لكل الافراد في السكان
            with Pool(processes=12) as pool:
                fitness_scores = pool.map(self.calculate_fitness, population)

            # Find the best fitness score in the current generation
            # بنلاقي احسن فتنس في الجيل الحالي
            best_fitness = max(fitness_scores)
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            if best_fitness == 1.0:
                # Perfect solution found; return the corresponding schedule
                # خلاص لاقينا جدول ممتاز
                best_idx = fitness_scores.index(best_fitness)
                return self.create_classes(population[best_idx])

            # Create a new population for the next generation
            new_population = []

            # Elitism: Preserve the top-performing individuals without changes
            # بنستخدم الإيليتزم بنحتفظ باحسن الافراد بدون تغييرات
            elite_indices = sorted(range(len(fitness_scores)),
                                   key=lambda i: fitness_scores[i],
                                   reverse=True)[:self.elite_size]
            new_population.extend([population[i] for i in elite_indices])

            # Fill the rest of the population with offspring
            # بنملا باقي السكان بالابناء
            while len(new_population) < self.population_size:
                # Select parents using tournament selection
                # بنختار الآباء باستخدام التورنمنت سيليكشن
                parent1 = self.tournament_select(population, fitness_scores)
                parent2 = self.tournament_select(population, fitness_scores)

                # Perform crossover to produce a child
                # بنعمل كروس اوفر لإنتاج طفل
                child = self.uniform_crossover(parent1, parent2)

                # Apply mutation to the child
                # بنطبق الميوتاشن على الطفل
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                # Add the child to the new population
                # بنضيف الطفل للسكان الجديد
                new_population.append(child)

            # Update the population for the next generation
            # بنحدث السكان للجيل التالي
            population = new_population

        # If the maximum number of generations is reached, return the best found solution
        # لو وصلنا لاقصى عدد من الاجيال، بنرجع احسن حل لقيناه
        best_idx = fitness_scores.index(max(fitness_scores))
        return self.create_classes(population[best_idx])

    def tournament_select(self, population: List[List[int]],
                          fitness_scores: List[float]) -> List[int]:
        """
        Select an individual using tournament selection.
        بنختار فرد باستخدام التورنمنت سيليكشن

        In tournament selection, a subset of the population is chosen at random,
        and the best individual from this subset is selected.
        في التورنمنت سيليكشن، بنختار مجموعة عشوائية من السكان، وبناخد احسن فرد فيهم

        Args:
            population (List[List[int]]): The current population.
            السكان الحاليين
            fitness_scores (List[float]): The fitness scores of the population.
            قيم الفتنس للسكان

        Returns:
            List[int]: The selected individual.
            الفرد اللي اخترناه
        """
        # Randomly select individuals to participate in the tournament
        # بنختار افراد عشوائيين للمشاركة في التورنمنت
        tournament = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament]

        # Select the individual with the highest fitness score among the tournament participants
        # بنختار الفرد اللي عنده اعلى فتنس بين المشاركين

        winner_idx = tournament[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform uniform crossover between two parents to create a child.
        بنعمل كروس اوفر موحد بين ابوين عشان ننتج طفل

        In uniform crossover, each gene is independently selected from one of the parents.
        في الكروس اوفر الموحد، كل جين بنختاره من احد الابوين بشكل مستقل

        Args:
            parent1 (List[int]): The first parent chromosome.
            الكروموسوم الاول
            parent2 (List[int]): The second parent chromosome.
            الكروموسوم الثاني

        Returns:
            List[int]: The child chromosome resulting from crossover.
            الكروموسوم الناتج عن الكروس اوفر
        """
        child = []
        # Loop through each gene and randomly select from one of the parents
        # بنلف على كل جين وبنختار عشوائيًا من احد الابوين
        for p1_gene, p2_gene in zip(parent1, parent2):
            child.append(p1_gene if random.random() < 0.5 else p2_gene)
        return child

    def mutate(self, individual: List[int]) -> List[int]:
        """
        Perform mutation on an individual.
        بنعمل ميوتاشن لفرد

        Mutation introduces genetic diversity into the population by randomly altering genes.
        الميوتاشن بيضيف تنوع جيني للسكان عن طريق تغيير الجينات عشوائيًا

        Args:
            individual (List[int]): The chromosome to mutate.
            الكروموسوم اللي هنعمل له ميوتاشن

        Returns:
            List[int]: The mutated chromosome.
            الكروموسوم بعد الميوتاشن
        """
        mutated = individual.copy()
        gene_length = 3  # Each class is represented by 3 genes: time slot, room, instructor
                         # كل محاضرة بيمثله 3 جينات: الوقت، القاعة، الدكتور
        num_classes = len(mutated) // gene_length  # Total number of classes
                                                   # إجمالي عدد المحاضرات
        
        # Loop through each class in the chromosome
        # بنلف على كل محاضرة في الكروموسوم
        for class_idx in range(num_classes):
            idx = class_idx * gene_length  # Index of the first gene for this class

             # Get group ID and course ID for this class
             # بنجيب رقم المجموعة ورقم الكورس للمحاضرة ده
            group_id, course_id = self.class_mappings[class_idx]
            group = next(g for g in self.timetable.groups if g.id == group_id)
            course = next(c for c in self.timetable.courses if c.id == course_id)

            # Mutate time slot gene
            # ميوتاشن لجين الوقت
            if random.random() < self.mutation_rate:
                # Get current instructor for this class
                # بنجيب الدكتور الحالي للمحاضرة ده
                instructor_id = mutated[idx + 2]
                instructor = next(instr for instr in self.timetable.instructors if instr.id == instructor_id)
                
                
                # Find available time slots for this instructor
                # بنلاقي الاوقات المتاحة للدكتور ده
                available_time_slots = []
                for idx_ts, (day, start_time, end_time) in enumerate(self.time_slots):
                    if day in instructor.availability:
                        for tr in instructor.availability[day]:
                            timeslot_tr = TimeRange(start_time, end_time)
                            if self.time_overlap(tr, timeslot_tr):
                                available_time_slots.append(idx_ts)
                                break  # Found a matching time slot
                if available_time_slots:
                    # Randomly select a new time slot from the available ones
                    # بنختار وقت جديد عشوائيًا من الاوقات المتاحة
                    mutated[idx] = random.choice(available_time_slots)
            
            # Mutate room gene
            # ميوتاشن لجين القاعة
            if random.random() < self.mutation_rate:
                # Find suitable rooms for this course and group
                # بنلاقي القاعات المناسبة للكورس والمجموعة
                suitable_rooms = [room for room in self.timetable.rooms
                                  if room.room_type == course.required_room_type and room.capacity >= group.size]
                if suitable_rooms:
                    # Randomly select a new room from the suitable ones
                    # بنختار قاعة جديدة عشوائيًا من القاعات المناسبة
                    mutated[idx + 1] = random.choice([room.id for room in suitable_rooms])
            # Mutate instructor gene
            # ميوتاشن لجين الدكتور
            if random.random() < self.mutation_rate:
                # Randomly select a new instructor from the allowed instructors for this course
                # بنختار دكتور جديد عشوائيًا من الدكاترة المسموحين للكورس ده
                mutated[idx + 2] = random.choice(course.allowed_instructors)

        return mutated
