import random
import math
from typing import List, Dict
from data_models import TimeRange, Class
from timetable import TimeTable


class BSOalgorithm:
    """
    Single-threaded PSO-based solver under the original GeneticAlgorithm name.
    Call run() as before to get a List[Class].
    """
    def __init__(self, timetable: TimeTable):
        self.timetable = timetable

        # PSO parameters
        self.population_size  = 2000
        self.max_iterations   = 500
        self.cognitive_weight = 0.45
        self.social_weight    = 0.45
        self.inertia_weight   = 0.10
        self.mutation_rate    = 0.20
        self.max_no_improve   = 300

        # time slots setup
        self.days_of_week  = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
        self.slots_per_day = 8
        self.time_slots    = [(d,s) for d in self.days_of_week for s in range(self.slots_per_day)]

        # (group, course) pairs
        self.class_mappings = []
        for g in self.timetable.groups:
            for cid in g.course_ids:
                self.class_mappings.append((g.id, cid))

    def run(self) -> List[Class]:
        swarm      = [self.create_individual() for _ in range(self.population_size)]
        velocities = [[0.5]*len(swarm[0]) for _ in range(self.population_size)]
        pbest      = swarm.copy()
        pbest_scores = [self.calculate_fitness(p) for p in pbest]

        # global best
        gidx = pbest_scores.index(max(pbest_scores))
        gbest     = pbest[gidx][:]
        gbest_val = pbest_scores[gidx]
        no_improve = 0

        for it in range(self.max_iterations):
            print(f"Iteration {it+1}: Best fitness = {gbest_val:.4f}")
            if gbest_val == 1.0:
                return self.create_classes(gbest)
            if no_improve >= self.max_no_improve:
                print(f"No improvement in {self.max_no_improve} iterations. Stopping.")
                break

            for i in range(self.population_size):
                particle = swarm[i]
                velocity = velocities[i]
                local_best = pbest[i]

                for g in range(len(particle)):
                    r1, r2 = random.random(), random.random()
                    velocity[g] = (
                        self.inertia_weight * velocity[g] +
                        self.cognitive_weight * r1 * (local_best[g] != particle[g]) +
                        self.social_weight    * r2 * (gbest[g]      != particle[g])
                    )
                    prob = 1/(1+math.exp(-velocity[g]))
                    if random.random() < prob:
                        # choose local or global best
                        particle[g] = local_best[g] if random.random() < 0.5 else gbest[g]
                    # mutation
                    if random.random() < self.mutation_rate:
                        particle[g] = self.random_gene_value(g)

                fitness = self.calculate_fitness(particle)
                if fitness > pbest_scores[i]:
                    pbest[i]        = particle[:]
                    pbest_scores[i] = fitness
                    if fitness > gbest_val:
                        gbest_val = fitness
                        gbest     = particle[:]
                        no_improve = 0

            no_improve += 1

        # if we exit early:
        return self.repair_solution(self.create_classes(gbest))

    # ---------------------------------------- helper methods ----------------------------------------
    def create_individual(self) -> List[int]:
        instructors = {i.id: i for i in self.timetable.instructors}
        courses     = {c.id: c for c in self.timetable.courses}
        groups      = {g.id: g for g in self.timetable.groups}
        chromosome  = []
        for (gid, cid) in self.class_mappings:
            course = courses[cid]
            group  = groups[gid]
            dur    = course.duration
            iid    = random.choice(course.allowed_instructors)
            inst   = instructors[iid]

            feasible = [idx for idx, (d, s) in enumerate(self.time_slots)
                        if s + dur <= self.slots_per_day
                        and self.can_instructor_teach(inst, d, s, dur)]
            ts = random.choice(feasible) if feasible else random.randint(0, len(self.time_slots)-1)
            room = self.select_most_efficient_room(course.required_room_type, group.size)
            chromosome.extend([ts, room, iid])
        return chromosome

    def can_instructor_teach(self, instructor, day: str, start: int, dur: int) -> bool:
        if day not in instructor.availability_slots:
            return False
        return all(s in instructor.availability_slots[day] for s in range(start, start + dur))

    def select_most_efficient_room(self, rtype: str, size: int) -> int:
        rooms = [r for r in self.timetable.rooms if r.room_type == rtype and r.capacity >= size]
        if not rooms:
            return random.choice([r.id for r in self.timetable.rooms])
        rooms.sort(key=lambda x: x.capacity)
        mincap = rooms[0].capacity
        small = [r.id for r in rooms if r.capacity == mincap]
        return random.choice(small)

    def calculate_total_possible_clashes(self) -> int:
        n = len(self.class_mappings)
        return n * (n - 1) // 2 * 3

    def calculate_fitness(self, chromosome: List[int]) -> float:
        clashes = self.count_clashes(self.create_classes(chromosome))
        total   = self.calculate_total_possible_clashes()
        return (total - clashes) / total

    def create_classes(self, chromosome: List[int]) -> List[Class]:
        out = []
        idx = 0
        cid = 1
        for (gid, course_id) in self.class_mappings:
            course = next(c for c in self.timetable.courses if c.id == course_id)
            dur    = course.duration
            ts, room, inst = chromosome[idx], chromosome[idx+1], chromosome[idx+2]
            day, start = self.time_slots[ts]
            end = start + dur
            out.append(Class(
                id=cid,
                group_id=gid,
                course_id=course_id,
                instructor_id=inst,
                room_id=room,
                day=day,
                time_range=TimeRange(f"Slot {start+1}", f"Slot {end+1}")
            ))
            idx += 3
            cid += 1
        return out

    def count_clashes(self, classes: List[Class]) -> int:
        groups      = {g.id: g for g in self.timetable.groups}
        courses     = {c.id: c for c in self.timetable.courses}
        rooms       = {r.id: r for r in self.timetable.rooms}
        instructors = {i.id: i for i in self.timetable.instructors}
        def slot_range(c):
            s = int(c.time_range.start_time.split()[1]) - 1
            e = int(c.time_range.end_time.split()[1]) - 1
            return range(s, e)
        clashes = 0
        for i, c1 in enumerate(classes):
            g1, co1 = groups[c1.group_id], courses[c1.course_id]
            r1, in1 = rooms[c1.room_id], instructors[c1.instructor_id]
            s1      = slot_range(c1)
            if g1.size > r1.capacity: clashes += 1
            if r1.room_type != co1.required_room_type: clashes += 1
            for sl in s1:
                if c1.day not in in1.availability_slots or sl not in in1.availability_slots[c1.day]:
                    clashes += 1; break
            if c1.instructor_id not in co1.allowed_instructors: clashes += 1
            if co1.session_type in ["Lab","Tutorial"] and in1.type != "TA": clashes += 1
            if co1.session_type == "Lecture" and in1.type != "Professor": clashes += 1
            for j in range(i+1, len(classes)):
                c2 = classes[j]
                if c1.day == c2.day:
                    s2 = slot_range(c2)
                    if any(x in s1 for x in s2):
                        if c1.room_id == c2.room_id: clashes += 1
                        if c1.instructor_id == c2.instructor_id: clashes += 1
                        if c1.group_id == c2.group_id: clashes += 1
        # main-subgroup overlap
        main, sub = {}, {}
        for cl in classes:
            grp = groups[cl.group_id]
            if grp.section == "0": main.setdefault(grp.group_name, []).append(cl)
            else: sub.setdefault(grp.group_name, []).append(cl)
        for name, mcls in main.items():
            for mcl in mcls:
                ms = slot_range(mcl)
                for scl in sub.get(name, []):
                    if scl.day == mcl.day and any(x in ms for x in slot_range(scl)):
                        clashes += 1
        return clashes

    def repair_solution(self, classes: List[Class]) -> List[Class]:
        # Local repair via your original GA code
        for _ in range(100):
            clashes = self.identify_clashes(classes)
            if not clashes: break
            cid = random.choice(list(clashes.keys()))
            chrom = self.classes_to_chromosome(classes)
            if self.try_to_repair_class(chrom, cid):
                classes = self.create_classes(chrom)
        return classes

    def identify_clashes(self, classes: List[Class]) -> Dict[int, List[str]]:
        info = {}
        def add(cid, reason): info.setdefault(cid, []).append(reason)
        # reuse count logic but record reasons
        groups      = {g.id: g for g in self.timetable.groups}
        courses     = {c.id: c for c in self.timetable.courses}
        rooms       = {r.id: r for r in self.timetable.rooms}
        instructors = {i.id: i for i in self.timetable.instructors}
        def slot_range(c):
            s = int(c.time_range.start_time.split()[1]) - 1
            e = int(c.time_range.end_time.split()[1]) - 1
            return range(s, e)
        for i, c1 in enumerate(classes):
            g1, co1 = groups[c1.group_id], courses[c1.course_id]
            r1, in1 = rooms[c1.room_id], instructors[c1.instructor_id]
            s1 = slot_range(c1)
            if g1.size > r1.capacity: add(c1.id, "capacity")
            if r1.room_type != co1.required_room_type: add(c1.id, "room_type")
            for sl in s1:
                if c1.day not in in1.availability_slots or sl not in in1.availability_slots[c1.day]:
                    add(c1.id, "instructor_availability"); break
            if c1.instructor_id not in co1.allowed_instructors: add(c1.id, "instructor_not_allowed")
            if co1.session_type in ["Lab","Tutorial"] and in1.type != "TA": add(c1.id, "instructor_type_lab_tutorial")
            if co1.session_type == "Lecture" and in1.type != "Professor": add(c1.id, "instructor_type_lecture")
            for j in range(i+1, len(classes)):
                c2 = classes[j]
                if c1.day == c2.day:
                    s2 = slot_range(c2)
                    if any(x in s1 for x in s2):
                        if c1.room_id == c2.room_id: add(c1.id, "overlap_room"); add(c2.id, "overlap_room")
                        if c1.instructor_id == c2.instructor_id: add(c1.id, "overlap_instructor"); add(c2.id, "overlap_instructor")
                        if c1.group_id == c2.group_id: add(c1.id, "overlap_group"); add(c2.id, "overlap_group")
        # main-sub overlap
        main, sub = {}, {}
        for cl in classes:
            grp = groups[cl.group_id]
            if grp.section == "0": main.setdefault(grp.group_name, []).append(cl)
            else: sub.setdefault(grp.group_name, []).append(cl)
        for name, mcls in main.items():
            for mcl in mcls:
                ms = slot_range(mcl)
                for scl in sub.get(name, []):
                    if scl.day == mcl.day and any(x in ms for x in slot_range(scl)):
                        add(mcl.id, "main_subgroup_overlap"); add(scl.id, "main_subgroup_overlap")
        return info

    def classes_to_chromosome(self, classes: List[Class]) -> List[int]:
        chrom = [0] * (len(self.class_mappings)*3)
        for i, (gid, cid) in enumerate(self.class_mappings):
            cl = next(c for c in classes if c.group_id == gid and c.course_id == cid)
            day = cl.day
            start = int(cl.time_range.start_time.split()[1]) - 1
            idx = i*3
            chrom[idx] = self.time_slots.index((day, start))
            chrom[idx+1] = cl.room_id
            chrom[idx+2] = cl.instructor_id
        return chrom

    def try_to_repair_class(self, chrom: List[int], class_id: int) -> bool:
        base_i = (class_id-1)*3
        orig = chrom[base_i:base_i+3]
        gid, cid = self.class_mappings[class_id-1]
        course = next(c for c in self.timetable.courses if c.id==cid)
        group  = next(g for g in self.timetable.groups  if g.id==gid)
        dur    = course.duration
        instrs = {i.id:i for i in self.timetable.instructors}
        best_fit = self.calculate_fitness(chrom)
        for nid in course.allowed_instructors:
            inst = instrs[nid]
            slots=[]
            for idx,(d,s) in enumerate(self.time_slots):
                if s+dur<=self.slots_per_day and self.can_instructor_teach(inst,d,s,dur): slots.append(idx)
            for new_ts in slots:
                new_room = self.select_most_efficient_room(course.required_room_type, group.size)
                chrom[base_i],chrom[base_i+1],chrom[base_i+2] = new_ts,new_room,nid
                nf = self.calculate_fitness(chrom)
                if nf>best_fit:
                    return True
                chrom[base_i:base_i+3] = orig
        return False

    def random_gene_value(self, gi: int) -> int:
        t = gi % 3
        if t==0: return random.randint(0,len(self.time_slots)-1)
        if t==1: return random.choice([r.id for r in self.timetable.rooms])
        return random.choice([i.id for i in self.timetable.instructors])