# Implements:
#   • pseudo-random proportional rule controlled by q0
#   • local pheromone update after each assignment
#   • global pheromone update by the iteration-best ant
# No post-run repair is included (per your earlier request).

import random
import math
from typing import List
from pathos.multiprocessing import ProcessingPool as Pool

from data_models import TimeRange, Class
from timetable import TimeTable


class AntCOlonyAlgorithm:
    """Ant Colony System optimiser for university timetables."""

    # ----------------------------------------------------------- initialise --
    def __init__(self, timetable: TimeTable):
        self.timetable = timetable

        # --- ACS hyper-parameters ------------------------------------------
        self.num_ants        = 400
        self.max_iterations  = 600
        self.alpha           = 1.0       # pheromone influence
        self.beta            = 2.0       # heuristic influence
        self.rho             = 0.15      # evaporation rate (also used in local update)
        self.q_const         = 1.0       # deposit constant Q
        self.q0              = 0.90      # pseudo-random rule threshold
        self.max_no_improve  = 250

        # --- domain data ----------------------------------------------------
        self.days_of_week  = [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday"
        ]
        self.slots_per_day = 8
        self.time_slots    = [
            (d, s) for d in self.days_of_week for s in range(self.slots_per_day)
        ]

        self.class_mappings = [
            (g.id, cid) for g in self.timetable.groups for cid in g.course_ids
        ]

        self.all_room_ids       = [r.id for r in self.timetable.rooms]
        self.all_instructor_ids = [i.id for i in self.timetable.instructors]

        # --- pheromone tables  (gene_index → {value: τ}) --------------------
        self.tau0 = 1.0                      # initial trail strength
        self.pheromones = [dict() for _ in range(len(self.class_mappings) * 3)]

        # parallel pool for fitness evaluations
        self.pool = Pool(nodes=min(16, self.num_ants))

    # -------------------------------------------------------------- run ACS --
    def run(self) -> List[Class]:
        best_global, best_score = None, -1.0
        no_improve = 0

        for it in range(self.max_iterations):
            # -------- construct colony -------------------------------------
            ants = [self._construct_ant() for _ in range(self.num_ants)]
            scores = self.pool.map(self.calculate_fitness, ants)

            # -------- iteration best ---------------------------------------
            ibest_idx   = max(range(self.num_ants), key=scores.__getitem__)
            ibest_score = scores[ibest_idx]
            ibest_chrom = ants[ibest_idx]

            print(f"Iter {it+1:3d}:  best = {ibest_score:.5f}"
                  f"   (global {best_score:.5f})")

            # -------- global best update -----------------------------------
            if ibest_score > best_score:
                best_score  = ibest_score
                best_global = ibest_chrom[:]
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= self.max_no_improve:
                    print(f"No improvement for {self.max_no_improve} iterations — stopping.")
                    break

            if best_score == 1.0:
                break

            # -------- pheromone evaporation --------------------------------
            for g_tau in self.pheromones:
                for v in list(g_tau.keys()):
                    g_tau[v] *= (1.0 - self.rho)
                    if g_tau[v] < 1e-6:
                        g_tau[v] = 1e-6

            # -------- global pheromone deposit (iteration best) ------------
            deposit = self.q_const * ibest_score
            for gi, val in enumerate(ibest_chrom):
                self.pheromones[gi][val] = self.pheromones[gi].get(val, self.tau0) + deposit

        self._close_pool()
        if best_global is None:  # failsafe
            best_global = self._construct_ant()
        return self.create_classes(best_global)

    # --------------------------------------- ant construction (ACS rules) ---
    def _construct_ant(self) -> List[int]:
        chromosome = []
        instructors = {i.id: i for i in self.timetable.instructors}
        courses     = {c.id: c for c in self.timetable.courses}
        groups      = {g.id: g for g in self.timetable.groups}

        for cidx, (gid, cid) in enumerate(self.class_mappings):
            course = courses[cid]
            group  = groups[gid]
            gbase  = cidx * 3

            # ----- gene 0 : timeslot --------------------------------------
            feasible_ts = [
                idx for idx, (d, s) in enumerate(self.time_slots)
                if s + course.duration <= self.slots_per_day
            ]
            ts = self._state_transition(gbase + 0, feasible_ts,
                                        lambda _: 1.0)
            chromosome.append(ts)
            self._local_update(gbase + 0, ts)    # ACS local pheromone update
            day, start_slot = self.time_slots[ts]

            # ----- gene 1 : room ------------------------------------------
            req = course.required_room_type
            feasible_rooms = [
                r.id for r in self.timetable.rooms
                if r.room_type == req and r.capacity >= group.size
            ] or self.all_room_ids
            room = self._state_transition(gbase + 1, feasible_rooms,
                                          lambda _: 1.0)
            chromosome.append(room)
            self._local_update(gbase + 1, room)

            # ----- gene 2 : instructor ------------------------------------
            feasible_instructors = course.allowed_instructors or self.all_instructor_ids
            instr = self._state_transition(gbase + 2, feasible_instructors,
                                           lambda iid: 1.0 if iid in course.allowed_instructors else 0.01)
            chromosome.append(instr)
            self._local_update(gbase + 2, instr)

        return chromosome

    # ------------------------ ACS pseudo-random proportional rule ----------
    def _state_transition(self, gi: int, candidates: List[int], heuristic_fn):
        tau_table = self.pheromones[gi]

        # exploitation vs exploration
        if random.random() < self.q0:
            # greedy choice
            best_val, best_score = None, -1.0
            for v in candidates:
                tau = tau_table.get(v, self.tau0)
                eta = heuristic_fn(v)
                score = (tau ** self.alpha) * (eta ** self.beta)
                if score > best_score:
                    best_val, best_score = v, score
            return best_val

        # roulette-wheel (exploration)
        probs, denom = [], 0.0
        for v in candidates:
            tau = tau_table.get(v, self.tau0)
            eta = heuristic_fn(v)
            p   = (tau ** self.alpha) * (eta ** self.beta)
            probs.append((v, p))
            denom += p
        if denom == 0.0:
            return random.choice(candidates)
        r = random.random() * denom
        acc = 0.0
        for v, p in probs:
            acc += p
            if acc >= r:
                return v
        return candidates[-1]

    # ----------------------------- ACS local pheromone update --------------
    def _local_update(self, gi: int, value: int):
        tau = self.pheromones[gi].get(value, self.tau0)
        tau = (1.0 - self.rho) * tau + self.rho * self.tau0
        self.pheromones[gi][value] = tau

    # -------------------------------------------------- helper utilities ---
    def _close_pool(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()

    # ------------------------------------------------ fitness & helpers ----
    # (identical to previous version -- unchanged)
    def calculate_total_possible_clashes(self) -> int:
        n = len(self.class_mappings)
        return n * (n - 1) // 2 * 3

    def calculate_fitness(self, chromosome: List[int]) -> float:
        clashes = self.count_clashes(self.create_classes(chromosome))
        return (
            self.calculate_total_possible_clashes() - clashes
        ) / self.calculate_total_possible_clashes()

    def create_classes(self, chromosome: List[int]) -> List[Class]:
        classes, idx, cid = [], 0, 1
        for gid, course_id in self.class_mappings:
            course = next(c for c in self.timetable.courses if c.id == course_id)
            dur    = course.duration
            ts, room, inst = chromosome[idx], chromosome[idx + 1], chromosome[idx + 2]
            day, start     = self.time_slots[ts]
            end            = start + dur
            classes.append(Class(
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
        return classes

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

            if g1.size > r1.capacity:
                clashes += 1
            if r1.room_type != co1.required_room_type:
                clashes += 1
            for sl in s1:
                if c1.day not in in1.availability_slots or sl not in in1.availability_slots[c1.day]:
                    clashes += 1
                    break
            if c1.instructor_id not in co1.allowed_instructors:
                clashes += 1
            if co1.session_type in ["Lab", "Tutorial"] and in1.type != "TA":
                clashes += 1
            if co1.session_type == "Lecture" and in1.type != "Professor":
                clashes += 1

            for j in range(i + 1, len(classes)):
                c2 = classes[j]
                if c1.day == c2.day:
                    s2 = slot_range(c2)
                    if any(x in s1 for x in s2):
                        if c1.room_id == c2.room_id:
                            clashes += 1
                        if c1.instructor_id == c2.instructor_id:
                            clashes += 1
                        if c1.group_id == c2.group_id:
                            clashes += 1

        main, sub = {}, {}
        for cl in classes:
            grp = groups[cl.group_id]
            (main if grp.section == "0" else sub).setdefault(grp.group_name, []).append(cl)

        for name, mcls in main.items():
            for mcl in mcls:
                ms = slot_range(mcl)
                for scl in sub.get(name, []):
                    if scl.day == mcl.day and any(x in ms for x in slot_range(scl)):
                        clashes += 1
        return clashes
