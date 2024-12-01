from timetable import TimeTable
from genetic_algorithm import GeneticAlgorithm
from schedule_printer import print_schedule

def main():
    # Create timetable
    timetable = TimeTable()

    #run genetic algorithm
    ga = GeneticAlgorithm(timetable)
    solution = ga.run()

    #Print generated schedule
    print_schedule(timetable, solution)

if __name__ == "__main__":
    main()
