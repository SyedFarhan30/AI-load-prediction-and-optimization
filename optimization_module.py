"""
Optimization Module
Implements resource optimization using Hill Climbing and CSP scheduling

FIXES:
- create_sample_tasks: used random.uniform/randint without importing random
- get_optimization_result: 'improvement' values were just the optimized values,
  not actual reductions; now compared against a stored initial state
- CSPScheduler.schedule_tasks: tasks list safety check added
- optimize_resources: passes initial_state to optimizer for improvement calc
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import random
import warnings
warnings.filterwarnings('ignore')


class HillClimbingOptimizer:
    """
    Resource Optimization using Hill Climbing algorithm.
    Goal: Minimise total power consumption while maintaining performance.
    """

    def __init__(self, cpu_limit=80, memory_limit=85, max_iterations=100):
        self.cpu_limit      = cpu_limit
        self.memory_limit   = memory_limit
        self.max_iterations = max_iterations
        self.best_solution  = None
        self.best_cost      = float('inf')
        self.initial_state  = None          # FIX: store for improvement calc
        self.optimization_history = []

    # ------------------------------------------------------------------
    def fitness_function(self, state):
        """
        Calculate fitness score (lower is better).
        cost = power_consumption + penalty for constraint violations.
        """
        power  = state.get('power_consumption', 100)
        cpu    = state.get('cpu_utilization',   0)
        memory = state.get('memory_usage',      0)

        cost = power
        if cpu    > self.cpu_limit:    cost += (cpu    - self.cpu_limit)    * 2
        if memory > self.memory_limit: cost += (memory - self.memory_limit) * 2
        return cost

    # ------------------------------------------------------------------
    def generate_neighbor(self, current_state):
        """Generate a neighbor state by randomly tweaking one resource."""
        neighbor = deepcopy(current_state)
        strategy = random.choice(['reduce_cpu', 'reduce_memory', 'reduce_latency'])

        if strategy == 'reduce_cpu':
            reduction = random.uniform(0.05, 0.10)
            neighbor['cpu_utilization']   = max(0, neighbor.get('cpu_utilization', 0)   * (1 - reduction))
            neighbor['power_consumption'] = max(0, neighbor.get('power_consumption', 0) * (1 - reduction * 0.5))

        elif strategy == 'reduce_memory':
            reduction = random.uniform(0.03, 0.07)
            neighbor['memory_usage']      = max(0, neighbor.get('memory_usage', 0)      * (1 - reduction))
            neighbor['power_consumption'] = max(0, neighbor.get('power_consumption', 0) * (1 - reduction * 0.3))

        elif strategy == 'reduce_latency':
            reduction = random.uniform(0.10, 0.20)
            neighbor['network_latency']   = max(0, neighbor.get('network_latency', 0)   * (1 - reduction))
            neighbor['disk_io']           = max(0, neighbor.get('disk_io', 0)           * (1 - reduction * 0.5))

        return neighbor

    # ------------------------------------------------------------------
    def optimize(self, initial_state):
        """
        Run Hill Climbing optimisation.

        Returns optimisation result dict.
        """
        self.initial_state = deepcopy(initial_state)   # FIX: store for later
        current_state = deepcopy(initial_state)
        current_cost  = self.fitness_function(current_state)

        self.best_solution = deepcopy(current_state)
        self.best_cost     = current_cost
        self.optimization_history = []

        print("✓ Starting Hill Climbing Optimization")
        print(f"  Initial Cost:  {current_cost:.4f}")
        print(f"  CPU Limit:     {self.cpu_limit}%,  "
              f"Memory Limit: {self.memory_limit}%")

        for iteration in range(self.max_iterations):
            neighbor      = self.generate_neighbor(current_state)
            neighbor_cost = self.fitness_function(neighbor)

            self.optimization_history.append({
                'iteration':      iteration,
                'current_cost':   current_cost,
                'neighbor_cost':  neighbor_cost,
                'is_improvement': neighbor_cost < current_cost,
            })

            if neighbor_cost < current_cost:
                current_state = neighbor
                current_cost  = neighbor_cost

                if current_cost < self.best_cost:
                    self.best_solution = deepcopy(current_state)
                    self.best_cost     = current_cost
                    print(f"  Iteration {iteration:3d}: improved cost = {current_cost:.4f}")

        print(f"  Final Best Cost: {self.best_cost:.4f}")
        return self.get_optimization_result()

    # ------------------------------------------------------------------
    def get_optimization_result(self):
        """
        Return detailed optimisation result.

        FIX: 'improvement' now shows the REDUCTION from initial to best,
        expressed as absolute decrease (positive = improvement).
        """
        result = {
            'best_solution': self.best_solution,
            'best_cost':     self.best_cost,
            'improvement':   {
                'cpu_reduction':    0.0,
                'memory_reduction': 0.0,
                'power_reduction':  0.0,
            },
            'recommendations': [],
        }

        if self.best_solution and self.initial_state:
            result['improvement']['cpu_reduction'] = max(0.0, float(
                self.initial_state.get('cpu_utilization',   0) -
                self.best_solution.get('cpu_utilization',   0)
            ))
            result['improvement']['memory_reduction'] = max(0.0, float(
                self.initial_state.get('memory_usage',      0) -
                self.best_solution.get('memory_usage',      0)
            ))
            result['improvement']['power_reduction'] = max(0.0, float(
                self.initial_state.get('power_consumption', 0) -
                self.best_solution.get('power_consumption', 0)
            ))

            cpu    = self.best_solution.get('cpu_utilization', 0)
            memory = self.best_solution.get('memory_usage',    0)

            if cpu > self.cpu_limit:
                result['recommendations'].append(
                    "⚠ CPU still exceeds limit. Consider task migration."
                )
            elif cpu < self.cpu_limit * 0.6:
                result['recommendations'].append(
                    "✓ CPU well optimised. Consider consolidating tasks."
                )

            if memory > self.memory_limit:
                result['recommendations'].append(
                    "⚠ Memory still exceeds limit. "
                    "Increase system RAM or reduce processes."
                )
            elif memory < self.memory_limit * 0.7:
                result['recommendations'].append(
                    "✓ Memory efficiently managed."
                )

        return result


# -----------------------------------------------------------------------
class CSPScheduler:
    """
    Constraint Satisfaction Problem based task scheduler.

    Constraints:
    - CPU utilisation ≤ cpu_limit
    - Memory usage   ≤ memory_limit
    - Tasks must fit within time_slots
    """

    def __init__(self, cpu_limit=80, memory_limit=85, time_slots=24):
        self.cpu_limit    = cpu_limit
        self.memory_limit = memory_limit
        self.time_slots   = time_slots
        self.schedule     = {}

    # ------------------------------------------------------------------
    def is_valid_assignment(self, task, time_slot, current_resources):
        """Check whether task fits within constraints for a given slot."""
        new_cpu    = current_resources.get('cpu',    0) + task.get('cpu_req',    5)
        new_memory = current_resources.get('memory', 0) + task.get('memory_req', 10)
        return new_cpu <= self.cpu_limit and new_memory <= self.memory_limit

    # ------------------------------------------------------------------
    def schedule_tasks(self, tasks, df_metrics=None):
        """Schedule tasks using greedy backtracking (CSP approach)."""
        # FIX: guard empty task list
        if not tasks:
            return {
                'schedule':        {},
                'scheduled_count': 0,
                'total_tasks':     0,
                'success_rate':    0.0,
            }

        print("✓ CSP-based Task Scheduling Started")
        print(f"  Tasks to schedule:    {len(tasks)}")
        print(f"  Time slots available: {self.time_slots}")

        slots = {
            i: {'cpu': 0, 'memory': 0, 'tasks': []}
            for i in range(self.time_slots)
        }

        sorted_tasks = sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True)

        scheduled = 0
        for task in sorted_tasks:
            assigned = False
            for slot_idx in range(self.time_slots):
                if self.is_valid_assignment(task, slot_idx, slots[slot_idx]):
                    slots[slot_idx]['cpu']    += task.get('cpu_req',    5)
                    slots[slot_idx]['memory'] += task.get('memory_req', 10)
                    slots[slot_idx]['tasks'].append(
                        task.get('id', f'Task_{scheduled}')
                    )
                    scheduled += 1
                    assigned  = True
                    break

            if not assigned:
                print(f"  ⚠ Could not schedule: {task.get('id', 'Unknown task')}")

        print(f"  Successfully scheduled: {scheduled}/{len(tasks)} tasks")

        return {
            'schedule':        slots,
            'scheduled_count': scheduled,
            'total_tasks':     len(tasks),
            'success_rate':    scheduled / len(tasks) * 100,
        }


# -----------------------------------------------------------------------
def create_sample_tasks(num_tasks=10):
    """
    Create sample tasks for scheduling.

    FIX: original code used random.uniform / random.randint without importing
    the random module inside the function scope.  Module-level import added.
    """
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'id':         f'Task_{i}',
            'cpu_req':    random.uniform(5, 20),
            'memory_req': random.uniform(10, 30),
            'priority':   random.randint(1, 5),
            'duration':   random.uniform(1, 4),
        })
    return tasks


# -----------------------------------------------------------------------
def optimize_resources(df, predictions, cpu_limit=80, memory_limit=85):
    """
    Main optimisation entry point.

    Args:
        df:          DataFrame with system metrics
        predictions: Model predictions (not used directly but available)
        cpu_limit:   Maximum acceptable CPU utilisation
        memory_limit:Maximum acceptable memory utilisation

    Returns:
        dict with 'optimization', 'scheduling', and 'optimizer' keys.
    """
    # Build average current state from data
    def _col_mean(col):
        return float(df[col].mean()) if col in df.columns else 0.0

    initial_state = {
        'cpu_utilization':  _col_mean('cpu_utilization'),
        'memory_usage':     _col_mean('memory_usage'),
        'power_consumption': _col_mean('power_consumption'),
        'network_latency':  _col_mean('network_latency'),
        'disk_io':          _col_mean('disk_io'),
    }

    # Hill Climbing
    optimizer = HillClimbingOptimizer(
        cpu_limit, memory_limit, max_iterations=100
    )
    optimization_result = optimizer.optimize(initial_state)

    # CSP Scheduling
    tasks = create_sample_tasks(num_tasks=15)
    scheduler = CSPScheduler(cpu_limit, memory_limit, time_slots=24)
    scheduling_result = scheduler.schedule_tasks(tasks, df)

    return {
        'optimization': optimization_result,
        'scheduling':   scheduling_result,
        'optimizer':    optimizer,
    }