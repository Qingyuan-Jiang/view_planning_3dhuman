import numpy as np
import random
import math
import torch
import matplotlib.pyplot as plt
from src.utils.tsp_utils import get_tsp_path

'''
Genetical path finding
Finds locally best ways from L service centers with [M0, M1, ..., ML] engineers
through atms_number ATMs and back to their service center
'''


class mtsp_planner:

    def __init__(self, pos_drones, pos_actors):
        # Bank parameters
        # self.atms_number = 10  # ATM quantity
        self.atms_number = len(pos_actors)  # ATM quantity
        self.service_centers = len(pos_drones)  # service centers quantity
        self.velocity = 100  # 100 / hour
        self.repair_time = 0  # 0.5 hour
        self.max_engi = 1  # maximum number of engineers in one service center

        # genetic parameters
        self.population_size = min(200, self.service_centers ** self.atms_number)  # population size (even number!)
        self.generations = 500  # population's generations
        self.mut_1_prob = 0.4  # prob of replacing together two atms in combined path
        self.mut_2_prob = 0.6  # prob of reversing the sublist in combined path
        self.mut_3_prob = 0.8  # probability of changing the length of paths for engineers
        self.two_opt_search = False  # better convergence, lower speed for large quantity of atms

        self.seed(0)

        # Initialize salespersons position.
        engineers = []
        for i in range(self.service_centers):
            for j in range(random.randint(1, self.max_engi)):
                engineers.append(i)
        self.engineers = np.array(engineers)
        print('Engineers: {}'.format(self.engineers))

        # Build distance matrix. First three rows are salespersons.
        self.dist = np.zeros((self.atms_number + self.service_centers, self.atms_number))
        # self.points_locations = np.random.randint(0, 100, (self.service_centers + self.atms_number) * 2)
        # self.points_locations = self.points_locations.reshape((self.service_centers + self.atms_number, 2))
        self.points_locations = np.vstack((pos_drones, pos_actors))
        for i in range(self.dist.shape[0]):
            for j in range(self.dist.shape[1]):
                self.dist[i, j] = math.sqrt(
                    (self.points_locations[i, 0] - self.points_locations[j + self.service_centers, 0]) ** 2 +
                    (self.points_locations[i, 1] - self.points_locations[j + self.service_centers, 1]) ** 2)
                if j + self.service_centers == i:
                    self.dist[i][j] = 0

    def seed(self, seed):
        # seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def fitness_pop(self, population):
        fitness_result = np.zeros(len(population))
        for i in range(len(fitness_result)):
            fitness_result[i] = self.fitness(population[i])
        return fitness_result

    def fitness(self, creature):
        sum_dist = np.zeros(len(creature))
        for j in range(len(creature)):
            mat_path = np.zeros((self.dist.shape[0], self.dist.shape[1]))
            path = creature[j]
            if len(path) != 0:
                for v in range(len(path)):
                    if v == 0:
                        mat_path[self.engineers[j], path[v]] = 1
                    else:
                        mat_path[path[v - 1] + self.service_centers, path[v]] = 1
                mat_path = mat_path * self.dist
                # sum_dist[j] = (np.sum(mat_path) + self.dist[self.engineers[j], path[-1]]) / self.velocity + self.repair_time * len(path)
                sum_dist[j] = (np.sum(mat_path)) / self.velocity
        return np.max(sum_dist)

    def birth_prob(self, fitness_result):
        birth_prob = np.abs(fitness_result - np.max(fitness_result))
        birth_prob = birth_prob / (np.sum(birth_prob) + 1e-6)
        return birth_prob

    def mutate(self, creat, engi):
        pnt_1 = random.randint(0, len(creat) - 1)
        pnt_2 = random.randint(0, len(creat) - 1)
        if random.random() < self.mut_1_prob:
            creat[pnt_1], creat[pnt_2] = creat[pnt_2], creat[pnt_1]
        if random.random() < self.mut_2_prob and pnt_1 != pnt_2:
            if pnt_1 > pnt_2:
                pnt_1, pnt_2 = pnt_2, pnt_1
            creat[pnt_1:pnt_2 + 1] = list(reversed(creat[pnt_1:pnt_2 + 1]))
        if random.random() < self.mut_3_prob:
            engi = [number - 1 for number in engi if number != 0]
            # engi = [number - 2 for number in engi if number > 1]
            while (sum(engi) != self.atms_number):
                engi[random.randint(0, len(engi) - 1)] += 1
        return creat, engi

    def two_opt(self, creature):
        sum_dist = np.zeros(len(creature))
        for j in range(len(creature)):
            mat_path = np.zeros((self.dist.shape[0], self.dist.shape[1]))
            path = creature[j]
            if len(path) != 0:
                for v in range(len(path)):
                    if v == 0:
                        mat_path[self.engineers[j], path[v]] = 1
                    else:
                        mat_path[path[v - 1] + self.service_centers, path[v]] = 1
                mat_path = mat_path * self.dist
                sum_dist[j] = (np.sum(mat_path) + self.dist[self.engineers[j], path[-1]]) / self.velocity + self.repair_time * len(path)
        for u in range(len(creature)):
            best_path = creature[u].copy()
            while True:
                previous_best_path = best_path.copy()
                for x in range(len(creature[u]) - 1):
                    for y in range(x + 1, len(creature[u])):
                        path = best_path.copy()
                        if len(path) != 0:
                            path = path[:x] + list(reversed(path[x:y])) + path[y:]  # 2-opt swap
                            mat_path = np.zeros((self.dist.shape[0], self.dist.shape[1]))
                            for v in range(len(path)):
                                if v == 0:
                                    mat_path[self.engineers[u], path[v]] = 1
                                else:
                                    mat_path[path[v - 1] + self.service_centers, path[v]] = 1
                            mat_path = mat_path * self.dist
                            sum_dist_path = (np.sum(mat_path) + self.dist[
                                self.engineers[u], path[-1]]) / self.velocity + self.repair_time * len(path)
                            if sum_dist_path < sum_dist[u]:
                                best_path = path.copy()
                                creature[u] = path.copy()
                if previous_best_path == best_path:
                    break
        return creature

    def crossover_mutation(self, population, birth_prob):
        new_population = []
        for i in range(round(len(population) / 2)):
            prob = np.random.rand(birth_prob.size) - birth_prob
            pair = np.zeros(2).astype(int)
            pair[0] = np.argmin(prob)
            pair[1] = random.randint(0, prob.size - 1)
            engi_1 = [len(population[pair[0]][v]) for v in range(len(population[pair[0]]))]
            engi_2 = [len(population[pair[1]][v]) for v in range(len(population[pair[1]]))]
            parent_1 = []
            parent_2 = []
            for j in range(len(engi_1)):
                parent_1 += population[pair[0]][j]
            for j in range(len(engi_2)):
                parent_2 += population[pair[1]][j]
            creat_1 = [-1] * len(parent_1)
            creat_2 = [-1] * len(parent_2)
            cross_point_1 = random.randint(0, len(parent_1) - 1)
            cross_point_2 = random.randint(0, len(parent_2) - 1)
            node_1 = parent_1[cross_point_1:]
            node_2 = parent_2[cross_point_2:]
            w = 0
            for v in range(len(creat_1)):
                if parent_2[v] not in node_1:
                    creat_1[v] = parent_2[v]
                else:
                    creat_1[v] = node_1[w]
                    w += 1
            w = 0
            for v in range(len(creat_2)):
                if parent_1[v] not in node_2:
                    creat_2[v] = parent_1[v]
                else:
                    creat_2[v] = node_2[w]
                    w += 1
            # mutations
            creat_1, engi_1 = self.mutate(creat_1, engi_1)
            creat_2, engi_2 = self.mutate(creat_2, engi_2)
            # children
            child_1 = []
            engi_sum = 0
            for v in range(len(engi_1)):
                child_1.append(creat_1[engi_sum:engi_sum + engi_1[v]])
                engi_sum += engi_1[v]
            child_2 = []
            engi_sum = 0
            for v in range(len(engi_2)):
                child_2.append(creat_2[engi_sum:engi_sum + engi_2[v]])
                engi_sum += engi_2[v]
            together = [child_1, child_2, population[pair[0]], population[pair[1]]]
            fit = np.array([self.fitness(creature) for creature in together])
            fit = fit.argsort()
            if self.two_opt_search:
                new_population.append(self.two_opt(together[fit[0]]))
                new_population.append(self.two_opt(together[fit[1]]))
            else:
                new_population.append(together[fit[0]])
                new_population.append(together[fit[1]])
        return new_population

    def plot_paths(self, paths):
        plt.clf()
        plt.title('Best path overall')
        for v in range(self.service_centers):
            plt.scatter(self.points_locations[v, 0], self.points_locations[v, 1], c='r')
        for v in range(self.atms_number):
            plt.scatter(self.points_locations[v + self.service_centers, 0], self.points_locations[v + self.service_centers, 1], c='b')
        for v in range(len(paths)):
            if len(paths[v]) != 0:
                path_locations = self.points_locations[self.service_centers:]
                path_locations = path_locations[np.array(paths[v])]
                path_locations = np.vstack((self.points_locations[self.engineers[v]], path_locations))
                # path_locations = np.vstack((path_locations, self.points_locations[self.engineers[v]]))
                plt.plot(path_locations[:, 0], path_locations[:, 1])
        plt.show()
        plt.pause(0.0001)

    def plan_path(self):
        # random population creation
        global path
        population = []
        for i in range(self.population_size):
            atms_range = list(range(self.atms_number))
            pop = [0] * self.engineers.size
            for j in range(self.engineers.size):
                pop[j] = []
                if len(atms_range) != 0:
                    if j != self.engineers.size - 1:
                        for v in range(random.randint(1, round(2 * self.atms_number / self.engineers.size))):
                            pop[j].append(random.choice(atms_range))
                            atms_range.remove(pop[j][-1])
                            if len(atms_range) == 0:
                                break
                    else:
                        for v in range(len(atms_range)):
                            pop[j].append(random.choice(atms_range))
                            atms_range.remove(pop[j][-1])
            population.append(pop)
        self.population = population

        fitness_result = self.fitness_pop(self.population)
        best_mean_creature_result = np.sum(fitness_result)
        # best_mean_creature_result = np.mean(fitness_result)
        best_creature_result = np.min(fitness_result)
        best_selection_prob = self.birth_prob(fitness_result)
        selection_prob = best_selection_prob
        self.plot_paths(population[np.argmin(fitness_result)])
        for i in range(self.generations):
            new_population = self.crossover_mutation(population, selection_prob)
            fitness_result = self.fitness_pop(new_population)
            # mean_creature_result = np.mean(fitness_result)
            mean_creature_result = np.sum(fitness_result)
            if np.min(fitness_result) < best_creature_result:
                self.plot_paths(population[np.argmin(fitness_result)])
                best_creature_result = np.min(fitness_result)
                path = population[np.argmin(fitness_result)]
            if mean_creature_result < best_mean_creature_result:
                best_mean_creature_result = mean_creature_result
                best_selection_prob = self.birth_prob(fitness_result)
                selection_prob = best_selection_prob
                population = new_population.copy()
            print('Mean population time: {0} Best time: {1}'.format(best_mean_creature_result, best_creature_result))

        print("Path before optimizing separately: ", path)
        path_opt = []
        for k in range(self.service_centers):
            path_k = path[k]
            pos_start = torch.from_numpy(self.points_locations[k]).float()
            pos_targets = torch.from_numpy(self.points_locations[np.asarray(path_k) + self.service_centers, :]).float()
            path_tsp_k, tsp_cand_k, configs_k = get_tsp_path(pos_targets, pos_start)
            path_idx_k = np.asarray(path_k)[np.asarray(path_tsp_k[1:]) - 1]
            path_opt.append(path_idx_k.tolist())
        self.plot_paths(path_opt)
        print("Path before optimizing separately: ", path_opt)
        return path