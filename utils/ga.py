import numpy as np
import torch
from utils.narxwithga import NARXModel

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, input_dim, X_tensor, Y_tensor, tournament_size=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.input_dim = input_dim
        self.X_tensor = X_tensor.to(self.device)
        self.Y_tensor = Y_tensor.to(self.device)
        self.population = [self.initialize_population() for _ in range(population_size)]

    def initialize_population(self):
        model = NARXModel(input_dim=self.input_dim).to(self.device)
        for param in model.parameters():
            param.data.uniform_(-1, 1)
        return model

    def evaluate_fitness(self, model):
        model.eval()
        with torch.no_grad():
            predictions = model(self.X_tensor)
            loss = ((predictions - self.Y_tensor) ** 2).mean().item()
        return loss

    def tournament_selection(self):
        candidates = np.random.choice(self.population, self.tournament_size, replace=False)
        best_candidate = min(candidates, key=self.evaluate_fitness)
        return best_candidate

    def select_parents(self):
        parent1 = self.tournament_selection()
        parent2 = self.tournament_selection()
        return parent1, parent2

    def two_point_crossover_tensor(self, tensor1, tensor2):
        flat1 = tensor1.view(-1).clone()
        flat2 = tensor2.view(-1).clone()
        if flat1.numel() < 2:
            return tensor1.clone()
        idx = np.sort(np.random.choice(flat1.numel(), 2, replace=False))
        new_flat = flat1.clone()
        new_flat[idx[0]:idx[1]] = flat2[idx[0]:idx[1]]
        return new_flat.view(tensor1.size())
    
    def blend_crossover_tensor(self, tensor1, tensor2, alpha=0.5):
        return alpha * tensor1 + (1 - alpha) * tensor2
    
    def uniform_crossover_tensor(self, tensor1, tensor2):
        mask = torch.rand_like(tensor1) < 0.5
        return torch.where(mask, tensor1, tensor2)


    def crossover(self, parent1, parent2):
        child = NARXModel(input_dim=self.input_dim).to(self.device)
        with torch.no_grad():
            for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
                if np.random.rand() < self.crossover_rate:
                    c.data.copy_(self.uniform_crossover_tensor(p1.data, p2.data))
                else:
                    c.data.copy_(p1.data if np.random.rand() < 0.5 else p2.data)
        return child

    def mutate(self, model):
        with torch.no_grad():
            for param in model.parameters():
                if np.random.rand() < self.mutation_rate:
                    param.data += torch.randn_like(param) * 0.1
        return model

    def select_population(self, offspring):
        combined_population = self.population + offspring
        combined_population = sorted(combined_population, key=self.evaluate_fitness)
        self.population = combined_population[:self.population_size]

    def run(self, generations):
        for gen in range(generations):
            offspring = []
            for _ in range(self.population_size // 2):
                p1, p2 = self.select_parents()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                offspring.append(child)

            self.select_population(offspring)

            best_fitness = self.evaluate_fitness(self.population[0])
            print(f"Generation {gen+1}, Best Fitness (MSE): {best_fitness:.5f}")

        return self.population[0]  # Best model
