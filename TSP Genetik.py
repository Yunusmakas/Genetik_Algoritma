import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

# 0. Verilen Koordinatlar
coordinates_list = [
 (78.064, 11.854), (53.234, 25.899), (75.424, 45.623), (34.001, 95.023), 
    (8.715, 3.341), (71.774, 84.746), (67.23, 19.399), (93.521, 40.346), 
    (59.425, 40.87), (80.047, 39.791), (5.184, 53.758), (78.324, 18.775), 
    (73.766, 5.11), (23.343, 29.992), (34.54, 77.66), (98.013, 82.273), 
    (38.915, 6.069), (50.141, 9.599), (2.374, 86.093), (84.687, 88.798), 
    (50.366, 50.549), (11.154, 34.556), (21.962, 44.887), (39.11, 24.92), 
    (10.84, 21.953), (56.624, 41.55), (37.673, 10.55), (18.684, 11.639), 
    (31.261, 53.821), (54.093, 15.14)
]
coordinates = np.array(coordinates_list)
num_cities = len(coordinates)

# 1. GA Parametreleri
POPULATION_SIZE = 500
NUM_GENERATIONS = 500 
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 10 # En iyi kaç bireyin doğrudan sonraki nesle aktarılacağı
TOURNAMENT_SIZE = 5 # Turnuva seçimi için

# 2. Yardımcı Fonksiyonlar
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_path_distance(path_indices, coords):
    total_distance = 0
    for i in range(len(path_indices)):
        from_city_idx = path_indices[i]
        to_city_idx = path_indices[(i + 1) % len(path_indices)] # Son şehirden ilkine dön
        total_distance += calculate_distance(coords[from_city_idx], coords[to_city_idx])
    return total_distance

# 3. Genetik Algoritma Fonksiyonları

# Birey (rota) oluşturma
def create_individual(num_cities):
    individual = list(range(num_cities))
    random.shuffle(individual)
    return individual

# Başlangıç popülasyonu oluşturma
def create_initial_population(pop_size, num_cities):
    return [create_individual(num_cities) for _ in range(pop_size)]

# Uygunluk (fitness) hesaplama - daha kısa mesafe daha iyi
def get_fitness(individual, coords):
    return 1 / calculate_path_distance(individual, coords) # Maksimizasyon için 1/mesafe

# Seçim (Turnuva Seçimi)
def tournament_selection(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)): # Yeni popülasyon kadar ebeveyn seç
        tournament_contenders_indices = random.sample(range(len(population)), tournament_size)
        tournament_contenders_fitness = [fitness_scores[i] for i in tournament_contenders_indices]
        winner_index_in_tournament = np.argmax(tournament_contenders_fitness)
        winner_index_in_population = tournament_contenders_indices[winner_index_in_tournament]
        selected_parents.append(population[winner_index_in_population])
    return selected_parents

# Çaprazlama (Sıralı Çaprazlama - Ordered Crossover OX1)
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1]*size, [-1]*size

    # Rastgele iki kesme noktası seç
    start, end = sorted(random.sample(range(size), 2))

    # Ebeveyn1'den kesiti çocuk1'e kopyala
    child1[start:end+1] = parent1[start:end+1]
    # Ebeveyn2'den kesiti çocuk2'ye kopyala
    child2[start:end+1] = parent2[start:end+1]

    # Kalan elemanları doldur
    # Çocuk1 için
    current_p2_idx = 0
    for i in range(size):
        if child1[i] == -1: # Henüz doldurulmamışsa
            while parent2[current_p2_idx] in child1: # Çocuk1'de zaten varsa atla
                current_p2_idx += 1
            child1[i] = parent2[current_p2_idx]
            current_p2_idx += 1
    # Çocuk2 için
    current_p1_idx = 0
    for i in range(size):
        if child2[i] == -1:
            while parent1[current_p1_idx] in child2:
                current_p1_idx += 1
            child2[i] = parent1[current_p1_idx]
            current_p1_idx += 1
            
    return child1, child2

# Mutasyon (Takas Mutasyonu - Swap Mutation)
def swap_mutation(individual, mutation_rate):
    mutated_individual = list(individual) # Kopyasını al
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
    return mutated_individual

# 4. Ana Genetik Algoritma Döngüsü
def genetic_algorithm(coords, pop_size, num_generations, mutation_rate, crossover_rate, elitism_count, tournament_size):
    num_cities = len(coords)
    population = create_initial_population(pop_size, num_cities)
    
    best_path_overall = None
    best_distance_overall = float('inf')
    
    history_best_paths = [] # Her jenerasyonun en iyi yolunu sakla
    history_best_distances = [] # Her jenerasyonun en iyi mesafesini sakla

    for generation in range(num_generations):
        # Uygunlukları hesapla
        fitness_scores = [get_fitness(ind, coords) for ind in population]
        
        # Bu jenerasyonun en iyisini bul
        current_best_idx = np.argmax(fitness_scores)
        current_best_path = population[current_best_idx]
        current_best_distance = calculate_path_distance(current_best_path, coords)
        
        history_best_paths.append(current_best_path)
        history_best_distances.append(current_best_distance)

        if current_best_distance < best_distance_overall:
            best_distance_overall = current_best_distance
            best_path_overall = current_best_path
            print(f"Jenerasyon {generation+1}: Yeni en iyi mesafe = {best_distance_overall:.2f}")

        # Yeni popülasyonu oluştur
        new_population = []

        # Elitizm: En iyi bireyleri doğrudan aktar
        sorted_population_indices = np.argsort(fitness_scores)[::-1] # En iyiden en kötüye sırala
        for i in range(elitism_count):
            new_population.append(population[sorted_population_indices[i]])

        # Kalan popülasyonu çaprazlama ve mutasyon ile doldur
        num_offspring = pop_size - elitism_count
        
        # Ebeveynleri seç (elitler hariç, kalan popülasyon içinden)
        # Bu kısım için tüm popülasyondan ebeveyn seçmek daha yaygın
        parents = tournament_selection(population, fitness_scores, tournament_size)

        offspring_count = 0
        while offspring_count < num_offspring:
            # Ebeveyn seçimi (turnuva yerine direkt seçilmiş ebeveynlerden kullanabiliriz)
            parent1, parent2 = random.sample(parents, 2)

            if random.random() < crossover_rate:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2 # Çaprazlama olmazsa ebeveynler kopyalanır
            
            new_population.append(swap_mutation(child1, mutation_rate))
            offspring_count +=1
            if offspring_count < num_offspring:
                new_population.append(swap_mutation(child2, mutation_rate))
                offspring_count +=1
        
        population = new_population[:pop_size] # Popülasyon boyutunu koru

    return best_path_overall, best_distance_overall, history_best_paths, history_best_distances

# 5. Animasyon Kurulumu
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("TSP Çözümü - Genetik Algoritma")
ax.set_xlabel("X Koordinatı")
ax.set_ylabel("Y Koordinatı")

# Şehirleri noktalarla çiz
ax.plot(coordinates[:, 0], coordinates[:, 1], 'bo', markersize=8, label="Şehirler")
for i, (x,y) in enumerate(coordinates):
    ax.text(x + 0.5, y + 0.5, str(i), color="red", fontsize=9)

# Başlangıçta boş bir çizgi (path)
line, = ax.plot([], [], 'r-', lw=2, label="Anlık En İyi Rota")
generation_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
distance_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
ax.legend(loc='lower left')
plt.grid(True)


# GA'yı çalıştır ve geçmişi al
best_path, best_distance, path_history, distance_history = genetic_algorithm(
    coordinates, POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT, TOURNAMENT_SIZE
)

print("\n--- Sonuç ---")
print(f"En İyi Rota (indeksler): {best_path}")
print(f"En İyi Rota (koordinatlar):")
for city_idx in best_path:
    print(f"  Şehir {city_idx}: {coordinates[city_idx]}")
print(f"Toplam Mesafe: {best_distance:.2f}")


def init_animation():
    line.set_data([], [])
    generation_text.set_text('')
    distance_text.set_text('')
    return line, generation_text, distance_text

def update_animation(frame_num):
    path_indices = path_history[frame_num]
    
    # Rotayı çizmek için koordinatları hazırla (ilk şehre geri dönmeyi unutma)
    path_coords = np.array([coordinates[i] for i in path_indices] + [coordinates[path_indices[0]]])
    
    line.set_data(path_coords[:, 0], path_coords[:, 1])
    generation_text.set_text(f"Jenerasyon: {frame_num + 1}/{NUM_GENERATIONS}")
    distance_text.set_text(f"Mesafe: {distance_history[frame_num]:.2f}")
    
    # Eksen sınırlarını ayarla (isteğe bağlı, veriye göre otomatik de olabilir)
    # Eğer animasyon sırasında eksenler çok değişiyorsa bunu sabitleyebilirsiniz.
    # min_x, max_x = np.min(coordinates[:,0]), np.max(coordinates[:,0])
    # min_y, max_y = np.min(coordinates[:,1]), np.max(coordinates[:,1])
    # padding = 10 # Sınırlara biraz boşluk ekle
    # ax.set_xlim(min_x - padding, max_x + padding)
    # ax.set_ylim(min_y - padding, max_y + padding)

    return line, generation_text, distance_text

# Animasyonu oluştur
# interval: ms cinsinden kareler arası süre
# blit=True optimizasyon sağlar ama bazen sorun çıkarabilir. False ile deneyebilirsiniz.
ani = FuncAnimation(fig, update_animation, frames=len(path_history),
                    init_func=init_animation, blit=True, repeat=False, interval=100)

plt.show()

# Animasyonu kaydetmek isterseniz (ffmpeg veya imagemagick kurulu olmalı):
# print("Animasyon kaydediliyor... Bu işlem biraz sürebilir.")
# ani.save('tsp_ga_animation.mp4', writer='ffmpeg', fps=10)
# ani.save('tsp_ga_animation.gif', writer='imagemagick', fps=10)
# print("Animasyon kaydedildi.")