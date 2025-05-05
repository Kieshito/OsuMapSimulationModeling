import numpy as np
import networkx as nx
from mesa.model import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from models.agent import PlayerAgent

class MarketModel(Model):
    def __init__(self, num_players=100000, new_maps_per_day=10):
        super().__init__()
        self.num_players = num_players
        self.new_maps_per_day = new_maps_per_day
        self.schedule = RandomActivation(self)
        self.map_repository = []
        self.map_plays = {}  # map_id -> количество игр
        self.agent_plays = {}  # agent_id -> множество сыгранных map_ids
        self.current_map_id = 0
        
        # Инициализация социальной сети
        self.network = nx.watts_strogatz_graph(num_players, 10, 0.1)
        
        # Создание агентов
        for i in range(self.num_players):
            agent = PlayerAgent(i, self)
            self.schedule.add(agent)
            self.agent_plays[i] = set()
            
            # Настройка социальной сети
            agent.network = set(self.network.neighbors(i))
        
        # Инициализация репозитория карт начальным набором карт
        self._generate_initial_maps(2500)
        
        # Настройка сбора данных
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Plays": self.get_total_plays,
                "Active_Maps": self.get_active_maps_count,
                "Average_Map_Quality": self.get_average_map_quality
            },
            agent_reporters={
                "Play_Count": lambda a: len(a.played_maps)
            }
        )
    
    def _generate_initial_maps(self, count):
        """Генерация начального набора карт"""
        genres = ['rock', 'pop', 'electronic', 'classical', 'jazz', 'metal', 'hiphop']
        for _ in range(count):
            self._add_new_map(genres[np.random.randint(len(genres))])
    
    def _add_new_map(self, genre):
        """Добавление новой карты в репозиторий"""
        map_data = {
            'id': self.current_map_id,
            # Игровые параметры
            'difficulty': np.random.normal(0.5, 0.2),
            'duration': np.random.normal(180, 30),
            'density': np.random.normal(0.5, 0.15),
            'object_balance': np.random.normal(0.5, 0.15),
            
            # Музыкальные характеристики
            'genre': genre,
            'artist_popularity': np.random.normal(0.5, 0.2),
            'release_year': np.random.randint(2000, 2025),
            'emotional_tone': np.random.normal(0.5, 0.2),
            
            # Характеристики маппинга
            'rhythm_sync': np.random.normal(0.7, 0.15),
            'pattern_readability': np.random.normal(0.7, 0.15),
            'design_creativity': np.random.normal(0.6, 0.2),
            'style_consistency': np.random.normal(0.7, 0.15),
            
            # Социальные факторы
            'mapper_reputation': np.random.normal(0.5, 0.2),
            'map_status': np.random.normal(0.5, 0.2),
            'tournament_usage': np.random.normal(0.3, 0.15),
            'collaborations': np.random.normal(0.3, 0.15),
            
            'creation_date': self.schedule.steps
        }
        self.map_repository.append(map_data)
        self.map_plays[self.current_map_id] = 0
        self.current_map_id += 1
    
    def get_available_maps(self):
        """Получение всех доступных карт"""
        return self.map_repository
    
    def get_map_plays(self, map_id):
        """Получение количества игр для конкретной карты"""
        return self.map_plays.get(map_id, 0)
    
    def get_agent_plays(self, agent_id):
        """Получение множества карт, сыгранных агентом"""
        return self.agent_plays.get(agent_id, set())
    
    def get_max_plays(self):
        """Получение максимального количества игр среди всех карт"""
        return max(self.map_plays.values()) if self.map_plays else 0
    
    def record_play(self, map_id, agent_id):
        """Запись игры карты агентом"""
        self.map_plays[map_id] = self.map_plays.get(map_id, 0) + 1
        self.agent_plays[agent_id].add(map_id)
    
    def get_total_plays(self):
        """Получение общего количества игр по всем картам"""
        return sum(self.map_plays.values())
    
    def get_active_maps_count(self):
        """Получение количества карт с хотя бы одной игрой"""
        return sum(1 for plays in self.map_plays.values() if plays > 0)
    
    def get_average_map_quality(self):
        """Получение среднего качества всех карт"""
        quality_scores = []
        for map_data in self.map_repository:
            # Рассчитываем общее качество карты как среднее всех её характеристик
            quality = np.mean([
                map_data['rhythm_sync'],
                map_data['pattern_readability'],
                map_data['design_creativity'],
                map_data['style_consistency']
            ])
            quality_scores.append(quality)
        return np.mean(quality_scores) if quality_scores else 0
    
    def step(self):
        """Выполнение одного шага модели"""
        # Генерация новых карт
        if self.schedule.steps % 24 == 0:  # Раз в день
            genres = ['rock', 'pop', 'electronic', 'classical', 'jazz', 'metal', 'hiphop']
            for _ in range(self.new_maps_per_day):
                self._add_new_map(genres[np.random.randint(len(genres))])
        
        # Выполнение шагов агентов
        self.schedule.step()
        
        # Сбор данных
        self.datacollector.collect(self) 