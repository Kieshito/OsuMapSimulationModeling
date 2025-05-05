import numpy as np
from mesa import Agent

class PlayerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Генерация случайных весов категорий
        category_weights = np.random.dirichlet(np.ones(4))  # Генерирует 4 случайных числа, сумма которых = 1
        self.category_weights = {
            'game': category_weights[0],    # Игровые параметры
            'music': category_weights[1],   # Музыкальные характеристики
            'mapping': category_weights[2], # Характеристики маппинга
            'social': category_weights[3]   # Социальные факторы
        }
        
        # Генерация случайных весов внутри категорий
        # Игровые параметры
        game_weights = np.random.dirichlet(np.ones(4))
        self.game_weights = {
            'star_rating': game_weights[0],      # Сложность карты
            'duration': game_weights[1],         # Длительность
            'object_density': game_weights[2],   # Плотность объектов
            'object_balance': game_weights[3]    # Баланс типов объектов
        }
        
        # Музыкальные характеристики
        music_weights = np.random.dirichlet(np.ones(4))
        self.music_weights = {
            'genre': music_weights[0],            # Жанр музыки
            'artist_popularity': music_weights[1], # Популярность исполнителя
            'release_year': music_weights[2],     # Год выпуска
            'emotional_tone': music_weights[3]    # Эмоциональный тон
        }
        
        # Характеристики маппинга
        mapping_weights = np.random.dirichlet(np.ones(4))
        self.mapping_weights = {
            'rhythm_sync': mapping_weights[0],      # Синхронизация с музыкой
            'pattern_readability': mapping_weights[1], # Читаемость паттернов
            'design_creativity': mapping_weights[2],   # Креативность дизайна
            'style_consistency': mapping_weights[3]    # Консистентность стиля
        }
        
        # Социальные факторы
        social_weights = np.random.dirichlet(np.ones(4))
        self.social_weights = {
            'mapper_reputation': social_weights[0],   # Репутация маппера
            'map_status': social_weights[1],          # Статус карты
            'tournament_usage': social_weights[2],     # Использование в турнирах
            'collaborations': social_weights[3]        # Коллаборации
        }
        
        # Базовые характеристики
        self.skill_level = np.random.normal(0.5, 0.15)
        self.music_preferences = self._generate_music_preferences()
        self.duration_preference = np.random.normal(180, 30)  # Предпочитаемая длительность карты в секундах
        self.density_preference = np.random.normal(0.5, 0.15)  # Предпочитаемая плотность нот
        self.tolerance = np.random.normal(0.2, 0.05)  # Толерантность к несоответствию сложности
        
        # Параметры активности
        self.play_frequency = np.random.poisson(100)  # Среднее количество игр в день
        self.active_hours = np.random.normal(2.5, 0.5)  # Среднее время активности в часах
        self.active_start = np.random.normal(14, 3)  # Среднее время начала активности (14:00)
        self.is_active = False  # Флаг активности
        self.current_play_time = 0  # Время, затраченное на текущую игру
        self.current_map = None  # Текущая играемая карта
        
        # Свойства социальной сети
        self.network = set()  # Будет заполнено моделью
        
        # Веса предпочтений
        self.alpha = np.random.normal(0.3, 0.1)  # Вес соответствия навыку
        self.beta = np.random.normal(0.2, 0.1)   # Вес соответствия музыке
        self.gamma = np.random.normal(0.3, 0.1)  # Вес качества карты
        self.delta = np.random.normal(0.2, 0.1)  # Вес социального влияния
        
        # История игр
        self.played_maps = set()
        
        # Нормализация весов
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Нормализация весов для каждой категории и между категориями"""
        # Нормализация весов внутри категорий
        for weights in [self.game_weights, self.music_weights, 
                       self.mapping_weights, self.social_weights]:
            total = sum(weights.values())
            for key in weights:
                weights[key] = max(0, weights[key] / total)
        
        # Применение весов категорий к внутренним весам
        for key in self.game_weights:
            self.game_weights[key] *= self.category_weights['game']
        for key in self.music_weights:
            self.music_weights[key] *= self.category_weights['music']
        for key in self.mapping_weights:
            self.mapping_weights[key] *= self.category_weights['mapping']
        for key in self.social_weights:
            self.social_weights[key] *= self.category_weights['social']
    
    def _generate_music_preferences(self):
        """Генерация случайных предпочтений по музыкальным жанрам"""
        genres = ['rock', 'pop', 'electronic', 'classical', 'jazz', 'metal', 'hiphop']
        preferences = np.random.dirichlet(np.ones(len(genres)))
        return dict(zip(genres, preferences))
    
    def calculate_utility(self, map_data):
        """Расчет привлекательности карты для данного игрока"""
        # Игровые параметры
        game_utility = (
            self.game_weights['star_rating'] * self._calculate_star_rating_match(map_data['difficulty']) +
            self.game_weights['duration'] * self._calculate_duration_match(map_data['duration']) +
            self.game_weights['object_density'] * self._calculate_density_match(map_data['density']) +
            self.game_weights['object_balance'] * self._calculate_balance_match(map_data['object_balance'])
        )
        
        # Музыкальные характеристики
        music_utility = (
            self.music_weights['genre'] * self._calculate_music_match(map_data['genre']) +
            self.music_weights['artist_popularity'] * map_data['artist_popularity'] +
            self.music_weights['release_year'] * self._calculate_year_match(map_data['release_year']) +
            self.music_weights['emotional_tone'] * self._calculate_emotional_match(map_data['emotional_tone'])
        )
        
        # Характеристики маппинга
        mapping_utility = (
            self.mapping_weights['rhythm_sync'] * map_data['rhythm_sync'] +
            self.mapping_weights['pattern_readability'] * map_data['pattern_readability'] +
            self.mapping_weights['design_creativity'] * map_data['design_creativity'] +
            self.mapping_weights['style_consistency'] * map_data['style_consistency']
        )
        
        # Социальные факторы
        social_utility = (
            self.social_weights['mapper_reputation'] * map_data['mapper_reputation'] +
            self.social_weights['map_status'] * map_data['map_status'] +
            self.social_weights['tournament_usage'] * map_data['tournament_usage'] +
            self.social_weights['collaborations'] * map_data['collaborations']
        )
        
        # Общая полезность
        utility = (
            game_utility +
            music_utility +
            mapping_utility +
            social_utility +
            np.random.normal(0, 0.1)  # Небольшой случайный шум
        )
        
        return max(0, min(1, utility))
    
    def _calculate_star_rating_match(self, difficulty):
        """Расчет соответствия звездного рейтинга уровню навыка"""
        optimal_difficulty = self.skill_level * (1 + 0.1)
        diff = abs(difficulty - optimal_difficulty)
        return max(0, 1 - (diff / 0.2) ** 2)
    
    def _calculate_duration_match(self, duration):
        """Расчет соответствия длительности"""
        return max(0, 1 - abs(duration - 180) / 120)
    
    def _calculate_density_match(self, density):
        """Расчет соответствия плотности объектов"""
        return max(0, 1 - abs(density - 0.5) / 0.5)
    
    def _calculate_balance_match(self, balance):
        """Расчет соответствия баланса объектов"""
        return balance
    
    def _calculate_music_match(self, genre):
        """Расчет соответствия жанра"""
        return self.music_preferences.get(genre, 0)
    
    def _calculate_year_match(self, year):
        """Расчет соответствия года выпуска"""
        current_year = 2024
        return max(0, 1 - abs(year - current_year) / 20)
    
    def _calculate_emotional_match(self, emotional_tone):
        """Расчет соответствия эмоционального тона"""
        return emotional_tone
    
    def _calculate_social_influence(self, map_id):
        """Расчет социального влияния для карты"""
        if not self.network:
            return 0
            
        # Расчет влияния от друзей
        friend_plays = sum(1 for friend in self.network 
                         if map_id in self.model.get_agent_plays(friend))
        friend_influence = friend_plays / len(self.network)
        
        # Расчет влияния глобальной популярности
        global_plays = self.model.get_map_plays(map_id)
        max_plays = self.model.get_max_plays()
        global_influence = global_plays / max_plays if max_plays > 0 else 0
        
        # Комбинирование влияний (70% друзья, 30% глобальное)
        return 0.7 * friend_influence + 0.3 * global_influence
    
    def _is_active_time(self, current_hour):
        """Проверка, активен ли агент в текущий час"""
        # Проверяем, находится ли текущий час в пределах времени активности
        active_end = (self.active_start + self.active_hours) % 24
        if self.active_start <= active_end:
            return self.active_start <= current_hour <= active_end
        else:
            return current_hour >= self.active_start or current_hour <= active_end
    
    def step(self):
        """Выполнение одного шага поведения агента"""
        current_hour = self.model.schedule.steps % 24
        
        # Если агент не активен, проверяем, не пора ли начать активность
        if not self.is_active:
            if self._is_active_time(current_hour):
                self.is_active = True
                self.remaining_plays = np.random.poisson(self.play_frequency)
            return
        
        # Если агент активен, но не играет в карту
        if self.current_map is None:
            if self.remaining_plays > 0:
                # Выбираем новую карту для игры
                available_maps = self.model.get_available_maps()
                if available_maps:
                    utilities = [self.calculate_utility(map_data) for map_data in available_maps]
                    selected_map = available_maps[np.argmax(utilities)]
                    self.current_map = selected_map
                    self.current_play_time = 0
                    self.remaining_plays -= 1
            else:
                # Если все игры на сегодня сыграны, деактивируем агента
                self.is_active = False
                return
        
        # Если агент играет в карту
        if self.current_map is not None:
            # Увеличиваем время игры
            self.current_play_time += 1
            
            # Если карта сыграна (время игры >= длительности карты)
            if self.current_play_time >= self.current_map['duration'] / 60:  # Конвертируем секунды в минуты
                self.played_maps.add(self.current_map['id'])
                self.model.record_play(self.current_map['id'], self.unique_id)
                self.current_map = None
                self.current_play_time = 0 