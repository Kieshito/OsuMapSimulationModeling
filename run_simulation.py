import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models.market_model import MarketModel
from tqdm import tqdm
import time
import numpy as np

def run_simulation(days=30, num_players=100):
    """Запуск симуляции на указанное количество дней"""
    print(f"Инициализация модели с {num_players} игроками...")
    model = MarketModel(num_players=num_players, new_maps_per_day=10)
    
    # Запуск симуляции с индикатором прогресса
    total_steps = days * 24
    print(f"Запуск симуляции на {days} дней ({total_steps} шагов)...")
    
    start_time = time.time()
    for _ in tqdm(range(total_steps), desc="Прогресс симуляции"):
        model.step()
    
    end_time = time.time()
    print(f"\nСимуляция завершена за {end_time - start_time:.2f} секунд")
    return model

def analyze_results(model):
    # Get model data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Создание DataFrame с данными о картах
    map_data = []
    for map_info in model.map_repository:
        map_data.append({
            'map_id': map_info['id'],
            'plays': model.map_plays[map_info['id']],
            'difficulty': map_info['difficulty'],
            'duration': map_info['duration'],
            'density': map_info['density'],
            'genre': map_info['genre'],
            'artist_popularity': map_info['artist_popularity'],
            'release_year': map_info['release_year'],
            'rhythm_sync': map_info['rhythm_sync'],
            'pattern_readability': map_info['pattern_readability'],
            'design_creativity': map_info['design_creativity'],
            'style_consistency': map_info['style_consistency'],
            'mapper_reputation': map_info['mapper_reputation'],
            'map_status': map_info['map_status'],
            'tournament_usage': map_info['tournament_usage'],
            'collaborations': map_info['collaborations']
        })
    map_plays = pd.DataFrame(map_data)
    
    """Анализ и визуализация результатов симуляции"""
    print("Анализ результатов...")
    
    # Сбор данных о весах факторов от всех агентов
    factor_importance = {
        'Игровые параметры': {
            'Звездный рейтинг': [],
            'Длительность карты': [],
            'Плотность объектов': [],
            'Баланс типов объектов': []
        },
        'Музыкальные характеристики': {
            'Жанр музыки': [],
            'Популярность исполнителя': [],
            'Год выпуска трека': [],
            'Эмоциональный тон': []
        },
        'Характеристики маппинга': {
            'Синхронизация с музыкой': [],
            'Читаемость паттернов': [],
            'Креативность дизайна': [],
            'Консистентность стиля': []
        },
        'Социальные факторы': {
            'Репутация маппера': [],
            'Статус карты': [],
            'Использование в турнирах': [],
            'Коллаборации': []
        }
    }
    
    # Сбор весов от всех агентов
    for agent in model.schedule.agents:
        # Игровые параметры
        factor_importance['Игровые параметры']['Звездный рейтинг'].append(agent.game_weights['star_rating'])
        factor_importance['Игровые параметры']['Длительность карты'].append(agent.game_weights['duration'])
        factor_importance['Игровые параметры']['Плотность объектов'].append(agent.game_weights['object_density'])
        factor_importance['Игровые параметры']['Баланс типов объектов'].append(agent.game_weights['object_balance'])
        
        # Музыкальные характеристики
        factor_importance['Музыкальные характеристики']['Жанр музыки'].append(agent.music_weights['genre'])
        factor_importance['Музыкальные характеристики']['Популярность исполнителя'].append(agent.music_weights['artist_popularity'])
        factor_importance['Музыкальные характеристики']['Год выпуска трека'].append(agent.music_weights['release_year'])
        factor_importance['Музыкальные характеристики']['Эмоциональный тон'].append(agent.music_weights['emotional_tone'])
        
        # Характеристики маппинга
        factor_importance['Характеристики маппинга']['Синхронизация с музыкой'].append(agent.mapping_weights['rhythm_sync'])
        factor_importance['Характеристики маппинга']['Читаемость паттернов'].append(agent.mapping_weights['pattern_readability'])
        factor_importance['Характеристики маппинга']['Креативность дизайна'].append(agent.mapping_weights['design_creativity'])
        factor_importance['Характеристики маппинга']['Консистентность стиля'].append(agent.mapping_weights['style_consistency'])
        
        # Социальные факторы
        factor_importance['Социальные факторы']['Репутация маппера'].append(agent.social_weights['mapper_reputation'])
        factor_importance['Социальные факторы']['Статус карты'].append(agent.social_weights['map_status'])
        factor_importance['Социальные факторы']['Использование в турнирах'].append(agent.social_weights['tournament_usage'])
        factor_importance['Социальные факторы']['Коллаборации'].append(agent.social_weights['collaborations'])
    
    # Расчет средних значений и создание таблицы
    results = []
    for category, factors in factor_importance.items():
        for factor, values in factors.items():
            mean_importance = np.mean(values) * 100  # Преобразование в проценты
            results.append({
                'Категория факторов': category,
                'Фактор': factor,
                'Относительная значимость (%)': mean_importance  # Убираем форматирование строки
            })
    
    # Создание DataFrame и сохранение в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('factor_importance.csv', index=False)
    
    # Вывод результатов
    print("\nОтносительная значимость факторов:")
    print(results_df.to_string(index=False))
    
    # Создание визуализации
    plt.figure(figsize=(15, 10))
    
    # График 1: Среднее количество игр в день
    plt.subplot(2, 2, 1)
    
    # Рассчитываем среднее количество игр в день
    daily_plays = model_data['Total_Plays'].diff().fillna(0)  # Разница между соседними значениями
    plt.plot(model_data.index / 24, daily_plays, label='Игры в день')
    
    # Добавляем скользящее среднее для сглаживания
    window = 24  # 24 часа = 1 день
    rolling_mean = daily_plays.rolling(window=window).mean()
    plt.plot(model_data.index / 24, rolling_mean, '--', label=f'Среднее за {window} часов')
    
    plt.title('Количество игр в день')
    plt.xlabel('Дни')
    plt.ylabel('Количество игр')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию с итоговой статистикой
    total_plays = model_data['Total_Plays'].iloc[-1]
    avg_plays_per_day = total_plays / (len(model_data) / 24)
    plt.annotate(f'Всего игр: {total_plays:,.0f}\nСреднее в день: {avg_plays_per_day:,.0f}',
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # График 2: Активные карты с течением времени
    plt.subplot(2, 2, 2)
    
    # Общее количество карт (линейный рост)
    total_maps = len(model.map_repository)
    days = len(model_data) / 24
    total_maps_line = np.linspace(0, total_maps, len(model_data))
    plt.plot(model_data.index / 24, total_maps_line, '--', label='Общее количество карт', color='gray')
    
    # Активные карты
    plt.plot(model_data.index / 24, model_data['Active_Maps'], label='Активные карты', color='blue')
    
    plt.title('Активные карты с течением времени')
    plt.xlabel('Дни')
    plt.ylabel('Количество карт')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию с итоговой статистикой
    final_active = model_data['Active_Maps'].iloc[-1]
    final_percentage = (final_active / total_maps) * 100
    plt.annotate(f'Всего карт: {total_maps}\nАктивных: {final_active} ({final_percentage:.1f}%)',
                xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # График 3: Распределение значимости факторов по категориям
    plt.subplot(2, 2, 3)
    category_importance = results_df.groupby('Категория факторов')['Относительная значимость (%)'].sum()
    plt.pie(category_importance.values, labels=category_importance.index, autopct='%1.1f%%')
    plt.title('Распределение значимости по категориям')
    
    # График 4: Топ-5 наиболее значимых факторов
    plt.subplot(2, 2, 4)
    top_factors = results_df.nlargest(5, 'Относительная значимость (%)')
    plt.barh(top_factors['Фактор'], top_factors['Относительная значимость (%)'])
    plt.title('Топ-5 наиболее значимых факторов')
    plt.xlabel('Относительная значимость (%)')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()
    
    # Сохранение детальных результатов
    model_data.to_csv('model_data.csv')
    agent_data.to_csv('agent_data.csv')
    map_plays.to_csv('map_plays.csv')

if __name__ == "__main__":
    # Запуск симуляции
    print("Запуск симуляции...")
    model = run_simulation(days=14, num_players=100)
    
    # Анализ результатов
    print("Анализ результатов...")
    analyze_results(model)
    
    print("Симуляция завершена! Результаты сохранены в CSV файлы, визуализация сохранена как 'simulation_results.png'")

#TODO: Так же добавить веса к категориям музыки. Более реалистичные, игра танцевальная ритмическая для виабушников.
# Возможно немного переделать выбор карты, изменить функцию полезности, добавить фактор случайности, что игрок выберет вообще полную фигню