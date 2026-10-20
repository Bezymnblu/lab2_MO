import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.decomposition import FactorAnalysis
import time
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, SpectralClustering, 
                             AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch, BisectingKMeans)
from sklearn.mixture import GaussianMixture

# ==========================================
# 0. ЗАГРУЗКА И ВЫДЕЛЕНИЕ ПРИЗНАКОВ ВАРИАНТА 7
# ==========================================
# Вставляем ваш точный путь к файлу:
file_name = r"C:\Users\lygan\Downloads\Telegram Desktop\Lab2_Dataset_PCA.xlsx"

try:
    # Читаем файл со 2-й строки (без автоматических заголовков), чтобы не запутаться в объединенных ячейках
    df = pd.read_excel(file_name, header=None)
    print(f"Файл успешно загружен! Общая размерность: {df.shape}")
except FileNotFoundError:
    print(f"ОШИБКА: Файл не найден по пути: {file_name}. Проверьте имя папки или файла.")
    exit()

# Для Варианта 7 блок колонок (a, b, c, d, e, y) начинается с 36-го индекса (37-й столбец в Excel)
start_col = 36
df_var7 = df.iloc[:, start_col:start_col+6].copy()
df_var7.columns = ['a', 'b', 'c', 'd', 'e', 'y']

# Очистка данных от текстовых строк (заголовков)
df_var7 = df_var7[pd.to_numeric(df_var7['a'], errors='coerce').notna()]
df_var7 = df_var7.apply(pd.to_numeric).fillna(df_var7.mean())

print(f"Размерность очищенных числовых данных Варианта 7: {df_var7.shape}")

X_var7 = df_var7[['a', 'b', 'c', 'd', 'e']]
y_var7 = df_var7['y'].astype(int)

# ==========================================
# 1.1 И 1.2 МАТРИЦЫ КОРРЕЛЯЦИЙ И РАЗБРОСА
# ==========================================
print("\n--- Вычисление матрицы взаимных корреляций ---")
corr_matrix = X_var7.corr()
print(corr_matrix.round(3))

# 1.1 Тепловая карта
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Задание 1.1: Матрица взаимных корреляций")
plt.tight_layout()
plt.show()

# 1.2 Матрица разброса
pd.plotting.scatter_matrix(X_var7, figsize=(9, 9), c=y_var7, marker='o', 
                           hist_kwds={'bins': 20}, s=40, alpha=0.7, cmap='viridis')
plt.suptitle("Задание 1.2: Матрица разброса признаков", y=0.95, fontsize=12)
plt.show()

# Масштабирование (Z-преобразование) перед методами понижения размерности
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var7)

# ==========================================
# 1.3 И 1.4 ПРИМЕНЕНИЕ PCA, SVD, ICA И ОЦЕНКА ЗНАЧИМОСТИ
# ==========================================
print("\n--- Запуск PCA, SVD и ICA ---")

# 1. Анализ главных компонент (PCA)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
print("PCA - Объясненная дисперсия по компонентам:", pca.explained_variance_ratio_.round(4))

# 2. Сингулярное разложение (SVD)
svd = TruncatedSVD(n_components=5)
X_svd = svd.fit_transform(X_scaled)
print("SVD - Объясненная дисперсия по компонентам:", svd.explained_variance_ratio_.round(4))

# 3. Анализ независимых компонент (ICA)
# Для ICA в scikit-learn "дисперсии" как у PCA нет, поэтому оценивается среднее правдоподобие (score)
ica = FastICA(n_components=5, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# 1.4 График значимости компонент (на примере PCA)
# ==========================================
# УЛУЧШЕННОЕ ЗАДАНИЕ 1.4: ГРАФИКИ ДЛЯ ВСЕХ МЕТОДОВ
# ==========================================
# 1. График значимости для PCA
plt.figure(figsize=(6, 4))
components = np.arange(1, 6)
plt.bar(components, pca.explained_variance_ratio_, alpha=0.6, label='Индивидуальная PCA')
plt.step(components, np.cumsum(pca.explained_variance_ratio_), where='mid', label='Кумулятивная PCA')
plt.xlabel('Индекс главной компоненты')
plt.ylabel('Доля объясненной дисперсии')
plt.title('Задание 1.3 и 1.4: PCA - Значимость компонент')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 2. График значимости для SVD
plt.figure(figsize=(6, 4))
plt.bar(components, svd.explained_variance_ratio_, alpha=0.6, color='orange', label='Индивидуальная SVD')
plt.step(components, np.cumsum(svd.explained_variance_ratio_), where='mid', color='red', label='Кумулятивная SVD')
plt.xlabel('Индекс компоненты SVD')
plt.ylabel('Доля объясненной дисперсии')
plt.title('Задание 1.3 и 1.4: SVD - Значимость компонент')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 3. Оценка значимости для ICA (через FactorAnalysis для получения score, как в методичке)
ica_scores = []
for n in range(1, 6):
    # max_iter=5000 и tol=0.01 гарантируют сходимость и убирают ConvergenceWarning
    ica_test = FastICA(n_components=n, random_state=42, max_iter=5000, tol=0.01)
    X_ica_features = ica_test.fit_transform(X_scaled)
    
    # Оцениваем правдоподобие (score) полученного пространства признаков через FA
    fa_eval = FactorAnalysis(n_components=n, random_state=42)
    fa_eval.fit(X_ica_features)
    ica_scores.append(fa_eval.score(X_ica_features))

plt.figure(figsize=(6, 4))
plt.plot(components, ica_scores, marker='o', color='purple', linestyle='-')
plt.xlabel('Количество независимых компонент')
plt.ylabel('Правдоподобие (Log-Likelihood Score)')
plt.title('Задание 1.3 и 1.4: ICA - График правдоподобия')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# ==========================================
# 1.5 ПОСТРОЕНИЕ SCATTER-ГРАФИКОВ ПРОЕКЦИЙ (Biplot)
# ==========================================
def biplot(scores, loadings, labels, targets, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Нормализация компонент для красивого наложения векторов
    scalex = 1.0 / (scores[:, 0].max() - scores[:, 0].min())
    scaley = 1.0 / (scores[:, 1].max() - scores[:, 1].min())
    
    # Отображаем точки данных (Проекции)
    scatter = ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley, c=targets, cmap='viridis', alpha=0.6, edgecolors='k')
    
    # Отрисовка векторов признаков (Главные направления)
    for i in range(loadings.shape[0]):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.7, 
                 head_width=0.02, head_length=0.02, linewidth=1.2)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, labels[i], 
                color='darkred', ha='center', va='center', fontweight='bold')
                
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-1.2, 1.3)
    ax.set_ylim(-1.2, 1.3)
    fig.colorbar(scatter, ax=ax, label='Классы (y)')
    plt.tight_layout()

# Строим Biplot проекции для PCA
feature_names = ['a', 'b', 'c', 'd', 'e']

# 1. Biplot для PCA
loadings_pca = pca.components_.T
biplot(X_pca, loadings_pca, feature_names, y_var7, "Задание 1.5: PCA Biplot")
plt.show()

# 2. Biplot для SVD
loadings_svd = svd.components_.T
biplot(X_svd, loadings_svd, feature_names, y_var7, "Задание 1.5: SVD Biplot")
plt.show()

# 3. Biplot для ICA (Используем матрицу смешивания в качестве loadings)
loadings_ica = ica.mixing_
biplot(X_ica, loadings_ica, feature_names, y_var7, "Задание 1.5: ICA Biplot")
plt.show()

# ==========================================
# ДОПОЛНИТЕЛЬНО: ВЫВОД ТАБЛИЦЫ ДАННЫХ В КОНСОЛЬ
# ==========================================
#print("\n=== ИТОГОВАЯ ТАБЛИЦА ДАННЫХ ВАРИАНТА 7 ===")
# Настройка, чтобы pandas выводил все строки и столбцы без сокращений
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Печатаем очищенный датасет Варианта 7
#print(df_var7)

# ==========================================
# ЧАСТЬ 2. ЗАДАНИЕ 2.1 (КЛАСТЕРИЗАЦИЯ)
# ==========================================
print("\n====================================================")
print(" ЗАПУСК ЧАСТИ 2. ЗАДАНИЕ 2.1")
print("====================================================")

# Путь ко второму файлу (кластеризация) из вашей папки Telegram Desktop
file_clustering = r"C:\Users\lygan\Downloads\Telegram Desktop\Lab2_Dataset_Clustering.xlsx"

try:
    df_clus = pd.read_excel(file_clustering, header=None)
    print(f"Файл кластеризации успешно загружен! Общая размерность: {df_clus.shape}")
except FileNotFoundError:
    print(f"\nОШИБКА: Файл не найден по пути: {file_clustering}")
    print("Проверьте, скачан ли файл Lab2_Dataset_Clustering.xlsx")
    plt.show()
    exit()

# Для Варианта 7 во второй части выделяем 3 колонки (2 признака + y)
# Индекс начала: (7 вариант - 1) * 3 столбца = 18-й индекс (столбцы 19, 20, 21 в Excel)
start_col_clus = 18 
df_clus_v7 = df_clus.iloc[:, start_col_clus:start_col_clus+3].copy()
df_clus_v7.columns = ['Feature_1', 'Feature_2', 'y']

# Очищаем данные от текстовых строк (заголовков)
df_clus_v7 = df_clus_v7[pd.to_numeric(df_clus_v7['Feature_1'], errors='coerce').notna()]
df_clus_v7 = df_clus_v7.apply(pd.to_numeric).fillna(df_clus_v7.mean())

print(f"Размерность числовых данных Варианта 7 (Кластеризация): {df_clus_v7.shape}")

X_clus = df_clus_v7[['Feature_1', 'Feature_2']].values
y_clus_true = df_clus_v7['y'].values.astype(int)

# Масштабируем данные для корректного анализа
X_clus_scaled = StandardScaler().fit_transform(X_clus)

# Строим эталонный график пространственного распределения
plt.figure(figsize=(7, 6))
scatter_clus = plt.scatter(X_clus_scaled[:, 0], X_clus_scaled[:, 1], c=y_clus_true, cmap='viridis', alpha=0.7, edgecolors='k', s=35)
plt.title("Задание 2.1: Истинное (эталонное) распределение кластеров", fontsize=11, fontweight='bold')
plt.xlabel("Масштабированный Признак 1")
plt.ylabel("Масштабированный Признак 2")
plt.grid(True, linestyle=':', alpha=0.6)
plt.colorbar(scatter_clus, label='Эталонные метки (y)')
plt.tight_layout()
plt.show()

# ==========================================
# ЧАСТЬ 2. ЗАДАНИЕ 2.2 (МАТРИЦА АЛГОРИТМОВ)
# ==========================================
print("\n--- Запуск 12 алгоритмов кластеризации для Задания 2.2 ---")
# Автоматически определяем число кластеров по уникальным меткам в y
n_clusters = len(np.unique(y_clus_true))
print(f"Истинное число кластеров для Варианта 7: {n_clusters}")

# Словарь со всеми 12 утилитами кластеризации по методичке
algorithms = {
    "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
    "Affinity Prop": AffinityPropagation(preference=-50, random_state=42),
    "Mean-Shift": MeanShift(),
    "Spectral": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42),
    "Ward": AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
    "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
    "HDBSCAN": HDBSCAN(min_cluster_size=5),
    "OPTICS": OPTICS(min_samples=5),
    "Gaussian Mix": GaussianMixture(n_components=n_clusters, random_state=42),
    "BIRCH": Birch(n_clusters=n_clusters),
    "Bisecting KM": BisectingKMeans(n_clusters=n_clusters, random_state=42)
}

# Создаем большую сетку графиков: 3 строки на 4 столбца
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, algo) in enumerate(algorithms.items()):
    ax = axes[idx]
    t0 = time.time()
    
    try:
        # Обучаем модели и собираем метки кластеров
        if name == "Gaussian Mix":
            algo.fit(X_clus_scaled)
            labels = algo.predict(X_clus_scaled)
        else:
            labels = algo.fit_predict(X_clus_scaled)
    except Exception as e:
        print(f"Ошибка при расчете метода {name}: {e}")
        labels = np.zeros(len(X_clus_scaled))
        
    t1 = time.time()
    elapsed_time = t1 - t0
    
    # Отрисовываем получившийся результат на сетке подграфиков
    ax.scatter(X_clus_scaled[:, 0], X_clus_scaled[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    ax.set_title(f"{name}\nВремя: {elapsed_time:.2f}с", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Задание 2.2: Сравнение 12 алгоритмов кластеризации", fontsize=15, y=0.98, fontweight='bold')
plt.tight_layout()
plt.show()
print("Расчеты кластеризации завершены. Матрица графиков подготовлена.")

# ==========================================
# УЛУЧШЕННЫЕ ЗАДАНИЯ 2.3 И 2.4: АНАЛИЗ ПЛОТНОСТНЫХ ВЫБРОСОВ
# ==========================================
print("\n--- Запуск расширенного анализа аномалий (DBSCAN, HDBSCAN, OPTICS) ---")

# 1. Оптимизированный DBSCAN
best_eps = 0.45
best_min_samples = 4
dbscan_opt = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan_opt.fit_predict(X_clus_scaled)

# 2. Оптимизированный HDBSCAN (Шум также помечается как -1)
hdbscan_opt = HDBSCAN(min_cluster_size=4, min_samples=3)
hdbscan_labels = hdbscan_opt.fit_predict(X_clus_scaled)

# 3. Оптимизированный OPTICS (Шум также помечается как -1)
optics_opt = OPTICS(min_samples=4, xi=0.05)
optics_labels = optics_opt.fit_predict(X_clus_scaled)

# Выводим статистику выбросов по всем методам в консоль
print(f"Выбросы DBSCAN (eps={best_eps}): {np.sum(dbscan_labels == -1)} точек")
print(f"Выбросы HDBSCAN: {np.sum(hdbscan_labels == -1)} точек")
print(f"Выбросы OPTICS: {np.sum(optics_labels == -1)} точек")

# Строим сетку графиков для отображения аномалий по разным плотностным методам
fig_anom, axes_anom = plt.subplots(1, 3, figsize=(18, 5))
density_results = {"DBSCAN": dbscan_labels, "HDBSCAN": hdbscan_labels, "OPTICS": optics_labels}

for idx, (name, labels) in enumerate(density_results.items()):
    ax = axes_anom[idx]
    normal = labels != -1
    anomaly = labels == -1
    
    # Сферические нормальные точки кластеров
    ax.scatter(X_clus_scaled[normal, 0], X_clus_scaled[normal, 1], c=labels[normal], cmap='tab10', s=35, alpha=0.7, edgecolors='k')
    # Маркеры-крестики для аномалий (Шума)
    if np.sum(anomaly) > 0:
        ax.scatter(X_clus_scaled[anomaly, 0], X_clus_scaled[anomaly, 1], color='black', marker='x', s=50, linewidths=1.5, label='Выбросы')
    
    est_c = len(set(labels)) - (1 if -1 in labels else 0)
    ax.set_title(f"Метод: {name}\nКластеров: {est_c}, Аномалий: {np.sum(anomaly)}")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best')

plt.suptitle("Задание 2.3 и 2.4: Сравнение эффективности отделения выбросов методами плотности", fontsize=13, y=0.98, fontweight='bold')
plt.tight_layout()

# Отображаем все сгенерированные графики и удерживаем окна на экране
print("\nВсе графики построены. Закройте графические окна для завершения программы.")
plt.show()
