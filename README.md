Введение

В данной работе представлен простой инструмент для классифицирования изображений. Основная цель данной работы – изучить
библиотеку PyTorch для создания нейросети архитектуры трансформер, способной обучиться на наборе изображений и сохранить
выведенные веса и параметры как отдельный файл. Программа реализована в консоли и имеет 2 режима: обучение и
классификация с использованием уже готовых параметров. 

Описание программы

Программа предоставляет следующие возможности:

1.	На примере показывает процесс обучения нейросети класса трансформер на подготовленном пользователем или заранее
готовом наборе изображений
2.	Может использоваться для классификации изображений
Технические требования
Программа требует установки следующих зависимостей:

•	Python 3.11

•	Torch 2.5.1+cu124

•	Torchvision 0.20.1+cu124

•	timm 1.0.11

•	tqdm 4.67.1

•	(Необязательно, но желательно)

o	CUDA>=12.6

Описание работы программы

Программа предоставляет полный цикл работы с моделью глубокого обучения: от загрузки и подготовки данных до обучения,
тестирования и использования для классификации изображений. Она демонстрирует основные принципы работы с PyTorch, 
включая создание нейронной сети, использование загрузчиков данных, обучение и тестирование модели.

1.	Импорт библиотек: импортируются необходимые библиотеки, такие как PyTorch, torchvision и PIL для работы с
изображениями.
2.	Определение путей к данным: задаются пути к директориям с обучающими, валидационными и тестовыми данными.
3.	Определение преобразований: определяются преобразования для изображений, включая изменение размера, преобразование в
тензор и нормализацию.
4.	Загрузка данных: загружаются обучающие, валидационные и тестовые данные с использованием ImageFolder, что позволяет
загружать изображения из папок, где каждая папка соответствует классу.
5.	Создание загрузчиков данных: создаются загрузчики данных (DataLoader) для пакетной обработки данных с заданным
размером пакета.
6.	Определение модели: создаётся класс Model, который наследует nn.Module. В представленной программе внутри класса
определяются свёрточные слои, слои подвыборки, полносвязные слои и функция активации ReLU. Модель принимает на вход
изображения и возвращает предсказания классов. (Стоит обратить внимание на файл Models, расположенный в репозитории.
Он содержит примеры использованных моделей, при желании пользователь легко может заменить представленную свёрточную
модель CNN на более простую или сложную)
7.	Определение устройства: определяется, будет ли использоваться GPU или CPU для вычислений.
8.	Инициализация модели: создаётся экземпляр модели Model, и пользователю предлагается выбрать, хочет ли он переобучить
 модель (будет необходимо ввести 0) или использовать уже обученные веса(будет необходимо ввести 1)(Примечание: при
попытке использовать обученную модель необходимо убедится, что файл saved_transformer_model.pth существует и находится
по указанному адресу. В противном случае будет необходимо запустить процесс обучения).
9.	Загрузка весов модели: если пользователь выбирает использование готовых весов и параметров, то модель загружает
необходимые данные из файла.
10.	Определение оптимизатора: инициализируется оптимизатор Adam для обновления параметров модели.
11.	Функция обучения: определяется функция train, которая выполняет обучение модели на обучающих данных и валидацию на
валидационных данных. В конце каждого цикла выводятся значения потерь и точности.
12.	Функция тестирования: определяется функция test, которая оценивает производительность модели на тестовых данных и
выводит точность. Используется в случае, если пользователь выбрал использование готовых весов и параметров.
13.	Основной цикл: в зависимости от выбора пользователя, программа либо обучает модель, либо тестирует её.
14.	Предсказания для новых изображений: программа загружает три изображения, преобразует их и выполняет предсказания,
выводя классы для каждого изображения.

Использование

Для корректной работы программы необходимо: в зависимости от требуемого режима программы убедится в наличии директорий с изображениями для обучения программы и/или файла saved_transformer_model.pth для работы нейросети в роли классификатора. Для лучшей производительности необходимо установить CUDA.
Примечание: далее есть ссылка на Яндекс.Диск, где расположены готовый архив изображений и файл saved_transformer_model.pth, результат обучения на вышеупомянутых изображениях.

Яндекс.Диск: https://disk.yandex.ru/d/a--41BXV54K3ZQ
