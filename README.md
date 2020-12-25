# grav_potentials
Модуль состоящий из набора затабулированных трёхмерных нормированных гравитационных потенциалов различных однородных фигур: тора, эллипсоида, и т.д. Все формульные выражения для потенциалов фигур берутся из учебника Кондратьева Б. П.: <Теория потенциала. Новые методы и задачи с решениями> //М.: Мир. – 2007.

upd 25.12.2020
Добавлен внутренний потенциал параллелепипеда и цилиндра с эллиптическим сечением. При тестировании проверено, что: потенциал круглого диска и потенциал кольца на бесконечности совпадают вдоль обеих цилиндрических координат, т.к. их асимптотики ведут себя как ~1/r. Также проверено: потенциал шара и эллипсоида, при равенстве его больших полуосей радиусу шара, на бесконечности также совпадают, благодаря асимптотике ~1/r. Проведено сравнение потенциала кубоида в центре с теоретическим значением, полученным в монографии. Проверено, что потенциал кубоида в центре отнесённый к потенциалу кубоида в вершине = 2. Проверено, что внутренний потенциал цилиндра с эллиптическим сечением совпадает с потенциалом цилиндра с круглым сечением, при равенстве больших полуосей в сечении цилиндра. Все отношения проверялись до точности ~1e-5.

upd 13.11.2020
Добавлен потенциал эллипсоида, диска, кольца и шара. Добавлены докстринги. Положено начало процессу тестирования: в частности проверено, что при определённых значениях, потенциал эллипсоида при равных больших полуосях переходит в потенциал шара. Также проект собран в пакеты.

upd 30.10.2020.
На данный момент в модуле присутствует выражение для потенциала кругового тора (формулы спрограммированы по книге из источника со с. 210). Потенциал определён во всём пространстве. Данная формула оттестирована в файле test.py, в котором строятся радиальные профили потенциала тора, а также исследуется зависимость максимума потенциала в радиальном профиле от величины отношения радиуса рукава тора к расстоянию центра этого рукава (такой параметр определяет геометрию тора). Сгенерированные графики приложены в директории /figures.
