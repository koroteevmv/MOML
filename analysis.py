import manim as m
import numpy as np
from math import sin, ceil
from theming import LinearTransformationScene_, ThreeDScene_, Scene_


# ============================================================
# 7. Функция как отображение
# ============================================================
class FunctionMapping(Scene_):
    """
    Два множества (определения и значений), стрелки от каждого элемента первого к единственному элементу второго.
    """
    def construct(self):
        setA = m.Rectangle(width=2, height=2, color=m.BLUE, fill_opacity=0.1).shift(m.LEFT*3)
        setB = m.Rectangle(width=2, height=2, color=m.RED, fill_opacity=0.1).shift(m.RIGHT*3)
        self.add(setA, setB)
        self.add(m.Text("Область определения", font_size=20).next_to(setA, m.DOWN))
        self.add(m.Text("Область значений", font_size=20).next_to(setB, m.DOWN))

        pointsA = [m.Dot(setA.get_left()+m.RIGHT*0.5+m.UP*0.5, color=m.BLUE),
                   m.Dot(setA.get_left()+m.RIGHT*0.5+m.DOWN*0.5, color=m.BLUE),
                   m.Dot(setA.get_left()+m.RIGHT*1.5+m.UP*0.5, color=m.BLUE),
                   m.Dot(setA.get_left()+m.RIGHT*1.5+m.DOWN*0.5, color=m.BLUE)]
        pointsB = [m.Dot(setB.get_left()+m.RIGHT*0.5+m.UP*0.5, color=m.RED),
                   m.Dot(setB.get_left()+m.RIGHT*1.0+m.DOWN*0.5, color=m.RED),
                   m.Dot(setB.get_left()+m.RIGHT*1.5+m.UP*0.5, color=m.RED)]
        self.add(*pointsA, *pointsB)

        mapping = [(0,0), (1,0), (2,1), (3,2)]
        for i, j in mapping:
            self.add(m.Arrow(pointsA[i].get_center(), pointsB[j].get_center(),
                             color=m.YELLOW, stroke_width=2, buff=0.2))

        caption = m.Text("Функция: каждый вход имеет единственный выход", font_size=24, color=m.YELLOW)
        caption.to_edge(m.DOWN).shift(m.UP * 0.2)
        self.add(caption)
        self.wait(2)



# ------------------------------------------------------------
# 1. Экспонента, логарифм и парабола
# ------------------------------------------------------------
class ExpLogParabola(Scene_):
    """
    Графики экспоненты, логарифма и квадратичной параболы на одной плоскости.
    Экспонента быстро растёт, логарифм медленно, парабола – посередине.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-2, 4, 1],
            y_range=[-2, 6, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        # Экспонента e^x
        exp_graph = axes.plot(lambda x: np.exp(x), x_range=[-2, 2.5], color=m.BLUE)
        exp_label = m.MathTex(r"e^x", color=m.BLUE).next_to(exp_graph.get_end(), m.UR)

        # Логарифм ln(x+2) чтобы начать с -2
        log_graph = axes.plot(lambda x: np.log(x + 2.1), x_range=[-1.9, 4], color=m.RED)
        log_label = m.MathTex(r"\ln(x+2)", color=m.RED).next_to(log_graph.get_end(), m.UR)

        # Парабола x^2/3 чтобы не уходила слишком высоко
        parab_graph = axes.plot(lambda x: x**2 / 3, x_range=[-2, 4], color=m.GREEN)
        parab_label = m.MathTex(r"x^2/3", color=m.GREEN).next_to(parab_graph.get_end(), m.UR)

        self.play(
            m.Create(exp_graph), m.Write(exp_label),
            m.Create(log_graph), m.Write(log_label),
            m.Create(parab_graph), m.Write(parab_label),
            run_time=2
        )

        title = m.Text("Экспонента, логарифм и парабола", font_size=28, color=m.WHITE)
        title.to_edge(m.UP).shift(m.DOWN * 0.1)
        self.add(title)

        # Пояснение
        note = m.Text("Экспонента растёт быстрее всех, логарифм – медленнее всего", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN).shift(m.UP * 0.1)
        self.add(note)
        self.wait(2)


# ------------------------------------------------------------
# 2. Сигмоида и ReLU
# ------------------------------------------------------------
class SigmoidReLU(Scene_):
    """
    Графики сигмоиды (S-образная) и ReLU (кусочно-линейная).
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 2, 0.5],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        # Сигмоида
        sig_graph = axes.plot(lambda x: 1/(1 + np.exp(-x)), x_range=[-4, 4], color=m.BLUE)
        sig_label = m.MathTex(r"\sigma(x) = \frac{1}{1+e^{-x}}", color=m.BLUE)
        sig_label.shift(m.UP * 0 + m.RIGHT * 3)

        # ReLU
        relu_graph = axes.plot(lambda x: max(0, x), x_range=[-4, 2], color=m.RED)
        relu_label = m.MathTex(r"\text{ReLU}(x) = \max(0, x)", color=m.RED)
        relu_label.next_to(m.RIGHT).shift(m.UP * 2 + m.RIGHT * 0.5)

        self.play(
            m.Create(sig_graph), m.Write(sig_label),
            m.Create(relu_graph), m.Write(relu_label),
            run_time=2
        )

        # # Горизонтальные линии для сигмоиды
        # line0 = axes.get_horizontal_line((0, 0, 0), color=m.GRAY, stroke_width=1)
        # line1 = axes.get_horizontal_line((0, 1, 0), color=m.GRAY, stroke_width=1)
        # self.add(line0, line1)
        # self.add(m.MathTex("0", color=m.GRAY).next_to(line0, m.LEFT))
        # self.add(m.MathTex("1", color=m.GRAY).next_to(line1, m.LEFT))

        title = m.Text("Сигмоида и ReLU", font_size=28, color=m.WHITE).to_edge(m.UP).shift(m.DOWN * 0.1)
        self.add(title)
        note = m.Text("Сигмоида – гладкая, ReLU – кусочно-линейная", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN).shift(m.UP * 0.1)
        self.add(note)
        self.wait(2)


# ------------------------------------------------------------
# 3. Теорема о сжатии (две последовательности, стремящиеся к L)
# ------------------------------------------------------------
class SqueezeTheorem(Scene_):
    """
    Три графика: нижняя (синяя), верхняя (красная) и зажатая (зелёная).
    Все стягиваются к горизонтальной линии L.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 10, 1],
            y_range=[-1, 3, 0.5],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes = axes.add_coordinates()
        self.add(axes)

        L = 1.0

        # Нижняя последовательность: 1 - 1/(x+1)
        lower_graph = axes.plot(lambda x: L - 1/(x+1), x_range=[0.1, 10], color=m.BLUE)
        lower_label = m.MathTex(r"a_n = 1 - \frac{1}{n+1}", color=m.BLUE).next_to(lower_graph.get_start(), m.DL).scale(0.6)

        # Верхняя: 1 + 1/(x+1)
        upper_graph = axes.plot(lambda x: L + 1/(x+1), x_range=[0.1, 10], color=m.RED)
        upper_label = m.MathTex(r"b_n = 1 + \frac{1}{n+1}", color=m.RED).next_to(upper_graph.get_start(), m.UL).scale(0.6)

        # Зажатая: 1 + 0.5*sin(1/x)/(x+1) (колеблется, но стремится к L)
        squeezed_graph = axes.plot(
            lambda x: L + 0.5 * np.sin(1/x) / (x+1),
            x_range=[0.2, 10],
            color=m.GREEN,
            discontinuities=[0]
        )
        squeezed_label = m.MathTex(r"c_n", color=m.GREEN).next_to(squeezed_graph.get_end(), m.UR)

        # Горизонтальная линия L
        L_line = axes.get_horizontal_line(axes @ (9.0, 1.0), color=m.YELLOW, stroke_width=2)
        L_label = m.MathTex(r"L", color=m.YELLOW).next_to(L_line, m.DR)

        self.play(
            m.Create(lower_graph), m.Write(lower_label),
            m.Create(upper_graph), m.Write(upper_label),
            m.Create(squeezed_graph), m.Write(squeezed_label),
            m.Create(L_line), m.Write(L_label),
            run_time=2
        )

        title = m.Text("Теорема о сжатии", font_size=28, color=m.WHITE).to_edge(m.UP)
        self.add(title)
        note = m.Text("Нижняя и верхняя сходятся к L → зажатая тоже", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN)
        self.add(note)
        self.wait(2)


# ------------------------------------------------------------
# 4. Скорости сходимости (линейная vs квадратичная) в логарифмическом масштабе
# ------------------------------------------------------------
class ConvergenceRates(Scene_):
    """
    Графики ошибки |a_n - A| в логарифмическом масштабе.
    Линейная сходимость – прямая, квадратичная – резко падает.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[0, 10, 1],
            y_range=[-8, 0, 1],  # логарифм ошибки
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.y_axis.add_labels({v: f"$10^{{{v}}}$" for v in range(-8, 1)})
        axes.x_axis.add_labels({i: str(i) for i in range(0, 11)})
        self.add(axes)

        # Линейная: log10(|A - a_n|) = -0.5 n (прямая)
        n_vals = np.linspace(0.1, 9.9, 100)
        linear_vals = -0.5 * n_vals
        linear_graph = axes.plot_line_graph(
            x_values=n_vals,
            y_values=linear_vals,
            line_color=m.BLUE,
            add_vertex_dots=False
        )
        linear_label = m.Text(r"Линейная", color=m.BLUE).next_to(linear_graph["line_graph"].get_end(), m.UR).scale(0.5).shift(m.LEFT)

        # Квадратичная: log10(|A - a_n|) = -1.5 * 2^n (резкий спад)
        quad_n = np.linspace(0.1, 5, 50)
        quad_vals = -1.5 * (2**quad_n)
        # ограничим, чтобы не уходило ниже -8
        quad_vals = np.clip(quad_vals, -8, 0)
        quad_graph = axes.plot_line_graph(
            x_values=quad_n,
            y_values=quad_vals,
            line_color=m.RED,
            add_vertex_dots=False
        )
        quad_label = m.Text("Квадратичная", color=m.RED).next_to(quad_graph["line_graph"].get_end(), m.DL).scale(0.5).shift(m.UP + m.RIGHT*2)

        self.play(
            m.Create(linear_graph), m.Write(linear_label),
            m.Create(quad_graph), m.Write(quad_label),
            run_time=2
        )

        title = m.Text("Скорости сходимости (логарифм ошибки)", font_size=28, color=m.WHITE).to_edge(m.UP)
        self.add(title)
        note = m.Text("Квадратичная сходимость резко обгоняет линейную", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN)
        self.add(note)
        self.wait(2)


# ------------------------------------------------------------
# 5. Зазор выпуклости
# ------------------------------------------------------------
class ConvexityGap(Scene_):
    """
    График выпуклой функции, отрезок между точками, зазор между хордой и графиком.
    """
    def construct(self):
        axes = m.Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        # Выпуклая функция f(x) = x^2 - 2x + 2
        def f(x):
            return x**2 - 2*x + 2

        graph = axes.plot(f, x_range=[-0.5, 4.5], color=m.PURPLE)
        f_label = m.MathTex(r"f(x)", color=m.PURPLE).next_to(graph.get_end(), m.UR)
        self.play(m.Create(graph), m.Write(f_label))

        # Две точки x и y
        x0, y0 = 0.5, f(0.5)
        x1, y1 = 2.5, f(2.5)
        pointA = m.Dot(axes.coords_to_point(x0, y0), color=m.GREEN)
        pointB = m.Dot(axes.coords_to_point(x1, y1), color=m.GREEN)
        labelA = m.MathTex(r"(x, f(x))", color=m.GREEN).next_to(pointA, m.DL)
        labelB = m.MathTex(r"(y, f(y))", color=m.GREEN).next_to(pointB, m.UR)

        # Отрезок (хорда)
        chord = m.Line(
            axes.coords_to_point(x0, y0),
            axes.coords_to_point(x1, y1),
            color=m.YELLOW
        )

        # Промежуточная точка на хорде (λ = 0.3)
        lam = 0.3
        x_mid = lam * x0 + (1 - lam) * x1
        y_mid_chord = lam * y0 + (1 - lam) * y1
        y_mid_graph = f(x_mid)
        point_chord = m.Dot(axes.coords_to_point(x_mid, y_mid_chord), color=m.ORANGE)
        point_graph = m.Dot(axes.coords_to_point(x_mid, y_mid_graph), color=m.RED)
        label_chord = m.MathTex(r"\lambda f(x) + (1-\lambda) f(y)", color=m.ORANGE)
        label_chord.next_to(point_chord, m.UP)
        label_graph = m.MathTex(r"f(\lambda x + (1-\lambda) y)", color=m.RED)
        label_graph.next_to(point_graph, m.DOWN)

        # Вертикальная стрелка между ними – зазор выпуклости
        gap_arrow = m.Arrow(
            axes.coords_to_point(x_mid, y_mid_graph + 0.1),
            axes.coords_to_point(x_mid, y_mid_chord - 0.1),
            color=m.YELLOW,
            buff=0.1
        )
        gap_label = m.Text("Зазор выпуклости", font_size=24, color=m.YELLOW)
        gap_label.next_to(gap_arrow, m.RIGHT)

        self.play(
            m.Create(pointA), m.Write(labelA),
            m.Create(pointB), m.Write(labelB),
            m.Create(chord),
            run_time=1
        )
        self.play(
            m.Create(point_chord), m.Write(label_chord),
            m.Create(point_graph), m.Write(label_graph),
            m.Create(gap_arrow), m.Write(gap_label),
            run_time=1
        )

        title = m.Text("Выпуклость: хорда выше графика", font_size=28, color=m.WHITE).to_edge(m.UP)
        self.add(title)
        note = m.Text("Для выпуклой функции хорда лежит не ниже графика", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN)
        self.add(note)
        self.wait(2)


class UnitCircleWithTrig(Scene_):
    """
    Точка движется по единичной окружности с параметром t.
    Справа – графики cos(t) и sin(t) как функций времени.
    """
    def construct(self):
        # Левая часть: координатная плоскость с окружностью
        plane_left = m.NumberPlane(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": m.GRAY, "stroke_opacity": 0.3}
        )
        plane_left.shift(m.LEFT * 3.5)
        self.add(plane_left)

        circle = m.Circle(radius=1, color=m.BLUE, stroke_width=2)
        circle.move_to(plane_left.coords_to_point(0, 0))
        self.add(circle)

        # Точка на окружности
        t_tracker = m.ValueTracker(0)
        dot = m.Dot(color=m.YELLOW)
        dot.add_updater(lambda d: d.move_to(
            plane_left.coords_to_point(np.cos(t_tracker.get_value()), np.sin(t_tracker.get_value()))
        ))
        self.add(dot)

        # Оси для графиков справа
        axes_right = m.Axes(
            x_range=[0, 2*np.pi, np.pi/2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=5,
            y_length=3,
            axis_config={"color": m.GRAY}
        )
        axes_right.shift(m.RIGHT * 2)
        self.add(axes_right)

        # Графики cos(t) и sin(t)
        cos_graph = axes_right.plot(lambda t: np.cos(t), x_range=[0, 2*np.pi], color=m.GREEN)
        sin_graph = axes_right.plot(lambda t: np.sin(t), x_range=[0, 2*np.pi], color=m.RED)
        self.add(cos_graph, sin_graph)

        # Подписи осей
        x_label = m.MathTex("t").next_to(axes_right.x_axis.get_end(), m.RIGHT)
        y_label = m.MathTex("\\text{value}").next_to(axes_right.y_axis.get_end(), m.UP)
        self.add(x_label, y_label)

        # Вертикальная линия, показывающая текущее значение t
        t_line = m.Line(
            axes_right.coords_to_point(0, -1.5),
            axes_right.coords_to_point(0, 1.5),
            color=m.YELLOW,
            stroke_width=2
        )
        t_line.add_updater(lambda l: l.move_to(axes_right.coords_to_point(t_tracker.get_value(), 0), aligned_edge=m.LEFT))
        self.add(t_line)

        # Точки на графиках в текущий момент
        cos_dot = m.Dot(color=m.GREEN)
        cos_dot.add_updater(lambda d: d.move_to(axes_right.coords_to_point(t_tracker.get_value(), np.cos(t_tracker.get_value()))))
        sin_dot = m.Dot(color=m.RED)
        sin_dot.add_updater(lambda d: d.move_to(axes_right.coords_to_point(t_tracker.get_value(), np.sin(t_tracker.get_value()))))
        self.add(cos_dot, sin_dot)

        # Анимация
        self.play(t_tracker.animate.set_value(2*np.pi), run_time=6, rate_func=m.rate_functions.linear)

        # Финальные подписи
        self.add(m.Text("cos t", color=m.GREEN).next_to(cos_graph.get_end(), m.RIGHT))
        self.add(m.Text("sin t", color=m.RED).next_to(sin_graph.get_end(), m.RIGHT))
        self.wait(1)


# ------------------------------------------------------------
# 2. Нелинейное преобразование сетки
# ------------------------------------------------------------
class NonlinearGridTransform(LinearTransformationScene_):
    """
    Регулярная сетка деформируется нелинейным отображением:
    f(x,y) = (x + 0.5 sin y, y + 0.3 x^2/(1+x^2)).
    Линии изгибаются, точки показывают перемещение.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        # # Создаём сетку с большим количеством линий для красоты
        # plane = m.NumberPlane(
        #     x_range=[-4, 4, 0.5],
        #     y_range=[-3, 3, 0.5],
        #     # background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.4}
        # )
        # self.add(plane)
        # self.add_transformable_mobject(plane)

        # Добавим несколько точек, чтобы видеть их перемещение
        points_coords = [
            (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
            (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5)
        ]
        dots = m.VGroup(*[m.Dot((x, y, 0), color=m.RED, radius=0.08) for x, y in points_coords])
        self.add(dots)
        self.add_transformable_mobject(dots)

        # Определяем нелинейное отображение
        def nonlinear_func(vec):
            x, y, z = vec[0], vec[1], vec[2]
            new_x = x + 0.5 * np.sin(y)
            new_y = y + 0.3 * (x**2) / (1 + x**2)
            return np.array([new_x, new_y, z])

        # Применяем
        self.moving_mobjects = []
        self.apply_nonlinear_transformation(nonlinear_func, run_time=4)

        # Заголовок и пояснение
        title = m.Text("Нелинейное преобразование сетки", font_size=28, color=m.WHITE)
        title.to_edge(m.UP)
        self.add_foreground_mobject(title)

        formula = m.MathTex(
            r"f(x, y) = (x + 0.5\sin y,\; y + \frac{0.3x^2}{1+x^2})",
            font_size=32
        ).to_edge(m.DOWN)
        self.add_foreground_mobject(formula)

        self.wait(2)

if __name__ == '__main__':
    import os
    from pathlib import Path
 
    SCENES = [
        # "FunctionMapping",
        # "ExpLogParabola",
        # "SigmoidReLU",
        # "SqueezeTheorem",
        # "ConvergenceRates",
        # "ConvexityGap",
        # "UnitCircleWithTrig",
        "NonlinearGridTransform",

    ]
    file_path = Path(__file__).resolve()

    for SCENE in SCENES:
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -qh")
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -sqh")