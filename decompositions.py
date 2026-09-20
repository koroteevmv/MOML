import manim as m
import numpy as np
from math import sin, ceil
from theming import LinearTransformationScene_, ThreeDScene_, Scene_
from scipy.special import erf

class SpectralDecompositionScene(m.Scene):
    """20 Спектральное разложение: 3 шага с сохранением исходной сетки"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = m.WHITE

    def construct(self):
        # 1. СТАТИЧНАЯ СЕТКА (РЕФЕРЕНС)
        # Она останется на месте, чтобы было видно исходное положение
        plane_static = m.NumberPlane(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1],
            x_length=9, y_length=9,
            background_line_style={"stroke_color": m.GRAY, "stroke_opacity": 0.3},
            axis_config={"stroke_color": m.BLACK, "stroke_width": 2}
        )
        self.add(plane_static)

        # 2. ТРАНСФОРМИРУЕМАЯ СЕТКА (ЦВЕТНАЯ)
        # Она будет изменяться, показывая деформацию
        plane_dynamic = m.NumberPlane(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1],
            x_length=9, y_length=9,
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5},
            axis_config={"stroke_color": m.BLUE, "stroke_width": 3}
        )
        self.add(plane_dynamic)

        # 3. Матрица A и спектральное разложение
        A = np.array([[4, 2], [2, 3]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(A)
        
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        V = eigvecs[:, idx]
        Lambda = np.diag(eigvals)
        VT = V.T

        # 4. Объект (круг), уменьшен для гарантии попадания в экран
        # max_eigenval ≈ 5.37. Радиус 0.7 -> полусось 3.75 (влезает в [-5, 5])
        shape = m.Circle(radius=0.7, color=m.RED, fill_opacity=0.3, stroke_width=4)
        shape.move_to(m.ORIGIN)
        self.add(shape)

        # ШАГ 1: Поворот (V^T) 
        mat_VT = m.Matrix(np.round(VT, 2), element_to_mobject_config={"font_size": 22, "color": m.BLACK})
        mat_VT.scale(0.8).get_brackets().set_color(m.BLACK)
        mat_VT.move_to([-4.2, 1.2, 0])

        text_VT = m.Text(
            "1. Поворот V^T:\nОси совпадают с\nсобственными направлениями",
            font_size=18, color=m.BLACK
        ).to_edge(m.RIGHT, buff=1.5).shift(m.UP * 1.5)

        self.play(m.Write(mat_VT), m.Write(text_VT))
        # Трансформируем ТОЛЬКО динамическую сетку и фигуру
        self.play(
            plane_dynamic.animate.apply_matrix(VT), 
            shape.animate.apply_matrix(VT), 
            run_time=2
        )
        self.wait(0.5)

        # ШАГ 2: Масштаб (Λ)
        mat_Lam = m.Matrix(np.round(Lambda, 2), element_to_mobject_config={"font_size": 22, "color": m.BLACK})
        mat_Lam.scale(0.8).get_brackets().set_color(m.BLACK)
        mat_Lam.move_to(mat_VT)

        text_Lam = m.Text(
            "2. Масштаб Λ:\nРастяжение вдоль осей\nна собственные числа",
            font_size=18, color=m.BLACK
        ).move_to(text_VT)

        self.play(
            m.ReplacementTransform(mat_VT, mat_Lam),
            m.ReplacementTransform(text_VT, text_Lam)
        )
        self.play(
            plane_dynamic.animate.apply_matrix(Lambda), 
            shape.animate.apply_matrix(Lambda), 
            run_time=2
        )
        self.wait(0.5)

        # 3: Обратный поворот (V) 
        mat_V = m.Matrix(np.round(V, 2), element_to_mobject_config={"font_size": 22, "color": m.BLACK})
        mat_V.scale(0.8).get_brackets().set_color(m.BLACK)
        mat_V.move_to(mat_Lam)

        text_V = m.Text(
            "3. Поворот V:\nВозврат в исходную\nсистему координат",
            font_size=18, color=m.BLACK
        ).move_to(text_Lam)

        self.play(
            m.ReplacementTransform(mat_Lam, mat_V),
            m.ReplacementTransform(text_Lam, text_V)
        )
        self.play(
            plane_dynamic.animate.apply_matrix(V), 
            shape.animate.apply_matrix(V), 
            run_time=2
        )
        self.wait(1)

        self.play(m.FadeOut(mat_V), m.FadeOut(text_V), run_time=1)

        formula = m.MathTex("A = V \\Lambda V^T", font_size=40, color=m.BLACK)
        formula.to_edge(m.UP, buff=1.0)

        '''explanation = m.Text(
            "Любое действие матрицы = Поворот + Масштаб + Поворот",
            font_size=20, color=m.GRAY
        ).next_to(formula, m.DOWN)

        self.play(m.Write(formula), m.Write(explanation), run_time=2)
        self.wait(3)
        
        self.play(m.FadeOut(*self.mobjects))'''

class SVD(LinearTransformationScene_):
    def __init__(self, **kwargs):
        LinearTransformationScene_.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        matrix = np.array([[0.8, 0], [0.5, 3]]).T

        s = 0.7

        self.play(m.Create(A := m.Matrix(matrix).scale(s).to_edge(m.DOWN, buff=1).to_edge(m.LEFT)), run_time=0.01)

        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()

        i = m.Vector([matrix[0][0], matrix[1][0]], color=m.GREEN)
        j = m.Vector([matrix[0][1], matrix[1][1]], color=m.RED)
        self.play(m.Create(i), run_time=0.01)
        self.play(m.Create(j), run_time=0.01)

        self.moving_mobjects = []
        self.apply_inverse(matrix)
        self.wait()

        U, S, Vh = np.linalg.svd(matrix, full_matrices=True)
        print(Vh)
        print(np.diag(S))
        print(U)

        self.play(m.Create(t1 := m.Text("=").scale(s).next_to(A, m.RIGHT)), run_time=0.01)
        self.play(m.Create(U_label := m.Matrix(np.round(U,2), stroke_width=7).scale(s).next_to(t1, m.RIGHT)), run_time=0.01)
        self.play(m.Create(t2 := m.Tex(r"$\times$").scale(s).next_to(U_label, m.RIGHT)), run_time=0.01)
        self.play(m.Create(S_label := m.Matrix(np.round(np.diag(S),2), stroke_width=7).scale(s).next_to(t2, m.RIGHT)), run_time=0.01)
        self.play(m.Create(t3 := m.Tex(r"$\times$").scale(s).next_to(S_label, m.RIGHT)), run_time=0.01)
        self.play(m.Create(V_label := m.Matrix(np.round(Vh,2), stroke_width=7).scale(s).next_to(t3, m.RIGHT)), run_time=0.01)
        self.wait()

        self.play(V_label.animate.set_color(m.YELLOW), run_time=0.3)
        self.play(m.Create(t := m.Text("Поворот", color=m.YELLOW).scale(s).next_to(V_label, m.UP)), run_time=0.01)
        self.moving_mobjects = []
        self.apply_matrix(Vh)
        self.play(m.Uncreate(t))
        self.play(V_label.animate.set_color(m.WHITE), run_time=0.3)
        self.wait()

        self.play(S_label.animate.set_color(m.YELLOW), run_time=0.3)
        self.play(m.Create(t := m.Text("Масштабирование", color=m.YELLOW).scale(s).next_to(S_label, m.UP)), run_time=0.01)
        self.moving_mobjects = []
        self.apply_matrix(np.diag(S))
        self.play(m.Uncreate(t))
        self.play(S_label.animate.set_color(m.WHITE), run_time=0.3)
        self.wait()

        self.play(U_label.animate.set_color(m.YELLOW), run_time=0.3)
        self.play(m.Create(t := m.Text("Поворот", color=m.YELLOW).scale(s).next_to(U_label, m.UP)), run_time=0.01)
        self.moving_mobjects = []
        self.apply_matrix(U)
        self.play(m.Uncreate(t))
        self.play(U_label.animate.set_color(m.WHITE), run_time=0.3)
        self.wait()


# ------------------------------------------------------------
# 1. Поворот в 3D относительно оси Z
# ------------------------------------------------------------
class Rotation3DZ(ThreeDScene_):
    """
    Анимация поворота трёхмерного пространства вокруг оси Z.
    Показывается куб и матрица поворота.
    """
    def construct(self):
        # Настройка осей и камеры
        axes = m.ThreeDAxes()
        axes.set_color(m.GRAY)
        self.add(axes)
        self.set_camera_orientation(phi=70 * m.DEGREES, theta=-45 * m.DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15)

        # Матрица поворота на 45° вокруг Z
        angle = np.deg2rad(45)
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])

        # Отображение матрицы на экране
        mat = m.Matrix(np.round(Rz, 2))
        mat.scale(0.6)
        mat.to_corner(m.UP + m.LEFT)
        self.add_fixed_in_frame_mobjects(mat)
        self.play(m.Write(mat), run_time=1)

        # Куб
        cube = m.Cube(side_length=2, fill_color=m.BLUE, fill_opacity=0.2, stroke_color=m.BLUE)
        cube.shift(m.RIGHT * 0.5 + m.UP * 0.5 + m.OUT * 0.5)  # чтобы центр в (0,0,0) был внутри
        self.play(m.Create(cube), run_time=1)

        # Векторы базиса
        i_vec = m.Vector([1, 0, 0], color=m.GREEN)
        j_vec = m.Vector([0, 1, 0], color=m.RED)
        k_vec = m.Vector([0, 0, 1], color=m.GOLD)
        self.play(m.GrowArrow(i_vec), m.GrowArrow(j_vec), m.GrowArrow(k_vec), run_time=1)

        # Применяем поворот
        self.play(
            m.ApplyMatrix(Rz, cube),
            m.Transform(i_vec, m.Vector(Rz @ [1, 0, 0], color=m.GREEN)),
            m.Transform(j_vec, m.Vector(Rz @ [0, 1, 0], color=m.RED)),
            m.Transform(k_vec, m.Vector(Rz @ [0, 0, 1], color=m.GOLD)),
            run_time=3
        )

        # Подпись
        label = m.Text("Поворот вокруг оси Z", font_size=24, color=m.WHITE)
        label.to_edge(m.DOWN)
        self.add_fixed_in_frame_mobjects(label)
        self.play(m.Write(label), run_time=0.5)
        self.wait(2)


# ------------------------------------------------------------
# 2. Разложение матрицы как композиция двух преобразований
# ------------------------------------------------------------
class MatrixDecomposition(LinearTransformationScene_):
    """
    Демонстрация разложения A = R * S, где R – поворот, S – масштабирование.
    Показывается последовательное применение S, затем R.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=True,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        # Матрица A = R * S
        angle = np.deg2rad(30)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        S = np.diag([2.0, 0.5])
        A = R @ S

        # Показываем матрицу A
        mat_A = m.Matrix(np.round(A, 2))
        mat_A.scale(0.8)
        mat_A.to_corner(m.UP + m.LEFT)
        self.play(m.Write(mat_A), run_time=1)
        self.add_transformable_mobject(mat_A)  # чтобы не исчезла

        # Исходный вектор (для наглядности)
        v = np.array([1.5, 1.0])
        vec = self.add_vector(v, color=m.PURPLE)
        label_v = m.MathTex(r"\vec{v}", color=m.PURPLE).next_to(vec.get_end(), m.RIGHT)
        self.play(m.Write(label_v), run_time=0.5)

        # Разложение: S затем R
        label_decomp = m.MathTex(r"A = R \cdot S", font_size=30).to_edge(m.UP)
        self.play(m.Write(label_decomp), run_time=1)

        # Показываем матрицы S и R
        mat_S = m.Matrix(np.round(S, 2)).scale(0.6).next_to(mat_A, m.DOWN, buff=0.5)
        mat_R = m.Matrix(np.round(R, 2)).scale(0.6).next_to(mat_S, m.DOWN, buff=0.5)
        self.play(m.Write(mat_S), m.Write(mat_R), run_time=1)

        # Шаг 1: масштабирование S
        self.moving_mobjects = []
        self.apply_matrix(S)
        self.wait(0.5)

        # Шаг 2: поворот R
        self.moving_mobjects = []
        self.apply_matrix(R)
        self.wait(0.5)

        # Финальная подпись
        final_label = m.Text("Композиция преобразований", font_size=24, color=m.YELLOW)
        final_label.to_edge(m.DOWN)
        self.play(m.Write(final_label), run_time=0.5)
        self.wait(2)


# ------------------------------------------------------------
# 3. Ладонь: положительно определённая vs неположительно определённая
# ------------------------------------------------------------
class HandPositiveDefinite(Scene_):
    """
    Сравнение действия положительно определённой матрицы и матрицы с отрицательным собственным числом.
    Используется упрощённая фигура "ладонь".
    """
    def construct(self):
        # Создаём фигуру ладони из эллипсов (примитивно)
        def create_hand():
            # Ладонь
            palm = m.Ellipse(width=1.2, height=1.5, color=m.TEAL, fill_opacity=0.5)
            palm.shift(m.DOWN * 0.3)
            # Пальцы
            fingers = m.VGroup()
            positions = [(-0.6, 0.9), (-0.3, 1.1), (0, 1.2), (0.3, 1.1), (0.6, 0.9)]
            for i, (x, y) in enumerate(positions):
                finger = m.Ellipse(width=0.25, height=0.6, color=m.TEAL, fill_opacity=0.5)
                finger.shift(m.RIGHT * x + m.UP * y)
                fingers.add(finger)
            hand = m.VGroup(palm, fingers)
            return hand

        # Две копии ладони
        hand1 = create_hand().shift(m.LEFT * 3)
        hand2 = create_hand().shift(m.RIGHT * 3)

        # Матрицы
        A_pos = np.array([[2.0, 0.5], [0.5, 1.5]])   # положительно определённая
        A_neg = np.array([[2.0, 0.0], [0.0, -1.0]])  # одно отрицательное собственное число

        # Собственные числа
        eig_pos = np.linalg.eigvalsh(A_pos)
        eig_neg = np.linalg.eigvalsh(A_neg)

        # Отображение матриц и собственных чисел
        mat_pos = m.Matrix(np.round(A_pos, 2)).scale(0.6).next_to(hand1, m.UP, buff=0.5)
        mat_neg = m.Matrix(np.round(A_neg, 2)).scale(0.6).next_to(hand2, m.UP, buff=0.5)

        self.play(
            m.Create(hand1), m.Create(hand2),
            m.Write(mat_pos), m.Write(mat_neg),
            run_time=2
        )

        # Подписи собственных чисел
        label_pos = m.MathTex(r"\lambda_1 = {:.2f}, \lambda_2 = {:.2f}".format(eig_pos[0], eig_pos[1]), color=m.GREEN)
        label_pos.next_to(mat_pos, m.DOWN, buff=0.2)
        label_neg = m.MathTex(r"\lambda_1 = {:.2f}, \lambda_2 = {:.2f}".format(eig_neg[0], eig_neg[1]), color=m.RED)
        label_neg.next_to(mat_neg, m.DOWN, buff=0.2)

        self.play(m.Write(label_pos), m.Write(label_neg), run_time=1)

        # Применяем преобразования
        self.play(
            m.ApplyMatrix(A_pos, hand1, about_point=hand1.get_center()),
            m.ApplyMatrix(A_neg, hand2, about_point=hand2.get_center()),
            run_time=3
        )

        # Пояснительный текст
        text_pos = m.Text("Положительно определённая", font_size=20, color=m.GREEN).next_to(hand1, m.DOWN, buff=0.5)
        text_neg = m.Text("Не положительно определённая", font_size=20, color=m.RED).next_to(hand2, m.DOWN, buff=0.5)
        self.play(m.Write(text_pos), m.Write(text_neg), run_time=1)

        self.wait(2)


# ------------------------------------------------------------
# 4. Разложение Холецкого и оси эллипса
# ------------------------------------------------------------
class CholeskyEllipse(LinearTransformationScene_):
    """
    Визуализация эллипса, заданного положительно определённой матрицей A,
    и его связь с разложением Холецкого A = L L^T.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        # Матрица A (положительно определённая)
        A = np.array([[3.0, 0.5], [0.5, 2.0]])
        L = np.linalg.cholesky(A)  # нижнетреугольная

        # Рисуем единичный круг
        circle = m.Circle(radius=1, color=m.BLUE, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # Преобразование L переводит круг в эллипс
        self.apply_matrix(L)
        self.wait(0.5)

        # Векторы-столбцы L (полуоси эллипса)
        col1 = L @ np.array([1, 0])
        col2 = L @ np.array([0, 1])
        vec1 = self.add_vector(col1, color=m.GREEN)
        vec2 = self.add_vector(col2, color=m.RED)
        label1 = m.MathTex(r"L_{*1}", color=m.GREEN).next_to(vec1.get_end(), m.RIGHT)
        label2 = m.MathTex(r"L_{*2}", color=m.RED).next_to(vec2.get_end(), m.RIGHT)
        self.play(m.Write(label1), m.Write(label2), run_time=0.5)

        # Отображение матриц
        mat_A = m.Matrix(np.round(A, 2)).scale(0.6).to_corner(m.UP + m.LEFT)
        mat_L = m.Matrix(np.round(L, 2)).scale(0.6).next_to(mat_A, m.RIGHT, buff=0.8)
        eq = m.MathTex(r"A = L L^T").scale(0.6).next_to(mat_L, m.RIGHT, buff=0.5)
        self.play(
            m.Write(mat_A), m.Write(mat_L), m.Write(eq),
            run_time=2
        )

        # Подпись
        caption = m.Text("Разложение Холецкого: оси эллипса задаются столбцами L", font_size=24, color=m.YELLOW)
        caption.to_edge(m.DOWN)
        self.play(m.Write(caption), run_time=1)
        self.wait(2)


# ------------------------------------------------------------
# 5. Ортогональная матрица vs произвольная матрица
# ------------------------------------------------------------
class OrthogonalVsArbitrary(LinearTransformationScene_):
    """
    Сравнение действия ортогональной матрицы (поворот) и произвольной матрицы.
    Единичный круг остаётся кругом в первом случае и становится эллипсом во втором.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        # Два круга: один слева (ортогональная), другой справа (произвольная)
        circle1 = m.Circle(radius=1.5, color=m.GREEN, fill_opacity=0.2).shift(m.LEFT * 3)
        circle2 = m.Circle(radius=1.5, color=m.RED, fill_opacity=0.2).shift(m.RIGHT * 3)
        self.add(circle1, circle2)
        self.add_transformable_mobject(circle1, circle2)

        # Матрицы
        angle = np.deg2rad(40)
        Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        A = np.array([[2.0, 1.0], [0.5, 1.5]])

        # Отображение матриц
        mat_Q = m.Matrix(np.round(Q, 2)).scale(0.6).next_to(circle1, m.UP, buff=0.5)
        mat_A = m.Matrix(np.round(A, 2)).scale(0.6).next_to(circle2, m.UP, buff=0.5)
        self.play(m.Write(mat_Q), m.Write(mat_A), run_time=1)

        # Применяем преобразования
        self.play(
            m.ApplyMatrix(Q, circle1, about_point=circle1.get_center()),
            m.ApplyMatrix(A, circle2, about_point=circle2.get_center()),
            run_time=3
        )

        # Подписи
        label_Q = m.Text("Ортогональная Q: форма сохраняется", font_size=20, color=m.GREEN)
        label_Q.next_to(circle1, m.DOWN, buff=0.5)
        label_A = m.Text("Произвольная A: круг -> эллипс", font_size=20, color=m.RED)
        label_A.next_to(circle2, m.DOWN, buff=0.5)
        self.play(m.Write(label_Q), m.Write(label_A), run_time=1)

        self.wait(2)

class EigenEllipse(LinearTransformationScene_):
    """
    Слева – единичная окружность с набором векторов.
    Справа – образ под действием матрицы A (эллипс).
    Собственные векторы (соответствующие λ₁, λ₂) выделены цветом и подписаны.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Левая часть (статичная)
        circle_left = m.Circle(radius=1, color=m.BLUE, fill_opacity=0.1).shift(m.LEFT * 3.5)
        self.add(circle_left)
        for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
            v = np.array([np.cos(a), np.sin(a), 0])
            vec = m.Vector(v, color=m.GRAY, opacity=0.5).shift(m.LEFT * 3.5)
            self.add(vec)

        # Правая часть (трансформируемая)
        circle_right = m.Circle(radius=1, color=m.RED, fill_opacity=0.1).shift(m.RIGHT * 3.5)
        self.add(circle_right)
        self.add_transformable_mobject(circle_right)

        # Применяем матрицу
        self.moving_mobjects = []
        self.apply_matrix(A, added_anims=[m.Transform(circle_right, circle_right.copy().apply_matrix(A))])

        # Собственные векторы (после трансформации)
        origin = self.plane.c2p(0, 0)
        for i in range(2):
            v = eigvecs[:, i]
            v_after = A @ v
            arrow = m.Arrow(origin, self.plane.c2p(v_after[0], v_after[1]),
                            color=m.YELLOW if i == 0 else m.GREEN, buff=0)
            self.add(arrow)
            label = m.MathTex(r"\lambda_{} = {:.2f}".format(i+1, eigvals[i]),
                              color=m.YELLOW if i == 0 else m.GREEN)
            label.next_to(arrow.get_end(), m.UR if i == 0 else m.DR)
            self.add(label)

        # Подписи
        self.add(m.Text("Единичная окружность", font_size=20).next_to(circle_left, m.DOWN, buff=0.3))
        self.add(m.Text("A · (окружность)", font_size=20).next_to(circle_right, m.DOWN, buff=0.3))
        mat = m.Matrix(np.round(A, 2)).scale(0.7).to_corner(m.UP + m.LEFT)
        self.add(mat)
        self.wait(2)


# ============================================================
# 2. SVD единичной окружности: V^T → Λ → V
# ============================================================
class SVDCircleSteps(LinearTransformationScene_):
    """
    Три шага:
    1) V^T – поворот (форма не меняется),
    2) Λ – масштабирование (круг → эллипс),
    3) V – обратный поворот.
    Результат совпадает с прямым умножением на A.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 1.0], [0.5, 1.5]])
        U, S, Vt = np.linalg.svd(A)
        V = Vt.T
        Lambda = np.diag(S)

        circle = m.Circle(radius=1, color=m.BLUE, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        mat_Vt = m.Matrix(np.round(Vt, 2)).scale(0.6).to_corner(m.UP + m.LEFT)
        mat_Lambda = m.Matrix(np.round(Lambda, 2)).scale(0.6).next_to(mat_Vt, m.DOWN, buff=0.5)
        mat_V = m.Matrix(np.round(V, 2)).scale(0.6).next_to(mat_Lambda, m.DOWN, buff=0.5)
        self.add(mat_Vt, mat_Lambda, mat_V)

        # Шаг 1
        self.moving_mobjects = []
        self.apply_matrix(Vt, added_anims=[m.Transform(circle, circle.copy().apply_matrix(Vt))])
        label = m.Text("1. Поворот V^T", font_size=20, color=m.YELLOW).to_edge(m.UP)
        self.add(label)
        self.wait(1)
        self.remove(label)

        # Шаг 2
        self.moving_mobjects = []
        self.apply_matrix(Lambda, added_anims=[m.Transform(circle, circle.copy().apply_matrix(Lambda))])
        label = m.Text("2. Масштабирование Λ", font_size=20, color=m.YELLOW).to_edge(m.UP)
        self.add(label)
        self.wait(1)
        self.remove(label)

        # Шаг 3
        self.moving_mobjects = []
        self.apply_matrix(V, added_anims=[m.Transform(circle, circle.copy().apply_matrix(V))])
        label = m.Text("3. Поворот V", font_size=20, color=m.YELLOW).to_edge(m.UP)
        self.add(label)
        self.wait(1)
        self.remove(label)

        final_label = m.Text("A = V Λ V^T", font_size=24, color=m.GREEN).to_edge(m.DOWN)
        self.add(final_label)
        self.wait(2)


# ============================================================
# 3. Спектральная кластеризация: два кольца
# ============================================================
class SpectralClustering(Scene_):
    """
    Слева – исходные точки (два концентрических кольца).
    Справа – после преобразования в пространство собственных векторов лапласиана
    точки становятся линейно разделимыми.
    """
    def construct(self):
        np.random.seed(42)
        n = 100
        r1, r2 = 1.5, 3.0
        theta1 = np.random.uniform(0, 2*np.pi, n)
        theta2 = np.random.uniform(0, 2*np.pi, n)
        noise = 0.1
        x1 = (r1 + np.random.normal(0, noise, n)) * np.cos(theta1)
        y1 = (r1 + np.random.normal(0, noise, n)) * np.sin(theta1)
        x2 = (r2 + np.random.normal(0, noise, n)) * np.cos(theta2)
        y2 = (r2 + np.random.normal(0, noise, n)) * np.sin(theta2)

        points = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        labels = np.array([0]*n + [1]*n)

        # Левая часть
        left_plane = m.NumberPlane(x_range=[-4, 4], y_range=[-4, 4], opacity=0.3).shift(m.LEFT * 3.5)
        self.add(left_plane)
        dots_left = m.VGroup(*[
            m.Dot(left_plane.c2p(p[0], p[1]), radius=0.05, color=m.BLUE if lab==0 else m.RED)
            for p, lab in zip(points, labels)
        ])
        self.add(dots_left)

        # Правая часть – после преобразования (эмуляция спектрального вложения)
        right_plane = m.NumberPlane(x_range=[-4, 4], y_range=[-4, 4], opacity=0.3).shift(m.RIGHT * 3.5)
        self.add(right_plane)
        r_vals = np.sqrt(points[:,0]**2 + points[:,1]**2)
        theta_vals = np.arctan2(points[:,1], points[:,0])
        x_new = r_vals * np.cos(2*theta_vals)
        y_new = r_vals * np.sin(2*theta_vals)
        dots_right = m.VGroup(*[
            m.Dot(right_plane.c2p(x_new[i], y_new[i]), radius=0.05, color=m.BLUE if lab==0 else m.RED)
            for i, lab in enumerate(labels)
        ])
        self.add(dots_right)
        # Разделяющая прямая
        line = m.Line(right_plane.c2p(0, -4), right_plane.c2p(0, 4), color=m.YELLOW, stroke_width=2)
        self.add(line)

        # Подписи и стрелка
        self.add(m.Text("Исходные данные", font_size=20).next_to(left_plane, m.DOWN, buff=0.3))
        self.add(m.Text("После спектрального вложения", font_size=20).next_to(right_plane, m.DOWN, buff=0.3))
        arrow = m.Arrow(left_plane.get_right(), right_plane.get_left(), color=m.WHITE)
        self.add(arrow)
        self.wait(2)


# ============================================================
# 4. Низкоранговая аппроксимация изображения (SVD)
# ============================================================
class LowRankApprox(Scene_):
    """
    Восстановление синтетического изображения с рангами 1, 5, 20, 50.
    Рядом – количество хранимых чисел.
    """
    def construct(self):
        size = 100
        img = np.zeros((size, size))
        cx, cy = size//2, size//2
        r = 30
        for i in range(size):
            for j in range(size):
                if (i-cx)**2 + (j-cy)**2 < r**2:
                    img[i, j] = 1.0
        for i in range(20, 80):
            for j in range(20, 80):
                if 20 < i < 80 and 20 < j < 80:
                    img[i, j] = 0.5
        img += np.random.normal(0, 0.05, (size, size))
        img = np.clip(img, 0, 1)

        U, S, Vt = np.linalg.svd(img, full_matrices=False)

        def create_image(arr):
            arr = np.uint8(255 * np.rot90(arr))
            return m.ImageMobject(arr)

        orig = create_image(img).scale(2).to_edge(m.UP)
        self.add(orig)
        self.add(m.Text("Оригинал", font_size=20).next_to(orig, m.DOWN))

        ranks = [1, 5, 20, 50]
        images = []
        for r in ranks:
            approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
            images.append(create_image(approx).scale(1.5))

        group = m.Group(*images).arrange(m.RIGHT, buff=0.5)
        group.next_to(orig, m.DOWN, buff=0.5)
        self.add(group)

        for i, r in enumerate(ranks):
            label = m.Text(f"Ранг {r}", font_size=18).next_to(images[i], m.DOWN)
            self.add(label)
            n_params = r * (size + size - r)
            self.add(m.Text(f"≈{n_params} чисел", font_size=14).next_to(label, m.DOWN))
        self.wait(3)


# ============================================================
# 5. PCA в 3D: облако точек и главные компоненты
# ============================================================
class PCAPlot3D(ThreeDScene_):
    """
    3D-облако с тремя осями главных компонент.
    Справа – проекция на первые две компоненты.
    """
    def construct(self):
        np.random.seed(42)
        cov = np.array([[5, 1, 0.5], [1, 2, 0.2], [0.5, 0.2, 0.5]])
        points = np.random.multivariate_normal([0,0,0], cov, 100)

        axes = m.ThreeDAxes(x_range=[-4,4], y_range=[-4,4], z_range=[-4,4])
        axes.set_color(m.GRAY)
        self.add(axes)
        self.set_camera_orientation(phi=70*m.DEGREES, theta=-45*m.DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

        dots = m.VGroup(*[m.Dot(axes.c2p(p[0], p[1], p[2]), radius=0.05, color=m.BLUE) for p in points])
        self.add(dots)

        centered = points - points.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        colors = [m.YELLOW, m.ORANGE, m.PURPLE]
        for i in range(3):
            length = np.sqrt(eigvals[i]) * 2.0
            vec = eigvecs[:, i] * length
            arrow = m.Arrow3D(axes.c2p(0,0,0), axes.c2p(vec[0], vec[1], vec[2]),
                              color=colors[i], thickness=0.02)
            self.add(arrow)
            label = m.Text(f"PC{i+1}", font_size=16, color=colors[i])
            label.next_to(arrow.get_end(), m.UR if i==0 else m.UL if i==1 else m.DR)
            self.add_fixed_in_frame_mobjects(label)

        # Проекция на PC1-PC2
        proj_axes = m.Axes(x_range=[-6,6], y_range=[-6,6], x_length=3, y_length=3)
        proj_axes.shift(m.RIGHT*4 + m.DOWN*2)
        self.add_fixed_in_frame_mobjects(proj_axes)
        proj_points = centered @ eigvecs[:, :2]
        proj_dots = m.VGroup(*[m.Dot(proj_axes.c2p(p[0], p[1]), radius=0.04, color=m.BLUE) for p in proj_points])
        self.add_fixed_in_frame_mobjects(proj_dots)
        self.add_fixed_in_frame_mobjects(m.Text("Проекция на PC1-PC2", font_size=16).next_to(proj_axes, m.DOWN))
        self.wait(3)


# ============================================================
# 6. Генеалогическое древо матричных разложений
# ============================================================
class MatrixDecompTree(Scene_):
    """
    Схема: SVD – корень, ответвления к собственному разложению, QR, Холецкому.
    На стрелках – условия.
    """
    def construct(self):
        root = m.Text("SVD", font_size=36, color=m.YELLOW).shift(m.UP * 0.5)
        self.add(root)

        branches = [
            (m.Text("Собственное\nразложение", font_size=24, color=m.GREEN), m.LEFT*3 + m.DOWN*2, "Квадратная"),
            (m.Text("QR-разложение", font_size=24, color=m.BLUE), m.RIGHT*2.5 + m.DOWN*2, "Произвольная"),
            (m.Text("Холецкого", font_size=24, color=m.PURPLE), m.RIGHT*3.5 + m.DOWN*3, "Симм. + полож. опр.")
        ]
        for branch, pos, cond in branches:
            branch.shift(pos)
            self.add(branch)
            arrow = m.Arrow(root.get_bottom() + m.DOWN*0.2, branch.get_top() + m.UP*0.2, color=m.WHITE, stroke_width=2)
            self.add(arrow)
            self.add(m.Text(cond, font_size=14, color=m.GRAY).move_to(arrow.get_center() + m.UP*0.2))

        self.add(m.Text("Условия для перехода", font_size=16, color=m.GRAY).next_to(root, m.DOWN, buff=1.5))
        self.wait(2)

# ============================================================
# 1. Действие матрицы A на единичную окружность (собственные векторы)
# ============================================================
class EigenEllipse(LinearTransformationScene_):
    """
    Слева — единичная окружность с набором векторов.
    Справа — образ под действием A (эллипс).
    Собственные векторы остаются на своих прямых и подписаны λ₁, λ₂.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # ---- Левая часть: эталонная окружность с векторами ----
        left_plane = m.NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            background_line_style={"stroke_color": m.GRAY_E, "stroke_opacity": 0.4}
        ).shift(m.LEFT * 3.5)
        self.add(left_plane)

        circle_ref = m.Circle(radius=1, color=m.BLUE, stroke_width=2).move_to(left_plane.c2p(0, 0))
        self.add(circle_ref)

        for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
            v = np.array([np.cos(a), np.sin(a), 0])
            self.add(m.Vector(v, color=m.GRAY_E, opacity=0.7).shift(left_plane.c2p(0, 0)))

        # ---- Правая часть: трансформируемая сцена ----
        right_plane = self.plane
        right_plane.shift(m.RIGHT * 0.5)

        circle_dyn = m.Circle(radius=1, color=m.RED, stroke_width=2, fill_opacity=0.1)
        self.add(circle_dyn)
        self.add_transformable_mobject(circle_dyn)

        # Векторы, которые тоже трансформируются
        vecs = m.VGroup()
        for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
            v = np.array([np.cos(a), np.sin(a), 0])
            vecs.add(m.Vector(v, color=m.GRAY_E, opacity=0.7))
        self.add(vecs)
        self.add_transformable_mobject(vecs)

        # Применяем матрицу A
        self.moving_mobjects = []
        self.apply_matrix(A)
        self.wait(0.5)

        # Собственные векторы (появляются после преобразования)
        origin = right_plane.c2p(0, 0)
        colors = [m.YELLOW, m.GREEN]
        for i in range(2):
            v_after = A @ eigvecs[:, i]
            arrow = m.Arrow(origin, right_plane.c2p(v_after[0], v_after[1]),
                            color=colors[i], buff=0, stroke_width=4)
            self.add(arrow)
            # Линия собственного направления (продлеваем)
            ext = 2.5
            line = m.DashedLine(
                right_plane.c2p(-ext * eigvecs[0, i], -ext * eigvecs[1, i]),
                right_plane.c2p(ext * eigvecs[0, i], ext * eigvecs[1, i]),
                color=colors[i], stroke_width=2, stroke_opacity=0.6
            )
            self.add(line)
            label = m.MathTex(r"\lambda_{} = {:.2f}".format(i+1, eigvals[i]),
                              color=colors[i], font_size=28)
            label.next_to(arrow.get_end(), m.UR if i == 0 else m.DR)
            self.add(label)

        # ---- Заголовки ----
        mat = m.Matrix(np.round(A, 2)).scale(0.7).to_corner(m.UP + m.LEFT)
        self.add(mat)

        title = m.Text("A · (единичная окружность) = эллипс", font_size=24, color=m.WHITE)
        title.to_edge(m.UP, buff=0.4)
        self.add(title)

        note = m.Text("Собственные векторы остаются на своих прямых", font_size=20, color=m.YELLOW)
        note.to_edge(m.DOWN, buff=0.6)
        self.add(note)
        self.wait(2)


# ============================================================
# 2. SVD единичной окружности: V^T → Λ → V
# ============================================================
class SVDCircleSteps(LinearTransformationScene_):
    """
    1) V^T — поворот, форма не меняется.
    2) Λ — растяжение вдоль осей (полуоси = |λ_i|).
    3) V — обратный поворот. Результат = A · (окружность).
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True,
            show_basis_vectors=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 1.0], [0.5, 1.5]])
        U, S, Vt = np.linalg.svd(A)
        V = Vt.T
        Lambda = np.diag(S)

        # Матрицы на экране
        mat_A = m.Matrix(np.round(A, 2)).scale(0.55).to_corner(m.UP + m.LEFT, buff=0.5)
        mat_Vt = m.Matrix(np.round(Vt, 2)).scale(0.55).next_to(mat_A, m.RIGHT, buff=0.5)
        mat_Lam = m.Matrix(np.round(Lambda, 2)).scale(0.55).next_to(mat_Vt, m.RIGHT, buff=0.5)
        mat_V = m.Matrix(np.round(V, 2)).scale(0.55).next_to(mat_Lam, m.RIGHT, buff=0.5)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.15)
        eq2 = m.MathTex(r"\cdot").next_to(mat_Vt, m.RIGHT, buff=0.15)
        eq3 = m.MathTex(r"\cdot").next_to(mat_Lam, m.RIGHT, buff=0.15)
        self.add(mat_A, eq1, mat_Vt, eq2, mat_Lam, eq3, mat_V)

        # Окружность
        circle = m.Circle(radius=1, color=m.BLUE, stroke_width=2, fill_opacity=0.15)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # Базисные векторы
        i_vec = m.Vector([1, 0, 0], color=m.GREEN).set_opacity(0.8)
        j_vec = m.Vector([0, 1, 0], color=m.RED).set_opacity(0.8)
        self.add(i_vec, j_vec)
        self.add_transformable_mobject(i_vec, j_vec)

        # ШАГ 1: V^T
        label1 = m.Text("Шаг 1: V^T — поворот", font_size=24, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(label1)
        self.moving_mobjects = []
        self.apply_matrix(Vt)
        self.wait(0.5)
        self.remove(label1)

        # ШАГ 2: Λ
        label2 = m.Text(r"Шаг 2: Λ — растяжение (полуоси = $\sigma_1, \sigma_2$)",
                        font_size=24, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(label2)
        self.moving_mobjects = []
        self.apply_matrix(Lambda)
        self.wait(0.5)
        self.remove(label2)

        # ШАГ 3: V
        label3 = m.Text("Шаг 3: V — обратный поворот", font_size=24, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(label3)
        self.moving_mobjects = []
        self.apply_matrix(V)
        self.wait(0.5)
        self.remove(label3)

        final = m.Text("A = U Λ V^T   (SVD-разложение)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 3. LU-разложение: геометрический смысл
# ============================================================
class LUScene(LinearTransformationScene_):
    """
    A = L · U:
    U — верхняя треугольная (масштаб + сдвиг по x),
    L — нижняя треугольная (сдвиг по y).
    Показан пошаговый проход L затем U.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 1.0], [1.0, 1.5]])
        # LU-разложение вручную (без перестановок, т.к. ведущий элемент ≠ 0)
        L = np.array([[1.0, 0.0], [A[1, 0] / A[0, 0], 1.0]])
        U = np.array([[A[0, 0], A[0, 1]], [0, A[1, 1] - L[1, 0] * A[0, 1]]])

        # Матрицы на экране
        mat_A = m.Matrix(np.round(A, 2)).scale(0.6).to_corner(m.UP + m.LEFT, buff=0.5)
        mat_L = m.Matrix(np.round(L, 2)).scale(0.6).next_to(mat_A, m.RIGHT, buff=0.6)
        mat_U = m.Matrix(np.round(U, 2)).scale(0.6).next_to(mat_L, m.RIGHT, buff=0.6)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.2)
        eq2 = m.MathTex(r"\cdot").next_to(mat_L, m.RIGHT, buff=0.2)
        self.add(mat_A, eq1, mat_L, eq2, mat_U)

        # Сетка
        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        # Единичный квадрат для ориентира
        square = m.Polygon(
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            fill_color=m.PURPLE, fill_opacity=0.25, stroke_color=m.PURPLE
        )
        self.add(square)
        self.add_transformable_mobject(square)

        # ШАГ 1: U (масштаб+сдвиг по x)
        lbl1 = m.Text("Шаг 1: U — верхний треугольный сдвиг+масштаб",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1)
        self.moving_mobjects = []
        self.apply_matrix(U)
        self.wait(0.5)
        self.remove(lbl1)

        # ШАГ 2: L (сдвиг по y)
        lbl2 = m.Text("Шаг 2: L — нижний треугольный сдвиг",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2)
        self.moving_mobjects = []
        self.apply_matrix(L)
        self.wait(0.5)
        self.remove(lbl2)

        final = m.Text("A = L · U  (LU-разложение)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 4. LUP-разложение: геометрический смысл
# ============================================================
class LUPScene(LinearTransformationScene_):
    """
    A = P^T · L · U, где P — матрица перестановок.
    Первый шаг — перестановка осей, затем LU.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[0.5, 1.5], [2.0, 1.0]])  # требует перестановки
        # P — перестановка строк 1 и 2
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        PA = P @ A
        # LU уже для PA
        L = np.array([[1.0, 0.0], [PA[1, 0] / PA[0, 0], 1.0]])
        U = np.array([[PA[0, 0], PA[0, 1]], [0, PA[1, 1] - L[1, 0] * PA[0, 1]]])

        mat_A = m.Matrix(np.round(A, 2)).scale(0.55).to_corner(m.UP + m.LEFT, buff=0.4)
        mat_P = m.Matrix(np.round(P, 2)).scale(0.55).next_to(mat_A, m.RIGHT, buff=0.4)
        mat_L = m.Matrix(np.round(L, 2)).scale(0.55).next_to(mat_P, m.RIGHT, buff=0.4)
        mat_U = m.Matrix(np.round(U, 2)).scale(0.55).next_to(mat_L, m.RIGHT, buff=0.4)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.12)
        eq2 = m.MathTex(r"\cdot").next_to(mat_P, m.RIGHT, buff=0.12)
        eq3 = m.MathTex(r"\cdot").next_to(mat_L, m.RIGHT, buff=0.12)
        self.add(mat_A, eq1, mat_P, eq2, mat_L, eq3, mat_U)

        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        square = m.Polygon(
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            fill_color=m.PURPLE, fill_opacity=0.25, stroke_color=m.PURPLE
        )
        self.add(square)
        self.add_transformable_mobject(square)

        # ШАГ 1: P — перестановка (отражение относительно y = x)
        lbl1 = m.Text("Шаг 1: P — перестановка осей", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1)
        self.moving_mobjects = []
        self.apply_matrix(P)
        self.wait(0.5)
        self.remove(lbl1)

        # ШАГ 2: U
        lbl2 = m.Text("Шаг 2: U — верхний треугольный", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2)
        self.moving_mobjects = []
        self.apply_matrix(U)
        self.wait(0.5)
        self.remove(lbl2)

        # ШАГ 3: L
        lbl3 = m.Text("Шаг 3: L — нижний треугольный", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl3)
        self.moving_mobjects = []
        self.apply_matrix(L)
        self.wait(0.5)
        self.remove(lbl3)

        final = m.Text("A = Pᵀ · L · U  (LUP-разложение)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 5. QR-разложение: геометрический смысл
# ============================================================
class QRScene(LinearTransformationScene_):
    """
    A = Q · R:
    Q — ортогональная (поворот/отражение, форма не меняется),
    R — верхняя треугольная (масштаб + сдвиг).
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[2.0, 1.0], [0.5, 1.5]])
        Q, R = np.linalg.qr(A)

        mat_A = m.Matrix(np.round(A, 2)).scale(0.6).to_corner(m.UP + m.LEFT, buff=0.5)
        mat_Q = m.Matrix(np.round(Q, 2)).scale(0.6).next_to(mat_A, m.RIGHT, buff=0.6)
        mat_R = m.Matrix(np.round(R, 2)).scale(0.6).next_to(mat_Q, m.RIGHT, buff=0.6)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.2)
        eq2 = m.MathTex(r"\cdot").next_to(mat_Q, m.RIGHT, buff=0.2)
        self.add(mat_A, eq1, mat_Q, eq2, mat_R)

        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        circle = m.Circle(radius=1, color=m.PURPLE, stroke_width=2, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # ШАГ 1: R (масштаб + сдвиг)
        lbl1 = m.Text("Шаг 1: R — масштаб и сдвиг", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1)
        self.moving_mobjects = []
        self.apply_matrix(R)
        self.wait(0.5)
        self.remove(lbl1)

        # ШАГ 2: Q (ортогональное — поворот/отражение)
        lbl2 = m.Text("Шаг 2: Q — ортогональное (поворот)", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2)
        self.moving_mobjects = []
        self.apply_matrix(Q)
        self.wait(0.5)
        self.remove(lbl2)

        final = m.Text("A = Q · R  (QR-разложение)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 6. Разложение Холецкого: геометрический смысл
# ============================================================
class CholeskyScene(LinearTransformationScene_):
    """
    A = L · Lᵀ для симметричной положительно определённой A.
    L — нижняя треугольная (сдвиг + масштаб), Lᵀ — транспозиция.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        L = np.linalg.cholesky(A)
        Lt = L.T

        mat_A = m.Matrix(np.round(A, 2)).scale(0.6).to_corner(m.UP + m.LEFT, buff=0.5)
        mat_L = m.Matrix(np.round(L, 2)).scale(0.6).next_to(mat_A, m.RIGHT, buff=0.6)
        mat_Lt = m.Matrix(np.round(Lt, 2)).scale(0.6).next_to(mat_L, m.RIGHT, buff=0.6)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.2)
        eq2 = m.MathTex(r"\cdot").next_to(mat_L, m.RIGHT, buff=0.2)
        self.add(mat_A, eq1, mat_L, eq2, mat_Lt)

        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        circle = m.Circle(radius=1, color=m.PURPLE, stroke_width=2, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # ШАГ 1: Lᵀ (верхняя треугольная)
        lbl1 = m.Text("Шаг 1: Lᵀ — транспозиция нижней треугольной",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1)
        self.moving_mobjects = []
        self.apply_matrix(Lt)
        self.wait(0.5)
        self.remove(lbl1)

        # ШАГ 2: L
        lbl2 = m.Text("Шаг 2: L — нижняя треугольная",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2)
        self.moving_mobjects = []
        self.apply_matrix(L)
        self.wait(0.5)
        self.remove(lbl2)

        # Собственные векторы A (оси эллипса) — визуальный акцент
        eigvals, eigvecs = np.linalg.eigh(A)
        origin = self.plane.c2p(0, 0)
        for i in range(2):
            v = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2
            arrow = m.Arrow(origin, self.plane.c2p(v[0], v[1]),
                            color=m.YELLOW if i == 0 else m.GREEN,
                            buff=0, stroke_width=4)
            self.add(arrow)

        final = m.Text("A = L · Lᵀ  (разложение Холецкого)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 7. Сингулярное разложение (SVD): общая геометрическая интерпретация
# ============================================================
class SVDGeomScene(LinearTransformationScene_):
    """
    A = U Σ Vᵀ:
    Vᵀ — поворот → Σ — растяжение → U — поворот.
    Результат — тот же эллипс, что и A · окружность.
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[1.5, 0.5], [0.0, 2.0]])
        U, S, Vt = np.linalg.svd(A)
        V = Vt.T
        Sigma = np.diag(S)

        mat_A = m.Matrix(np.round(A, 2)).scale(0.55).to_corner(m.UP + m.LEFT, buff=0.4)
        mat_U = m.Matrix(np.round(U, 2)).scale(0.55).next_to(mat_A, m.RIGHT, buff=0.4)
        mat_S = m.Matrix(np.round(Sigma, 2)).scale(0.55).next_to(mat_U, m.RIGHT, buff=0.4)
        mat_Vt = m.Matrix(np.round(Vt, 2)).scale(0.55).next_to(mat_S, m.RIGHT, buff=0.4)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.12)
        eq2 = m.MathTex(r"\cdot").next_to(mat_U, m.RIGHT, buff=0.12)
        eq3 = m.MathTex(r"\cdot").next_to(mat_S, m.RIGHT, buff=0.12)
        self.add(mat_A, eq1, mat_U, eq2, mat_S, eq3, mat_Vt)

        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        circle = m.Circle(radius=1, color=m.PURPLE, stroke_width=2, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # ШАГ 1: Vᵀ
        lbl1 = m.Text("1) Vᵀ — поворот", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1); self.moving_mobjects = []; self.apply_matrix(Vt)
        self.wait(0.4); self.remove(lbl1)

        # ШАГ 2: Σ
        lbl2 = m.Text(r"2) Σ — растяжение (полуоси = $\sigma_i$)",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2); self.moving_mobjects = []; self.apply_matrix(Sigma)
        self.wait(0.4); self.remove(lbl2)

        # ШАГ 3: U
        lbl3 = m.Text("3) U — обратный поворот", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl3); self.moving_mobjects = []; self.apply_matrix(U)
        self.wait(0.4); self.remove(lbl3)

        # Сингулярные числа как полуоси
        origin = self.plane.c2p(0, 0)
        for i, color in enumerate([m.YELLOW, m.GREEN]):
            axis = np.zeros(2)
            axis[i] = S[i]
            # Направление оси в мировой системе после U
            direction = U @ axis
            arrow = m.Arrow(origin, self.plane.c2p(direction[0], direction[1]),
                            color=color, buff=0, stroke_width=4)
            self.add(arrow)
            label = m.MathTex(r"\sigma_{} = {:.2f}".format(i+1, S[i]),
                              color=color, font_size=26)
            label.next_to(arrow.get_end(), m.UR if i == 0 else m.DR)
            self.add(label)

        final = m.Text("A = U · Σ · Vᵀ  (SVD)", font_size=24, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


# ============================================================
# 8. Спектральное разложение симметричной матрицы
# ============================================================
class SpectralScene(LinearTransformationScene_):
    """
    A = V Λ Vᵀ для симметричной A.
    V — ортогональная (собственные векторы), Λ — диагональная (собственные числа).
    """
    def __init__(self, **kwargs):
        super().__init__(
            show_coordinates=True, show_basis_vectors=False, leave_ghost_vectors=False, **kwargs
        )

    def construct(self):
        A = np.array([[2.5, 1.0], [1.0, 1.5]])
        eigvals, V = np.linalg.eigh(A)
        # сортируем по убыванию
        idx = np.argsort(eigvals)[::-1]
        eigvals, V = eigvals[idx], V[:, idx]
        Lambda = np.diag(eigvals)
        Vt = V.T

        mat_A = m.Matrix(np.round(A, 2)).scale(0.55).to_corner(m.UP + m.LEFT, buff=0.4)
        mat_V = m.Matrix(np.round(V, 2)).scale(0.55).next_to(mat_A, m.RIGHT, buff=0.4)
        mat_L = m.Matrix(np.round(Lambda, 2)).scale(0.55).next_to(mat_V, m.RIGHT, buff=0.4)
        mat_Vt = m.Matrix(np.round(Vt, 2)).scale(0.55).next_to(mat_L, m.RIGHT, buff=0.4)
        eq1 = m.MathTex("=").next_to(mat_A, m.RIGHT, buff=0.12)
        eq2 = m.MathTex(r"\cdot").next_to(mat_V, m.RIGHT, buff=0.12)
        eq3 = m.MathTex(r"\cdot").next_to(mat_L, m.RIGHT, buff=0.12)
        self.add(mat_A, eq1, mat_V, eq2, mat_L, eq3, mat_Vt)

        plane = m.NumberPlane(
            x_range=[-3, 4, 1], y_range=[-3, 4, 1],
            background_line_style={"stroke_color": m.BLUE, "stroke_opacity": 0.5}
        ).shift(m.RIGHT * 0.5 + m.DOWN * 0.5)
        self.add(plane)
        self.add_transformable_mobject(plane)

        circle = m.Circle(radius=1, color=m.PURPLE, stroke_width=2, fill_opacity=0.2)
        self.add(circle)
        self.add_transformable_mobject(circle)

        # Собственные направления в исходной системе
        origin = self.plane.c2p(0, 0)
        for i, color in enumerate([m.RED, m.BLUE]):
            v = V[:, i] * 2.0
            self.add(m.DashedLine(
                self.plane.c2p(-v[0], -v[1]), self.plane.c2p(v[0], v[1]),
                color=color, stroke_width=2, stroke_opacity=0.5
            ))

        # ШАГ 1: Vᵀ
        lbl1 = m.Text("1) Vᵀ — переход в собственный базис", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl1); self.moving_mobjects = []; self.apply_matrix(Vt)
        self.wait(0.4); self.remove(lbl1)

        # ШАГ 2: Λ
        lbl2 = m.Text(r"2) Λ — растяжение по собственным осям",
                      font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl2); self.moving_mobjects = []; self.apply_matrix(Lambda)
        self.wait(0.4); self.remove(lbl2)

        # ШАГ 3: V
        lbl3 = m.Text("3) V — возврат в исходный базис", font_size=22, color=m.YELLOW).to_edge(m.DOWN, buff=0.6)
        self.add(lbl3); self.moving_mobjects = []; self.apply_matrix(V)
        self.wait(0.4); self.remove(lbl3)

        # Собственные векторы после (это оси эллипса)
        for i, color in enumerate([m.RED, m.BLUE]):
            v_after = A @ V[:, i]
            arrow = m.Arrow(origin, self.plane.c2p(v_after[0], v_after[1]),
                            color=color, buff=0, stroke_width=4)
            self.add(arrow)
            label = m.MathTex(r"\lambda_{} = {:.2f}".format(i+1, eigvals[i]),
                              color=color, font_size=26)
            label.next_to(arrow.get_end(), m.UR if i == 0 else m.DR)
            self.add(label)

        final = m.Text("A = V · Λ · Vᵀ  (спектральное разложение)",
                       font_size=22, color=m.GREEN).to_edge(m.DOWN, buff=0.6)
        self.add(final)
        self.wait(2)


class LowRankImageApprox(Scene_):
    """
    Изображение представлено как матрица.
    Последовательное восстановление рангами 1, 5, 20, 50.
    Рядом с каждым шагом — количество хранимых чисел.
    """
    def construct(self):
        # -------- 1. Генерация синтетического изображения --------
        size = 128
        img = self._make_synthetic_image(size)

        # -------- 2. SVD по каждому каналу --------
        channels = [img[:, :, c] for c in range(3)]
        svd_channels = [np.linalg.svd(ch, full_matrices=False) for ch in channels]

        # -------- 3. Заголовок и оригинал --------
        title = m.Text("Низкоранговая аппроксимация изображения (SVD)",
                       font_size=28, color=m.WHITE).to_edge(m.UP, buff=0.4)
        self.add(title)

        orig_mob = m.ImageMobject(self._to_uint8(img)).scale(2.4)
        orig_mob.to_corner(m.UP, buff=0.6).shift(m.DOWN*0.5)
        orig_label = m.Text("Оригинал", font_size=22, color=m.YELLOW)
        orig_label.next_to(orig_mob, m.DOWN, buff=0.15)
        n_orig = size * size * 3
        orig_count = m.Text(f"{n_orig} чисел", font_size=18, color=m.GRAY)
        orig_count.next_to(orig_label, m.DOWN, buff=0.05)

        self.play(m.FadeIn(orig_mob), m.Write(orig_label), m.Write(orig_count))
        self.wait(0.5)

        # -------- 4. Реконструкции для разных рангов --------
        ranks = [1, 3, 10, 30]
        images = []
        labels = []
        counts = []

        for r in ranks:
            approx = self._low_rank_reconstruct(svd_channels, r)
            mob = m.ImageMobject(self._to_uint8(approx)).scale(2.4)
            images.append(mob)
            # количество хранимых чисел: r*(m + n + 1) на каждый канал
            n_stored = 3 * r * (size + size + 1)
            labels.append(m.Text(f"Ранг {r}", font_size=22, color=m.YELLOW))
            counts.append(m.Text(f"{n_stored} чисел", font_size=18, color=m.GRAY))

        # Группируем по 4 картинки в ряд
        row = m.Group(*images).arrange(m.RIGHT, buff=0.6)
        row.next_to(orig_mob, m.DOWN, buff=0.7)
        # row.scale_to_fit_width(11)

        for i, (im, lbl, cnt) in enumerate(zip(images, labels, counts)):
            lbl.next_to(im, m.DOWN, buff=0.1)
            cnt.next_to(lbl, m.DOWN, buff=0.05)

        # Анимированное появление
        for im, lbl, cnt in zip(images, labels, counts):
            self.play(m.FadeIn(im), run_time=0.6)
            self.play(m.Write(lbl), m.Write(cnt), run_time=0.3)

        # -------- 5. Итоговая подпись --------
        note = m.Text(
            "Чем выше ранг, тем точнее восстановление, но больше хранимых чисел",
            font_size=24, color=m.YELLOW
        ).to_edge(m.DOWN, buff=0.3)
        self.play(m.Create(note))
        self.wait(3)

    # ---------- вспомогательные методы ----------
    def _make_synthetic_image(self, size):
        """Простая узнаваемая сцена: небо, солнце, горы."""
        img = np.zeros((size, size, 3))
        yy, xx = np.mgrid[0:size, 0:size]

        # Небо — вертикальный градиент
        for i in range(size):
            img[i, :, 0] = 0.20 + 0.55 * i / size
            img[i, :, 1] = 0.40 + 0.40 * i / size
            img[i, :, 2] = 0.85

        # Солнце
        cy, cx = size // 4, size // 2
        r_sun = size // 8
        sun_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r_sun ** 2
        img[sun_mask] = [1.0, 0.9, 0.2]

        # Гора 1 (правее и ниже)
        for i in range(size):
            for j in range(size):
                if i > 2 * size // 3 and abs(j - size // 3) < (i - 2 * size // 3) * 0.9:
                    img[i, j] = [0.10, 0.30, 0.10]
                if i > size // 2 and abs(j - 2 * size // 3) < (i - size // 2) * 0.7:
                    img[i, j] = [0.05, 0.20, 0.05]

        return np.clip(img, 0, 1)

    def _low_rank_reconstruct(self, svd_channels, r):
        """Собираем приближение ранга r по всем каналам."""
        out = np.zeros((svd_channels[0][0].shape[0],
                        svd_channels[0][2].shape[1], 3))
        for c, (U, S, Vt) in enumerate(svd_channels):
            approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
            out[:, :, c] = approx
        return np.clip(out, 0, 1)

    def _to_uint8(self, arr):
        return np.uint8(np.clip(arr, 0, 1) * 255)


# ============================================================
# 2. Площадь под кривой нормального распределения
# ============================================================
class NormalDistributionArea(Scene_):
    """
    Кривая нормального распределения; закрашивается площадь под кривой
    от a до b, рядом — значение интеграла.
    Интервал плавно сдвигается и сжимается, площадь меняется.
    """
    def construct(self):
        # ---------- Оси и кривая ----------
        axes = m.Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.5, 0.1],
            x_length=10,
            y_length=4.5,
            axis_config={"color": m.GRAY}
        )
        axes.add_coordinates()
        self.add(axes)

        mu, sigma = 0.0, 1.0

        def pdf(x):
            return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

        curve = axes.plot(pdf, x_range=[-5, 5], color=m.BLUE, stroke_width=3)
        curve_label = m.MathTex(r"f(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}",
                                color=m.BLUE, font_size=24)
        curve_label.next_to(curve.get_end(), m.UR)
        self.play(m.Create(curve), m.Write(curve_label), run_time=1.5)

        # ---------- Трекеры границ ----------
        a_tracker = m.ValueTracker(-1.5)
        b_tracker = m.ValueTracker(1.5)

        # ---------- Закрашенная площадь ----------
        def make_area():
            return axes.get_area(
                curve,
                x_range=[a_tracker.get_value(), b_tracker.get_value()],
                color=m.YELLOW,
                opacity=0.5
            )
        area = m.always_redraw(make_area)
        self.add(area)

        # ---------- Вертикальные границы a и b ----------
        def vert_line(tracker):
            return m.always_redraw(
                lambda: axes.get_vertical_line(
                    axes.input_to_graph_point(tracker.get_value(), curve),
                    color=m.RED,
                    stroke_width=3
                )
            )
        line_a = vert_line(a_tracker)
        line_b = vert_line(b_tracker)
        self.add(line_a, line_b)

        # Подписи a и b
        label_a = m.always_redraw(lambda: m.MathTex("a", color=m.RED, font_size=28)
                                  .next_to(axes.c2p(a_tracker.get_value(), 0), m.DOWN, buff=0.2))
        label_b = m.always_redraw(lambda: m.MathTex("b", color=m.RED, font_size=28)
                                  .next_to(axes.c2p(b_tracker.get_value(), 0), m.DOWN, buff=0.2))
        self.add(label_a, label_b)

        # ---------- Значение интеграла ----------
        def integral_value():
            a = a_tracker.get_value()
            b = b_tracker.get_value()
            cdf = lambda x: 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))
            return cdf(b) - cdf(a)

        value = m.DecimalNumber(integral_value(), num_decimal_places=3,
                                color=m.YELLOW, font_size=36)
        value.add_updater(lambda d: d.set_value(integral_value()))

        formula_label = m.MathTex(r"P(a \leq X \leq b) = \int_a^b f(x)\,dx \approx ",
                                  font_size=26, color=m.WHITE)
        value_group = m.VGroup(formula_label, value).arrange(m.RIGHT, buff=0.15)
        value_group.to_corner(m.UP + m.RIGHT, buff=0.5)
        self.add(value_group)

        # ---------- Заголовок и пояснение ----------
        title = m.Text("Площадь под кривой нормального распределения",
                       font_size=24, color=m.WHITE).to_edge(m.UP, buff=0.3)
        self.add(title)
        note = m.Text("Площадь = вероятность попадания в интервал [a, b]",
                      font_size=20, color=m.YELLOW).to_edge(m.DOWN, buff=0.4)
        self.add(note)

        self.wait(0.5)

        # ---------- Анимация сдвига и сжатия интервала ----------
        # Широкий интервал
        self.play(
            a_tracker.animate.set_value(-3.0),
            b_tracker.animate.set_value(3.0),
            run_time=2
        )
        self.wait(0.5)

        # Узкий симметричный
        self.play(
            a_tracker.animate.set_value(-0.5),
            b_tracker.animate.set_value(0.5),
            run_time=2
        )
        self.wait(0.5)

        # Сдвиг вправо
        self.play(
            a_tracker.animate.set_value(0.5),
            b_tracker.animate.set_value(2.5),
            run_time=2
        )
        self.wait(0.5)

        # Сдвиг влево (в хвост распределения)
        self.play(
            a_tracker.animate.set_value(-4.0),
            b_tracker.animate.set_value(-2.0),
            run_time=2
        )
        self.wait(0.5)

        # Финальный кадр — широкий центральный интервал
        self.play(
            a_tracker.animate.set_value(-1.96),
            b_tracker.animate.set_value(1.96),
            run_time=2
        )
        self.wait(3)

if __name__ == '__main__':
    import os
    from pathlib import Path
 
    SCENES = [
        # "SpectralDecompositionScene",
        # "SVD",
        # "Rotation3DZ",
        # "MatrixDecomposition",
        # "HandPositiveDefinite",
        # "CholeskyEllipse",
        # "OrthogonalVsArbitrary",
        # "EigenEllipse",
        # "SVDCircleSteps",
        # "SpectralClustering",
        # "LowRankApprox",
        # "PCAPlot3D",
        # "MatrixDecompTree",
        # "FunctionMapping",
        # "EigenEllipse",
        # "SVDCircleSteps",
        # "LUScene",
        # "LUPScene",
        # "QRScene",
        # "CholeskyScene",
        # "SVDGeomScene",
        # "SpectralScene",
        "LowRankImageApprox",
        # "NormalDistributionArea",
    ]
    file_path = Path(__file__).resolve()

    for SCENE in SCENES:
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -qh")
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -sqh")