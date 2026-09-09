import manim as m
import numpy as np
from math import sin, ceil
from theming import LinearTransformationScene_, ThreeDScene_

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



if __name__ == '__main__':
    import os
    from pathlib import Path
 
    SCENES = [
        # "SpectralDecompositionScene",
        # "SVD"
    ]
    file_path = Path(__file__).resolve()

    for SCENE in SCENES:
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -qh")
        os.system(f"manim {Path(__file__).resolve()} {SCENE} -s")